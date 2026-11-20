//! Translation of decoded Maxwell shaders to SPIR-V for Vulkan.
//!
//! Registers become `uint` variables (instructions mix integer and float
//! views of them; floats are bitcast). Attribute space maps to Vulkan
//! interface variables: generic attribute/varying `i` is location `i`
//! (a `vec4`), the VS position output is `BuiltIn Position`, the PS
//! position input is `BuiltIn FragCoord`. Pixel shader outputs are the
//! registers named by the header's output map, written at the end.
//!
//! Constant buffer `i` is a uniform block `{ uvec4 data[4096]; }` at
//! descriptor set 0, binding [`cbuf_binding`]`(stage, i)`.
//!
//! Control flow becomes a loop around a switch over the program's basic
//! blocks, with the block to run next in a variable; SSY targets live on
//! a small private stack, so SYNC can jump to them. Predicates are
//! `bool` variables. Anything unsupported returns an error so the caller
//! can fall back to the software renderer.

use super::decode::{
    Bop, FCmp, ICmp, Inst, IpaMode, LopOp, Mods, MufuOp, Op, Operand, Pred, Round, XmadMode, PT, RZ,
};
use super::interp::{SSY_DEPTH, STEP_LIMIT};
use super::{attr, Header, Stage};
use std::collections::{BTreeMap, BTreeSet};

mod op {
    pub const NAME: u16 = 5;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT: u16 = 43;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_STRUCT: u16 = 30;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const BITCAST: u16 = 124;
    pub const F_NEGATE: u16 = 127;
    pub const F_ADD: u16 = 129;
    pub const F_MUL: u16 = 133;
    pub const LABEL: u16 = 248;
    pub const RETURN: u16 = 253;
    pub const TYPE_BOOL: u16 = 20;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT_NULL: u16 = 46;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const S_NEGATE: u16 = 126;
    pub const I_ADD: u16 = 128;
    pub const I_SUB: u16 = 130;
    pub const I_MUL: u16 = 132;
    pub const F_DIV: u16 = 136;
    pub const IS_NAN: u16 = 156;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_EQUAL: u16 = 181;
    pub const F_ORD_NOT_EQUAL: u16 = 182;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_UNORD_LESS_THAN: u16 = 185;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_UNORD_GREATER_THAN: u16 = 187;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_UNORD_LESS_THAN_EQUAL: u16 = 189;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const F_UNORD_GREATER_THAN_EQUAL: u16 = 191;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const BIT_FIELD_S_EXTRACT: u16 = 202;
    pub const BIT_FIELD_U_EXTRACT: u16 = 203;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
    pub const KILL: u16 = 252;
    pub const TYPE_IMAGE: u16 = 25;
    pub const TYPE_SAMPLED_IMAGE: u16 = 27;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
}

// Storage classes, decorations, built-ins, GLSL.std.450 instructions.
const UNIFORM_CONSTANT: u32 = 0;
const INPUT: u32 = 1;
const OUTPUT: u32 = 3;
const PRIVATE: u32 = 6;
const LOCATION: u32 = 30;
const UNIFORM: u32 = 2;
const BLOCK: u32 = 2;
const ARRAY_STRIDE: u32 = 6;
const OFFSET: u32 = 35;
const BINDING: u32 = 33;
const DESCRIPTOR_SET: u32 = 34;
const BUILTIN: u32 = 11;
const FLAT: u32 = 14;
const BUILTIN_POSITION: u32 = 0;
const BUILTIN_FRAG_COORD: u32 = 15;
const GLSL_FABS: u32 = 4;
const GLSL_FCLAMP: u32 = 43;
const GLSL_FMA: u32 = 50;
const GLSL_ROUND_EVEN: u32 = 2;
const GLSL_TRUNC: u32 = 3;
const GLSL_SABS: u32 = 5;
const GLSL_FLOOR: u32 = 8;
const GLSL_CEIL: u32 = 9;
const GLSL_SIN: u32 = 13;
const GLSL_COS: u32 = 14;
const GLSL_EXP2: u32 = 29;
const GLSL_LOG2: u32 = 30;
const GLSL_SQRT: u32 = 31;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_UMIN: u32 = 38;
const GLSL_SMIN: u32 = 39;
const GLSL_UMAX: u32 = 41;
const GLSL_SMAX: u32 = 42;
const GLSL_NMIN: u32 = 79;
const GLSL_NMAX: u32 = 80;

#[derive(Default)]
struct Module {
    next_id: u32,
    /// Sections in SPIR-V's required order.
    preamble: Vec<u32>,
    debug: Vec<u32>,
    annotations: Vec<u32>,
    globals: Vec<u32>,
    body: Vec<u32>,
}

fn inst(out: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    out.push(((operands.len() as u32 + 1) << 16) | u32::from(opcode));
    out.extend_from_slice(operands);
}

fn string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl Module {
    fn id(&mut self) -> u32 {
        self.next_id += 1;
        self.next_id
    }
}

/// Ids of the types and constants every shader uses.
struct Common {
    glsl: u32,
    float: u32,
    uint: u32,
    vec4: u32,
    zero_u: u32,
    zero_f: u32,
    one_f: u32,
    p_private_uint: u32,
    p_in_vec4: u32,
    p_out_vec4: u32,
    p_in_float: u32,
    p_out_float: u32,
    component: [u32; 4],
    p_uniform_cbuf: u32,
    p_uniform_uint: u32,
    bool_t: u32,
    true_c: u32,
    false_c: u32,
    p_private_bool: u32,
    p_private_stack: u32,
    null_stack: u32,
    vec2: u32,
    sampled_image: u32,
    p_sampled_image: u32,
}

/// Descriptor binding of constant buffer `index` for a stage.
pub fn cbuf_binding(stage: Stage, index: u32) -> u32 {
    match stage {
        Stage::Pixel => 18 + index,
        _ => index,
    }
}

/// Most textures a stage may sample.
pub const MAX_TEXTURES: usize = 16;

/// Descriptor binding of the `i`-th texture a stage samples (a combined
/// image sampler).
pub fn texture_binding(stage: Stage, i: u32) -> u32 {
    match stage {
        Stage::Pixel => 36 + MAX_TEXTURES as u32 + i,
        _ => 36 + i,
    }
}

/// A translated shader.
pub struct Translated {
    pub words: Vec<u32>,
    /// Bit `i` set: constant buffer `i` is read.
    pub cbufs: u32,
    /// Handles (bindless constant buffer word offsets) of the textures
    /// sampled; the `i`-th is at [`texture_binding`]`(stage, i)`.
    pub textures: Vec<u32>,
}

struct Translator {
    m: Module,
    c: Common,
    regs: BTreeMap<u8, u32>,
    inputs: BTreeMap<u32, u32>,
    outputs: BTreeMap<u32, u32>,
    flat: BTreeSet<u32>,
    cbufs: BTreeMap<u8, u32>,
    preds: BTreeMap<u8, u32>,
    /// Texture variables by handle, in binding order.
    textures: Vec<(u32, u32)>,
    stage: Stage,
}

impl Translator {
    fn reg_var(&mut self, r: u8) -> u32 {
        if let Some(&v) = self.regs.get(&r) {
            return v;
        }
        let v = self.m.id();
        let (p, zero) = (self.c.p_private_uint, self.c.zero_u);
        inst(&mut self.m.globals, op::VARIABLE, &[p, v, PRIVATE, zero]);
        self.regs.insert(r, v);
        v
    }

    fn load_reg(&mut self, r: u8) -> u32 {
        if r == RZ {
            return self.c.zero_u;
        }
        let var = self.reg_var(r);
        let id = self.m.id();
        inst(&mut self.m.body, op::LOAD, &[self.c.uint, id, var]);
        id
    }

    fn store_reg(&mut self, r: u8, value_uint: u32) {
        if r != RZ {
            let var = self.reg_var(r);
            inst(&mut self.m.body, op::STORE, &[var, value_uint]);
        }
    }

    fn constant_u(&mut self, v: u32) -> u32 {
        let id = self.m.id();
        inst(&mut self.m.globals, op::CONSTANT, &[self.c.uint, id, v]);
        id
    }

    fn operand_uint(&mut self, o: Operand) -> Result<u32, String> {
        match o {
            Operand::Reg(r) => Ok(self.load_reg(r)),
            Operand::Imm(v) => Ok(self.constant_u(v)),
            Operand::Cbuf { index, offset } => {
                if offset >= 0x1_0000 || !offset.is_multiple_of(4) {
                    return Err(format!("constant buffer offset {offset:#x}"));
                }
                let var = self.cbuf_var(index);
                let (zero, vec, comp) = (
                    self.c.zero_u,
                    self.constant_u(offset / 16),
                    self.c.component[((offset / 4) % 4) as usize],
                );
                let p = self.m.id();
                inst(
                    &mut self.m.body,
                    op::ACCESS_CHAIN,
                    &[self.c.p_uniform_uint, p, var, zero, vec, comp],
                );
                let v = self.m.id();
                inst(&mut self.m.body, op::LOAD, &[self.c.uint, v, p]);
                Ok(v)
            }
        }
    }

    fn cbuf_var(&mut self, index: u8) -> u32 {
        if let Some(&v) = self.cbufs.get(&index) {
            return v;
        }
        let v = self.m.id();
        inst(
            &mut self.m.globals,
            op::VARIABLE,
            &[self.c.p_uniform_cbuf, v, UNIFORM],
        );
        let binding = cbuf_binding(self.stage, u32::from(index));
        inst(
            &mut self.m.annotations,
            op::DECORATE,
            &[v, DESCRIPTOR_SET, 0],
        );
        inst(
            &mut self.m.annotations,
            op::DECORATE,
            &[v, BINDING, binding],
        );
        self.cbufs.insert(index, v);
        v
    }

    fn bitcast(&mut self, ty: u32, v: u32) -> u32 {
        let id = self.m.id();
        inst(&mut self.m.body, op::BITCAST, &[ty, id, v]);
        id
    }

    fn bitcast_float(&mut self, v: u32) -> u32 {
        self.bitcast(self.c.float, v)
    }

    fn glsl(&mut self, ty: u32, f: u32, args: &[u32]) -> u32 {
        let id = self.m.id();
        let mut ops = vec![ty, id, self.c.glsl, f];
        ops.extend_from_slice(args);
        inst(&mut self.m.body, op::EXT_INST, &ops);
        id
    }

    fn mods(&mut self, v: u32, m: Mods) -> u32 {
        let v = if m.abs {
            self.glsl(self.c.float, GLSL_FABS, &[v])
        } else {
            v
        };
        if m.neg {
            let id = self.m.id();
            inst(&mut self.m.body, op::F_NEGATE, &[self.c.float, id, v]);
            id
        } else {
            v
        }
    }

    fn binary(&mut self, opcode: u16, a: u32, b: u32) -> u32 {
        let id = self.m.id();
        inst(&mut self.m.body, opcode, &[self.c.float, id, a, b]);
        id
    }

    fn store_float(&mut self, dst: u8, f: u32, sat: bool) {
        let f = if sat {
            let (z, o) = (self.c.zero_f, self.c.one_f);
            self.glsl(self.c.float, GLSL_FCLAMP, &[f, z, o])
        } else {
            f
        };
        let u = self.bitcast(self.c.uint, f);
        self.store_reg(dst, u);
    }

    /// The interface variable for attribute address `addr` (a vec4).
    fn interface(&mut self, addr: u32, output: bool, stage: Stage) -> Result<u32, String> {
        let slot = addr & !0xF;
        let map = if output { &self.outputs } else { &self.inputs };
        if let Some(&v) = map.get(&slot) {
            return Ok(v);
        }
        let v = self.m.id();
        let ptr = if output {
            self.c.p_out_vec4
        } else {
            self.c.p_in_vec4
        };
        inst(
            &mut self.m.globals,
            op::VARIABLE,
            &[ptr, v, if output { OUTPUT } else { INPUT }],
        );
        if slot == attr::POSITION {
            let builtin = match (stage, output) {
                (Stage::Vertex, true) => BUILTIN_POSITION,
                (Stage::Pixel, false) => BUILTIN_FRAG_COORD,
                _ => return Err(format!("attribute {addr:#x} not supported here")),
            };
            inst(
                &mut self.m.annotations,
                op::DECORATE,
                &[v, BUILTIN, builtin],
            );
        } else if (attr::generic(0)..attr::generic(32)).contains(&slot) {
            inst(
                &mut self.m.annotations,
                op::DECORATE,
                &[v, LOCATION, (slot - attr::generic(0)) / 16],
            );
        } else {
            return Err(format!("attribute {addr:#x} not supported"));
        }
        if output {
            self.outputs.insert(slot, v);
        } else {
            self.inputs.insert(slot, v);
        }
        Ok(v)
    }

    fn component_ptr(&mut self, var: u32, addr: u32, output: bool) -> u32 {
        let id = self.m.id();
        let ptr = if output {
            self.c.p_out_float
        } else {
            self.c.p_in_float
        };
        let comp = self.c.component[((addr / 4) % 4) as usize];
        inst(&mut self.m.body, op::ACCESS_CHAIN, &[ptr, id, var, comp]);
        id
    }

    fn op1(&mut self, opcode: u16, ty: u32, a: u32) -> u32 {
        let id = self.m.id();
        inst(&mut self.m.body, opcode, &[ty, id, a]);
        id
    }

    fn op2(&mut self, opcode: u16, ty: u32, a: u32, b: u32) -> u32 {
        let id = self.m.id();
        inst(&mut self.m.body, opcode, &[ty, id, a, b]);
        id
    }

    fn select(&mut self, ty: u32, cond: u32, a: u32, b: u32) -> u32 {
        let id = self.m.id();
        inst(&mut self.m.body, op::SELECT, &[ty, id, cond, a, b]);
        id
    }

    fn constant_f(&mut self, v: f32) -> u32 {
        let id = self.m.id();
        inst(
            &mut self.m.globals,
            op::CONSTANT,
            &[self.c.float, id, v.to_bits()],
        );
        id
    }

    fn label(&mut self, id: u32) {
        inst(&mut self.m.body, op::LABEL, &[id]);
    }

    fn branch(&mut self, to: u32) {
        inst(&mut self.m.body, op::BRANCH, &[to]);
    }

    fn private_var(&mut self, ptr: u32, init: u32) -> u32 {
        let v = self.m.id();
        inst(&mut self.m.globals, op::VARIABLE, &[ptr, v, PRIVATE, init]);
        v
    }

    fn load(&mut self, ty: u32, var: u32) -> u32 {
        let id = self.m.id();
        inst(&mut self.m.body, op::LOAD, &[ty, id, var]);
        id
    }

    fn store(&mut self, var: u32, v: u32) {
        inst(&mut self.m.body, op::STORE, &[var, v]);
    }

    fn pred_var(&mut self, i: u8) -> u32 {
        if let Some(&v) = self.preds.get(&i) {
            return v;
        }
        let (p, f) = (self.c.p_private_bool, self.c.false_c);
        let v = self.private_var(p, f);
        self.preds.insert(i, v);
        v
    }

    fn load_pred(&mut self, p: Pred) -> u32 {
        let v = if p.index == 7 {
            self.c.true_c
        } else {
            let var = self.pred_var(p.index);
            self.load(self.c.bool_t, var)
        };
        if p.neg {
            self.op1(op::LOGICAL_NOT, self.c.bool_t, v)
        } else {
            v
        }
    }

    fn store_pred(&mut self, i: u8, v: u32) {
        if i != 7 {
            let var = self.pred_var(i);
            self.store(var, v);
        }
    }

    fn operand_float(&mut self, o: Operand, m: Mods) -> Result<u32, String> {
        let u = self.operand_uint(o)?;
        let f = self.bitcast_float(u);
        Ok(self.mods(f, m))
    }

    fn reg_float(&mut self, r: u8, m: Mods) -> u32 {
        let u = self.load_reg(r);
        let f = self.bitcast_float(u);
        self.mods(f, m)
    }

    fn is_nan(&mut self, f: u32) -> u32 {
        self.op1(op::IS_NAN, self.c.bool_t, f)
    }

    fn fcmp(&mut self, cmp: FCmp, x: u32, y: u32) -> u32 {
        let b = self.c.bool_t;
        let opcode = match cmp & 15 {
            0 => return self.c.false_c,
            15 => return self.c.true_c,
            7 | 8 => {
                let (nx, ny) = (self.is_nan(x), self.is_nan(y));
                let nan = self.op2(op::LOGICAL_OR, b, nx, ny);
                return if cmp == 8 {
                    nan
                } else {
                    self.op1(op::LOGICAL_NOT, b, nan)
                };
            }
            1 => op::F_ORD_LESS_THAN,
            2 => op::F_ORD_EQUAL,
            3 => op::F_ORD_LESS_THAN_EQUAL,
            4 => op::F_ORD_GREATER_THAN,
            5 => op::F_ORD_NOT_EQUAL,
            6 => op::F_ORD_GREATER_THAN_EQUAL,
            9 => op::F_UNORD_LESS_THAN,
            10 => op::F_UNORD_EQUAL,
            11 => op::F_UNORD_LESS_THAN_EQUAL,
            12 => op::F_UNORD_GREATER_THAN,
            13 => op::F_UNORD_NOT_EQUAL,
            _ => op::F_UNORD_GREATER_THAN_EQUAL,
        };
        self.op2(opcode, b, x, y)
    }

    fn icmp(&mut self, cmp: ICmp, signed: bool, x: u32, y: u32) -> u32 {
        let opcode = match (cmp & 7, signed) {
            (0, _) => return self.c.false_c,
            (7, _) => return self.c.true_c,
            (2, _) => op::I_EQUAL,
            (5, _) => op::I_NOT_EQUAL,
            (1, false) => op::U_LESS_THAN,
            (1, true) => op::S_LESS_THAN,
            (3, false) => op::U_LESS_THAN_EQUAL,
            (3, true) => op::S_LESS_THAN_EQUAL,
            (4, false) => op::U_GREATER_THAN,
            (4, true) => op::S_GREATER_THAN,
            (_, false) => op::U_GREATER_THAN_EQUAL,
            (_, true) => op::S_GREATER_THAN_EQUAL,
        };
        self.op2(opcode, self.c.bool_t, x, y)
    }

    fn bop(&mut self, o: Bop, x: u32, y: u32) -> u32 {
        let opcode = match o {
            Bop::And => op::LOGICAL_AND,
            Bop::Or => op::LOGICAL_OR,
            Bop::Xor => op::LOGICAL_NOT_EQUAL,
        };
        self.op2(opcode, self.c.bool_t, x, y)
    }

    /// Writes `r bop c` and `!r bop c` to two predicates.
    fn set_preds(&mut self, dst: u8, dst2: u8, r: u32, o: Bop, c: Pred) {
        let c = self.load_pred(c);
        let v = self.bop(o, r, c);
        self.store_pred(dst, v);
        if dst2 != 7 {
            let nr = self.op1(op::LOGICAL_NOT, self.c.bool_t, r);
            let v2 = self.bop(o, nr, c);
            self.store_pred(dst2, v2);
        }
    }

    fn set_value(&mut self, cond: u32, bf: bool) -> u32 {
        let t = self.constant_u(if bf { 1.0f32.to_bits() } else { u32::MAX });
        self.select(self.c.uint, cond, t, self.c.zero_u)
    }

    fn negate_int(&mut self, v: u32, on: bool) -> u32 {
        if on {
            self.op1(op::S_NEGATE, self.c.uint, v)
        } else {
            v
        }
    }

    fn round(&mut self, f: u32, r: Round) -> u32 {
        let g = match r {
            Round::Pass => return f,
            Round::Nearest => GLSL_ROUND_EVEN,
            Round::Floor => GLSL_FLOOR,
            Round::Ceil => GLSL_CEIL,
            Round::Trunc => GLSL_TRUNC,
        };
        self.glsl(self.c.float, g, &[f])
    }

    fn half(&mut self, v: u32, hi: bool, signed: bool) -> u32 {
        let off = self.c.component[0];
        let off = if hi { self.constant_u(16) } else { off };
        let count = self.constant_u(16);
        let id = self.m.id();
        let opcode = if signed {
            op::BIT_FIELD_S_EXTRACT
        } else {
            op::BIT_FIELD_U_EXTRACT
        };
        inst(&mut self.m.body, opcode, &[self.c.uint, id, v, off, count]);
        id
    }

    /// Emits `f` so that it only runs where predicate `p` holds.
    fn guarded(
        &mut self,
        p: Pred,
        f: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        if p == PT {
            return f(self);
        }
        let cond = self.load_pred(p);
        let (then, merge) = (self.m.id(), self.m.id());
        inst(&mut self.m.body, op::SELECTION_MERGE, &[merge, 0]);
        inst(
            &mut self.m.body,
            op::BRANCH_CONDITIONAL,
            &[cond, then, merge],
        );
        self.label(then);
        f(self)?;
        self.branch(merge);
        self.label(merge);
        Ok(())
    }

    fn texture_var(&mut self, handle: u32) -> Result<u32, String> {
        if let Some(&(_, v)) = self.textures.iter().find(|t| t.0 == handle) {
            return Ok(v);
        }
        if self.textures.len() == MAX_TEXTURES {
            return Err("too many textures".into());
        }
        let v = self.m.id();
        inst(
            &mut self.m.globals,
            op::VARIABLE,
            &[self.c.p_sampled_image, v, UNIFORM_CONSTANT],
        );
        let binding = texture_binding(self.stage, self.textures.len() as u32);
        inst(
            &mut self.m.annotations,
            op::DECORATE,
            &[v, DESCRIPTOR_SET, 0],
        );
        inst(
            &mut self.m.annotations,
            op::DECORATE,
            &[v, BINDING, binding],
        );
        self.textures.push((handle, v));
        Ok(v)
    }

    /// Samples level 0 of a texture; returns the RGBA components (floats).
    fn sample(&mut self, handle: u32, u: u32, v: u32) -> Result<[u32; 4], String> {
        let var = self.texture_var(handle)?;
        let img = self.load(self.c.sampled_image, var);
        let coord = self.m.id();
        inst(
            &mut self.m.body,
            op::COMPOSITE_CONSTRUCT,
            &[self.c.vec2, coord, u, v],
        );
        let texel = self.m.id();
        // Image operand Lod (0x2) = 0.
        inst(
            &mut self.m.body,
            op::IMAGE_SAMPLE_EXPLICIT_LOD,
            &[self.c.vec4, texel, img, coord, 0x2, self.c.zero_f],
        );
        Ok([0, 1, 2, 3].map(|k| {
            let id = self.m.id();
            inst(
                &mut self.m.body,
                op::COMPOSITE_EXTRACT,
                &[self.c.float, id, texel, k],
            );
            id
        }))
    }

    /// Translates a non-control-flow instruction.
    fn exec(&mut self, i: Op, stage: Stage) -> Result<(), String> {
        let no = Mods::default();
        let neg = |v: bool| Mods { neg: v, abs: false };
        let (uint, float) = (self.c.uint, self.c.float);
        match i {
            Op::Mov { dst, src } => {
                let v = self.operand_uint(src)?;
                self.store_reg(dst, v);
            }
            Op::Fadd {
                dst,
                a,
                a_mods,
                b,
                b_mods,
                sat,
            } => {
                let x = self.reg_float(a, a_mods);
                let y = self.operand_float(b, b_mods)?;
                let r = self.binary(op::F_ADD, x, y);
                self.store_float(dst, r, sat);
            }
            Op::Fmul {
                dst,
                a,
                b,
                neg_b,
                sat,
            } => {
                let x = self.reg_float(a, no);
                let y = self.operand_float(b, neg(neg_b))?;
                let r = self.binary(op::F_MUL, x, y);
                self.store_float(dst, r, sat);
            }
            Op::Ffma {
                dst,
                a,
                b,
                neg_b,
                c,
                neg_c,
                sat,
            } => {
                let x = self.reg_float(a, no);
                let y = self.operand_float(b, neg(neg_b))?;
                let z = self.operand_float(c, neg(neg_c))?;
                let r = self.glsl(float, GLSL_FMA, &[x, y, z]);
                self.store_float(dst, r, sat);
            }
            Op::Fmnmx {
                dst,
                a,
                a_mods,
                b,
                b_mods,
                min,
            } => {
                let x = self.reg_float(a, a_mods);
                let y = self.operand_float(b, b_mods)?;
                let lo = self.glsl(float, GLSL_NMIN, &[x, y]);
                let hi = self.glsl(float, GLSL_NMAX, &[x, y]);
                let c = self.load_pred(min);
                let r = self.select(float, c, lo, hi);
                self.store_float(dst, r, false);
            }
            Op::Mufu {
                dst,
                func,
                a,
                mods,
                sat,
            } => {
                let x = self.reg_float(a, mods);
                let r = match func {
                    MufuOp::Cos => self.glsl(float, GLSL_COS, &[x]),
                    MufuOp::Sin => self.glsl(float, GLSL_SIN, &[x]),
                    MufuOp::Ex2 => self.glsl(float, GLSL_EXP2, &[x]),
                    MufuOp::Lg2 => self.glsl(float, GLSL_LOG2, &[x]),
                    MufuOp::Rsq => self.glsl(float, GLSL_INVERSE_SQRT, &[x]),
                    MufuOp::Sqrt => self.glsl(float, GLSL_SQRT, &[x]),
                    MufuOp::Rcp => {
                        let one = self.c.one_f;
                        self.binary(op::F_DIV, one, x)
                    }
                };
                self.store_float(dst, r, sat);
            }
            Op::Rro { dst, src, mods } => {
                let r = self.operand_float(src, mods)?;
                self.store_float(dst, r, false);
            }
            Op::Fsetp {
                dst,
                dst2,
                cmp,
                a,
                a_mods,
                b,
                b_mods,
                bop,
                c,
            } => {
                let x = self.reg_float(a, a_mods);
                let y = self.operand_float(b, b_mods)?;
                let r = self.fcmp(cmp, x, y);
                self.set_preds(dst, dst2, r, bop, c);
            }
            Op::Fset {
                dst,
                cmp,
                a,
                a_mods,
                b,
                b_mods,
                bop,
                c,
                bf,
            } => {
                let x = self.reg_float(a, a_mods);
                let y = self.operand_float(b, b_mods)?;
                let r = self.fcmp(cmp, x, y);
                let c = self.load_pred(c);
                let r = self.bop(bop, r, c);
                let v = self.set_value(r, bf);
                self.store_reg(dst, v);
            }
            Op::Isetp {
                dst,
                dst2,
                cmp,
                signed,
                a,
                b,
                bop,
                c,
            } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let r = self.icmp(cmp, signed, x, y);
                self.set_preds(dst, dst2, r, bop, c);
            }
            Op::Iset {
                dst,
                cmp,
                signed,
                a,
                b,
                bop,
                c,
                bf,
            } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let r = self.icmp(cmp, signed, x, y);
                let c = self.load_pred(c);
                let r = self.bop(bop, r, c);
                let v = self.set_value(r, bf);
                self.store_reg(dst, v);
            }
            Op::Psetp {
                dst,
                dst2,
                op1,
                op2,
                a,
                b,
                c,
            } => {
                let x = self.load_pred(a);
                let y = self.load_pred(b);
                let r = self.bop(op1, x, y);
                self.set_preds(dst, dst2, r, op2, c);
            }
            Op::Iadd {
                dst,
                a,
                neg_a,
                b,
                neg_b,
            } => {
                let x = self.load_reg(a);
                let x = self.negate_int(x, neg_a);
                let y = self.operand_uint(b)?;
                let y = self.negate_int(y, neg_b);
                let r = self.op2(op::I_ADD, uint, x, y);
                self.store_reg(dst, r);
            }
            Op::Iscadd {
                dst,
                a,
                neg_a,
                b,
                neg_b,
                shift,
            } => {
                let x = self.load_reg(a);
                let x = self.negate_int(x, neg_a);
                let s = self.constant_u(shift);
                let x = self.op2(op::SHIFT_LEFT_LOGICAL, uint, x, s);
                let y = self.operand_uint(b)?;
                let y = self.negate_int(y, neg_b);
                let r = self.op2(op::I_ADD, uint, x, y);
                self.store_reg(dst, r);
            }
            Op::Shl { dst, a, b }
            | Op::Shr {
                dst,
                a,
                b,
                signed: false,
            } => {
                let x = self.load_reg(a);
                let s = self.operand_uint(b)?;
                let opcode = if matches!(i, Op::Shl { .. }) {
                    op::SHIFT_LEFT_LOGICAL
                } else {
                    op::SHIFT_RIGHT_LOGICAL
                };
                let shifted = self.op2(opcode, uint, x, s);
                let w = self.constant_u(32);
                let big = self.op2(op::U_GREATER_THAN_EQUAL, self.c.bool_t, s, w);
                let r = self.select(uint, big, self.c.zero_u, shifted);
                self.store_reg(dst, r);
            }
            Op::Shr {
                dst,
                a,
                b,
                signed: true,
            } => {
                let x = self.load_reg(a);
                let s = self.operand_uint(b)?;
                let max = self.constant_u(31);
                let s = self.glsl(uint, GLSL_UMIN, &[s, max]);
                let r = self.op2(op::SHIFT_RIGHT_ARITHMETIC, uint, x, s);
                self.store_reg(dst, r);
            }
            Op::Lop {
                dst,
                op: o,
                a,
                inv_a,
                b,
                inv_b,
            } => {
                let x = self.load_reg(a);
                let x = if inv_a { self.op1(op::NOT, uint, x) } else { x };
                let y = self.operand_uint(b)?;
                let y = if inv_b { self.op1(op::NOT, uint, y) } else { y };
                let r = match o {
                    LopOp::And => self.op2(op::BITWISE_AND, uint, x, y),
                    LopOp::Or => self.op2(op::BITWISE_OR, uint, x, y),
                    LopOp::Xor => self.op2(op::BITWISE_XOR, uint, x, y),
                    LopOp::PassB => y,
                };
                self.store_reg(dst, r);
            }
            Op::Imnmx {
                dst,
                a,
                b,
                signed,
                min,
            } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let (fmin, fmax) = if signed {
                    (GLSL_SMIN, GLSL_SMAX)
                } else {
                    (GLSL_UMIN, GLSL_UMAX)
                };
                let lo = self.glsl(uint, fmin, &[x, y]);
                let hi = self.glsl(uint, fmax, &[x, y]);
                let c = self.load_pred(min);
                let r = self.select(uint, c, lo, hi);
                self.store_reg(dst, r);
            }
            Op::Sel { dst, a, b, cond } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let c = self.load_pred(cond);
                let r = self.select(uint, c, x, y);
                self.store_reg(dst, r);
            }
            Op::Xmad {
                dst,
                a,
                a_hi,
                a_signed,
                b,
                b_hi,
                b_signed,
                c,
                mode,
                psl,
                mrg,
            } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let z = self.operand_uint(c)?;
                let hx = self.half(x, a_hi, a_signed);
                let hy = self.half(y, b_hi, b_signed);
                let mut p = self.op2(op::I_MUL, uint, hx, hy);
                let sixteen = self.constant_u(16);
                if psl {
                    p = self.op2(op::SHIFT_LEFT_LOGICAL, uint, p, sixteen);
                }
                let z = match mode {
                    XmadMode::C => z,
                    XmadMode::Clo => {
                        let m = self.constant_u(0xFFFF);
                        self.op2(op::BITWISE_AND, uint, z, m)
                    }
                    XmadMode::Chi => self.op2(op::SHIFT_RIGHT_LOGICAL, uint, z, sixteen),
                    XmadMode::Cbcc => {
                        let bs = self.op2(op::SHIFT_LEFT_LOGICAL, uint, y, sixteen);
                        self.op2(op::I_ADD, uint, z, bs)
                    }
                };
                let mut r = self.op2(op::I_ADD, uint, p, z);
                if mrg {
                    let m = self.constant_u(0xFFFF);
                    let lo = self.op2(op::BITWISE_AND, uint, r, m);
                    let bs = self.op2(op::SHIFT_LEFT_LOGICAL, uint, y, sixteen);
                    r = self.op2(op::BITWISE_OR, uint, lo, bs);
                }
                self.store_reg(dst, r);
            }
            Op::F2i {
                dst,
                src,
                mods,
                signed,
                round,
            } => {
                // Saturating like the hardware (and Rust's casts); NaN
                // gives 0. Out-of-range conversions are undefined in
                // SPIR-V, so clamp first.
                let x = self.operand_float(src, mods)?;
                let x = self.round(x, round);
                let (lo, hi, limit, top) = if signed {
                    (
                        -2_147_483_648.0,
                        2_147_483_520.0,
                        2_147_483_648.0,
                        i32::MAX as u32,
                    )
                } else {
                    (0.0, 4_294_967_040.0, 4_294_967_296.0, u32::MAX)
                };
                let (lo, hi, limit) = (
                    self.constant_f(lo),
                    self.constant_f(hi),
                    self.constant_f(limit),
                );
                let clamped = self.glsl(float, GLSL_FCLAMP, &[x, lo, hi]);
                let opcode = if signed {
                    op::CONVERT_F_TO_S
                } else {
                    op::CONVERT_F_TO_U
                };
                let v = self.op1(opcode, uint, clamped);
                let over = self.op2(op::F_ORD_GREATER_THAN_EQUAL, self.c.bool_t, x, limit);
                let top = self.constant_u(top);
                let v = self.select(uint, over, top, v);
                let nan = self.is_nan(x);
                let v = self.select(uint, nan, self.c.zero_u, v);
                self.store_reg(dst, v);
            }
            Op::I2f {
                dst,
                src,
                signed,
                neg: n,
                abs,
            } => {
                let x = self.operand_uint(src)?;
                let f = if signed {
                    let x = if abs {
                        self.glsl(uint, GLSL_SABS, &[x])
                    } else {
                        x
                    };
                    self.op1(op::CONVERT_S_TO_F, float, x)
                } else {
                    self.op1(op::CONVERT_U_TO_F, float, x)
                };
                let f = self.mods(f, neg(n));
                self.store_float(dst, f, false);
            }
            Op::F2f {
                dst,
                src,
                mods,
                round,
                sat,
            } => {
                let x = self.operand_float(src, mods)?;
                let r = self.round(x, round);
                self.store_float(dst, r, sat);
            }
            Op::Ald {
                dst,
                base,
                addr,
                count,
            } => {
                if base != RZ {
                    return Err("indexed attribute loads are not translated yet".into());
                }
                for k in 0..count {
                    let a = addr + 4 * k;
                    let var = self.interface(a, false, stage)?;
                    let p = self.component_ptr(var, a, false);
                    let f = self.load(float, p);
                    let u = self.bitcast(uint, f);
                    self.store_reg(dst.wrapping_add(k as u8), u);
                }
            }
            Op::Ast {
                src,
                base,
                addr,
                count,
            } => {
                if base != RZ {
                    return Err("indexed attribute stores are not translated yet".into());
                }
                for k in 0..count {
                    let a = addr + 4 * k;
                    let var = self.interface(a, true, stage)?;
                    let p = self.component_ptr(var, a, true);
                    let u = self.load_reg(src.wrapping_add(k as u8));
                    let f = self.bitcast_float(u);
                    self.store(p, f);
                }
            }
            Op::Ipa {
                dst,
                addr,
                mode,
                sat,
            } => {
                let var = self.interface(addr, false, stage)?;
                if mode == IpaMode::Constant {
                    self.flat.insert(var);
                }
                let p = self.component_ptr(var, addr, false);
                let f = self.load(float, p);
                self.store_float(dst, f, sat);
            }
            Op::Texs {
                dst0,
                dst1,
                a,
                b,
                handle,
                comps,
                ..
            } => {
                let u = self.reg_float(a, no);
                let v = self.reg_float(b, no);
                let t = self.sample(handle, u, v)?;
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.store_float(super::interp::texs_dst(dst0, dst1, k), t[c], false);
                }
            }
            Op::Tex {
                dst,
                a,
                handle,
                comps,
                ..
            } => {
                let u = self.reg_float(a, no);
                let v = self.reg_float(a.wrapping_add(1), no);
                let t = self.sample(handle, u, v)?;
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    if dst != RZ {
                        self.store_float(dst.wrapping_add(k as u8), t[c], false);
                    }
                }
            }
            Op::Nop => {}
            Op::Bra { .. } | Op::Ssy { .. } | Op::Sync | Op::Kil | Op::Exit => {
                unreachable!("control flow is translated by the caller")
            }
        }
        Ok(())
    }

    /// Translates a program as a loop over its basic blocks:
    ///
    /// ```text
    /// pc = 0
    /// do {
    ///     switch (pc) { case k: <block k>; pc = <next block>; }
    /// } while (pc != EXIT);
    /// ```
    ///
    /// Leaves the body in the loop's merge block (after the program).
    fn translate(&mut self, program: &[Inst], stage: Stage) -> Result<(), String> {
        // Basic blocks start at 0, at branch targets and after control
        // flow.
        let mut leaders = BTreeSet::from([0usize]);
        for (i, ins) in program.iter().enumerate() {
            match ins.op {
                Op::Bra { target } | Op::Ssy { target } => {
                    leaders.insert(target);
                    if matches!(ins.op, Op::Bra { .. }) {
                        leaders.insert(i + 1);
                    }
                }
                Op::Exit | Op::Sync => {
                    leaders.insert(i + 1);
                }
                Op::Kil if ins.pred == PT => {
                    leaders.insert(i + 1);
                }
                _ => {}
            }
        }
        leaders.retain(|&l| l < program.len());
        let starts: Vec<usize> = leaders.iter().copied().collect();
        let block_of = |i: usize| starts.binary_search(&i).map_or(EXIT, |b| b as u32);
        let labels: Vec<u32> = starts.iter().map(|_| self.m.id()).collect();

        let (uint, bool_t) = (self.c.uint, self.c.bool_t);
        let p_uint = self.c.p_private_uint;
        let zero = self.c.zero_u;
        let pc = self.private_var(p_uint, zero);
        let sp = self.private_var(p_uint, zero);
        let steps = self.private_var(p_uint, zero);
        let stack_ptr = self.c.p_private_stack;
        let stack = self.private_var(stack_ptr, self.c.null_stack);
        let exit = self.constant_u(EXIT);

        let (header, dispatch, sw_merge, cont, merge) = (
            self.m.id(),
            self.m.id(),
            self.m.id(),
            self.m.id(),
            self.m.id(),
        );
        self.store(pc, zero);
        self.branch(header);
        self.label(header);
        inst(&mut self.m.body, op::LOOP_MERGE, &[merge, cont, 0]);
        self.branch(dispatch);
        self.label(dispatch);
        let cur = self.load(uint, pc);
        inst(&mut self.m.body, op::SELECTION_MERGE, &[sw_merge, 0]);
        let mut sw = vec![cur, sw_merge];
        for (k, &l) in labels.iter().enumerate() {
            sw.extend([k as u32, l]);
        }
        inst(&mut self.m.body, op::SWITCH, &sw);

        for (k, &start) in starts.iter().enumerate() {
            let end = starts.get(k + 1).copied().unwrap_or(program.len());
            self.label(labels[k]);
            let mut open = true;
            for (i, ins) in program.iter().enumerate().take(end).skip(start) {
                let p = ins.pred;
                match ins.op {
                    Op::Bra { .. } | Op::Exit | Op::Sync => {
                        if p != PT {
                            let next = self.constant_u(block_of(i + 1));
                            self.store(pc, next);
                        }
                        let flow = ins.op;
                        self.guarded(p, |t| {
                            match flow {
                                Op::Bra { target } => {
                                    let to = t.constant_u(block_of(target));
                                    t.store(pc, to);
                                }
                                Op::Exit => t.store(pc, exit),
                                _ => {
                                    // Pop the innermost SSY target; an empty
                                    // stack ends the program.
                                    let s = t.load(uint, sp);
                                    let empty = t.op2(op::I_EQUAL, bool_t, s, zero);
                                    let one = t.c.component[1];
                                    let s1 = t.op2(op::I_SUB, uint, s, one);
                                    let top = t.constant_u(SSY_DEPTH as u32 - 1);
                                    let idx = t.glsl(uint, GLSL_UMIN, &[s1, top]);
                                    let ptr = t.m.id();
                                    inst(
                                        &mut t.m.body,
                                        op::ACCESS_CHAIN,
                                        &[p_uint, ptr, stack, idx],
                                    );
                                    let to = t.load(uint, ptr);
                                    let to = t.select(uint, empty, exit, to);
                                    t.store(pc, to);
                                    let s1 = t.select(uint, empty, zero, s1);
                                    t.store(sp, s1);
                                }
                            }
                            Ok(())
                        })?;
                        self.branch(sw_merge);
                        open = false;
                        break;
                    }
                    Op::Kil => {
                        if p == PT {
                            inst(&mut self.m.body, op::KILL, &[]);
                            open = false;
                            break;
                        }
                        let cond = self.load_pred(p);
                        let (kill, after) = (self.m.id(), self.m.id());
                        inst(&mut self.m.body, op::SELECTION_MERGE, &[after, 0]);
                        inst(
                            &mut self.m.body,
                            op::BRANCH_CONDITIONAL,
                            &[cond, kill, after],
                        );
                        self.label(kill);
                        inst(&mut self.m.body, op::KILL, &[]);
                        self.label(after);
                    }
                    Op::Ssy { target } => {
                        self.guarded(p, |t| {
                            let s = t.load(uint, sp);
                            let top = t.constant_u(SSY_DEPTH as u32 - 1);
                            let idx = t.glsl(uint, GLSL_UMIN, &[s, top]);
                            let ptr = t.m.id();
                            inst(&mut t.m.body, op::ACCESS_CHAIN, &[p_uint, ptr, stack, idx]);
                            let to = t.constant_u(block_of(target));
                            t.store(ptr, to);
                            let one = t.c.component[1];
                            let s1 = t.op2(op::I_ADD, uint, s, one);
                            let depth = t.constant_u(SSY_DEPTH as u32);
                            let s1 = t.glsl(uint, GLSL_UMIN, &[s1, depth]);
                            t.store(sp, s1);
                            Ok(())
                        })?;
                    }
                    op => self.guarded(p, |t| t.exec(op, stage))?,
                }
            }
            if open {
                let next = self.constant_u(block_of(end));
                self.store(pc, next);
                self.branch(sw_merge);
            }
        }

        self.label(sw_merge);
        self.branch(cont);
        // Continue: leave at EXIT, or after too many blocks (a stuck
        // program must not hang the GPU).
        self.label(cont);
        let cur = self.load(uint, pc);
        let done = self.op2(op::I_EQUAL, bool_t, cur, exit);
        let n = self.load(uint, steps);
        let one = self.c.component[1];
        let n = self.op2(op::I_ADD, uint, n, one);
        self.store(steps, n);
        let limit = self.constant_u(STEP_LIMIT as u32);
        let over = self.op2(op::U_GREATER_THAN_EQUAL, bool_t, n, limit);
        let leave = self.op2(op::LOGICAL_OR, bool_t, done, over);
        inst(
            &mut self.m.body,
            op::BRANCH_CONDITIONAL,
            &[leave, merge, header],
        );
        self.label(merge);
        Ok(())
    }
}

/// Block number meaning "the program has ended".
const EXIT: u32 = u32::MAX;

/// Translates a vertex or pixel program to a SPIR-V module with entry
/// point "main".
pub fn translate(program: &[Inst], header: &Header) -> Result<Translated, String> {
    let stage = header.stage;
    let model = match stage {
        Stage::Vertex => 0,
        Stage::Pixel => 4,
        Stage::Other(n) => return Err(format!("shader stage {n} not translated")),
    };
    let mut m = Module::default();
    let glsl = m.id();
    let (void, fn_ty, float, uint, vec4) = (m.id(), m.id(), m.id(), m.id(), m.id());
    let (p_private_uint, p_in_vec4, p_out_vec4, p_in_float, p_out_float) =
        (m.id(), m.id(), m.id(), m.id(), m.id());
    let (zero_u, zero_f, one_f) = (m.id(), m.id(), m.id());
    let component = [m.id(), m.id(), m.id(), m.id()];
    let (uvec4, len_4096, arr, cbuf_struct, p_uniform_cbuf, p_uniform_uint) =
        (m.id(), m.id(), m.id(), m.id(), m.id(), m.id());
    let g = &mut m.globals;
    inst(g, op::TYPE_VOID, &[void]);
    inst(g, op::TYPE_FUNCTION, &[fn_ty, void]);
    inst(g, op::TYPE_FLOAT, &[float, 32]);
    inst(g, op::TYPE_INT, &[uint, 32, 0]);
    inst(g, op::TYPE_VECTOR, &[vec4, float, 4]);
    inst(g, op::TYPE_POINTER, &[p_private_uint, PRIVATE, uint]);
    inst(g, op::TYPE_POINTER, &[p_in_vec4, INPUT, vec4]);
    inst(g, op::TYPE_POINTER, &[p_out_vec4, OUTPUT, vec4]);
    inst(g, op::TYPE_POINTER, &[p_in_float, INPUT, float]);
    inst(g, op::TYPE_POINTER, &[p_out_float, OUTPUT, float]);
    inst(g, op::CONSTANT, &[uint, zero_u, 0]);
    inst(g, op::CONSTANT, &[float, zero_f, 0]);
    inst(g, op::CONSTANT, &[float, one_f, 1.0f32.to_bits()]);
    for (i, c) in component.iter().enumerate() {
        inst(g, op::CONSTANT, &[uint, *c, i as u32]);
    }
    // The constant buffer block type.
    inst(g, op::TYPE_VECTOR, &[uvec4, uint, 4]);
    inst(g, op::CONSTANT, &[uint, len_4096, 4096]);
    inst(g, op::TYPE_ARRAY, &[arr, uvec4, len_4096]);
    inst(g, op::TYPE_STRUCT, &[cbuf_struct, arr]);
    inst(g, op::TYPE_POINTER, &[p_uniform_cbuf, UNIFORM, cbuf_struct]);
    inst(g, op::TYPE_POINTER, &[p_uniform_uint, UNIFORM, uint]);
    let (bool_t, true_c, false_c, p_private_bool) = (m.id(), m.id(), m.id(), m.id());
    let (len_ssy, stack_t, p_private_stack, null_stack) = (m.id(), m.id(), m.id(), m.id());
    let (vec2, image_t, sampled_image, p_sampled_image) = (m.id(), m.id(), m.id(), m.id());
    let g = &mut m.globals;
    inst(g, op::TYPE_BOOL, &[bool_t]);
    inst(g, op::CONSTANT_TRUE, &[bool_t, true_c]);
    inst(g, op::CONSTANT_FALSE, &[bool_t, false_c]);
    inst(g, op::TYPE_POINTER, &[p_private_bool, PRIVATE, bool_t]);
    inst(g, op::CONSTANT, &[uint, len_ssy, SSY_DEPTH as u32]);
    inst(g, op::TYPE_ARRAY, &[stack_t, uint, len_ssy]);
    inst(g, op::TYPE_POINTER, &[p_private_stack, PRIVATE, stack_t]);
    inst(g, op::CONSTANT_NULL, &[stack_t, null_stack]);
    inst(g, op::TYPE_VECTOR, &[vec2, float, 2]);
    // 2D, not depth, not arrayed, single-sampled, sampled, format unknown.
    inst(g, op::TYPE_IMAGE, &[image_t, float, 1, 0, 0, 0, 1, 0]);
    inst(g, op::TYPE_SAMPLED_IMAGE, &[sampled_image, image_t]);
    inst(
        g,
        op::TYPE_POINTER,
        &[p_sampled_image, UNIFORM_CONSTANT, sampled_image],
    );
    let a = &mut m.annotations;
    inst(a, op::DECORATE, &[arr, ARRAY_STRIDE, 16]);
    inst(a, op::DECORATE, &[cbuf_struct, BLOCK]);
    inst(a, op::MEMBER_DECORATE, &[cbuf_struct, 0, OFFSET, 0]);
    let c = Common {
        glsl,
        float,
        uint,
        vec4,
        zero_u,
        zero_f,
        one_f,
        p_private_uint,
        p_in_vec4,
        p_out_vec4,
        p_in_float,
        p_out_float,
        component,
        p_uniform_cbuf,
        p_uniform_uint,
        bool_t,
        true_c,
        false_c,
        p_private_bool,
        p_private_stack,
        null_stack,
        vec2,
        sampled_image,
        p_sampled_image,
    };
    let mut t = Translator {
        m,
        c,
        regs: BTreeMap::new(),
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
        flat: BTreeSet::new(),
        cbufs: BTreeMap::new(),
        preds: BTreeMap::new(),
        textures: Vec::new(),
        stage,
    };
    let main = t.m.id();
    let label = t.m.id();
    t.translate(program, stage)?;

    // Pixel shader outputs: color target 0 from the registers its
    // output map names (r0, r1, ... in RGBA order).
    let mut color_out = None;
    if stage == Stage::Pixel {
        let mask = header.omap_targets[0];
        if header.omap_targets[1..].iter().any(|&m| m != 0) {
            return Err("pixel shaders writing several targets are not translated yet".into());
        }
        let out = t.m.id();
        inst(
            &mut t.m.globals,
            op::VARIABLE,
            &[t.c.p_out_vec4, out, OUTPUT],
        );
        inst(&mut t.m.annotations, op::DECORATE, &[out, LOCATION, 0]);
        let mut next = 0u8;
        let mut parts = Vec::new();
        for comp in 0..4 {
            if mask & (1 << comp) != 0 {
                let u = t.load_reg(next);
                parts.push(t.bitcast_float(u));
                next += 1;
            } else {
                parts.push(t.c.zero_f);
            }
        }
        let v = t.m.id();
        let mut ops = vec![t.c.vec4, v];
        ops.extend(parts);
        inst(&mut t.m.body, op::COMPOSITE_CONSTRUCT, &ops);
        inst(&mut t.m.body, op::STORE, &[out, v]);
        color_out = Some(out);
    }
    for &v in &t.flat {
        inst(&mut t.m.annotations, op::DECORATE, &[v, FLAT]);
    }

    let mut words = Vec::new();
    let bound = t.m.next_id + 1;
    words.extend([0x0723_0203, 0x0001_0300, 0, bound, 0]);
    inst(&mut words, op::CAPABILITY, &[1]); // Shader
    let mut ext = vec![glsl];
    ext.extend(string("GLSL.std.450"));
    inst(&mut words, op::EXT_INST_IMPORT, &ext);
    inst(&mut words, op::MEMORY_MODEL, &[0, 1]);
    let mut entry = vec![model, main];
    entry.extend(string("main"));
    entry.extend(t.inputs.values().chain(t.outputs.values()).copied());
    entry.extend(color_out);
    inst(&mut words, op::ENTRY_POINT, &entry);
    if stage == Stage::Pixel {
        inst(&mut words, op::EXECUTION_MODE, &[main, 7]); // OriginUpperLeft
    }
    let mut name = vec![main];
    name.extend(string("main"));
    inst(&mut t.m.debug, op::NAME, &name);
    words.extend(&t.m.preamble);
    words.extend(&t.m.debug);
    words.extend(&t.m.annotations);
    words.extend(&t.m.globals);
    inst(&mut words, op::FUNCTION, &[void, main, 0, fn_ty]);
    inst(&mut words, op::LABEL, &[label]);
    words.extend(&t.m.body);
    inst(&mut words, op::RETURN, &[]);
    inst(&mut words, op::FUNCTION_END, &[]);
    Ok(Translated {
        words,
        cbufs: t.cbufs.keys().fold(0, |m, &i| m | 1 << i),
        textures: t.textures.iter().map(|t| t.0).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::asm::*;
    use super::super::decode::decode_program;
    use super::*;

    /// Validates with spirv-val when it is installed.
    fn validate(words: &[u32]) {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("emu-spirv-{}-{n}.spv", std::process::id()));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        match std::process::Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.1")
            .arg(&path)
            .output()
        {
            Ok(out) => assert!(
                out.status.success(),
                "spirv-val: {}",
                String::from_utf8_lossy(&out.stderr)
            ),
            Err(_) => eprintln!("spirv-val not installed; skipping validation"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn vertex_and_pixel_shaders_validate() {
        let vs = program(
            &vertex_header(),
            &[
                ald(0, 0x80, 4),
                ald(4, 0x90, 4),
                fmul32i(0, 0, 0.5),
                ffma(1, 1, 1, 0),
                fadd_imm(2, 2, -0.5),
                mov32i(3, 0x3F80_0000),
                mov_cbuf(5, 2, 0x1234),
                ffma_cbuf(6, 5, 0, 0x40, 4),
                ast(0, 0x70, 4),
                ast(4, 0x80, 4),
                exit(),
            ],
        );
        let h = super::super::Header::parse(&vs);
        let t = translate(&decode_program(&vs[0x50..]).unwrap(), &h).unwrap();
        assert_eq!(t.cbufs, 0b101);
        validate(&t.words);
        let ps = program(
            &pixel_header(),
            &[
                ipa(0, 0x80),
                ipa(1, 0x84),
                ipa(2, 0x88),
                mov(3, 255),
                fadd(0, 0, 1),
                fmul_cbuf(2, 2, 3, 0x10),
                exit(),
            ],
        );
        let h = super::super::Header::parse(&ps);
        let t = translate(&decode_program(&ps[0x50..]).unwrap(), &h).unwrap();
        assert_eq!(t.cbufs, 1 << 3);
        validate(&t.words);
    }

    #[test]
    fn control_flow_validates() {
        let ps = control_flow_sample();
        let h = super::super::Header::parse(&ps);
        let t = translate(&decode_program(&ps[0x50..]).unwrap(), &h).unwrap();
        validate(&t.words);
    }
}
