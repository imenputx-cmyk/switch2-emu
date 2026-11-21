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
    AtomOp, Bop, FCmp, HalfOp, HalfSrc, ICmp, Inst, IntFunc, IpaMode, LopOp, Mods, MufuOp, Op,
    Operand, Pred, Round, Space, XmadMode, PT, RZ,
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
    pub const BIT_REVERSE: u16 = 204;
    pub const BIT_COUNT: u16 = 205;
    pub const I_ADD_CARRY: u16 = 149;
    pub const I_SUB_BORROW: u16 = 150;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const ARRAY_LENGTH: u16 = 68;
    pub const ATOMIC_EXCHANGE: u16 = 229;
    pub const ATOMIC_I_ADD: u16 = 234;
    pub const ATOMIC_S_MIN: u16 = 236;
    pub const ATOMIC_U_MIN: u16 = 237;
    pub const ATOMIC_S_MAX: u16 = 238;
    pub const ATOMIC_U_MAX: u16 = 239;
    pub const ATOMIC_AND: u16 = 240;
    pub const ATOMIC_OR: u16 = 241;
    pub const ATOMIC_XOR: u16 = 242;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const U_MUL_EXTENDED: u16 = 151;
    pub const S_MUL_EXTENDED: u16 = 152;
    pub const IMAGE_GATHER: u16 = 96;
    pub const IMAGE_QUERY_SIZE_LOD: u16 = 103;
    pub const IMAGE_QUERY_LEVELS: u16 = 106;
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
    pub const IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
    pub const IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
    pub const IMAGE_FETCH: u16 = 95;
    pub const IMAGE: u16 = 100;
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
const GLSL_FIND_S_MSB: u32 = 74;
const GLSL_PACK_HALF_2X16: u32 = 58;
const GLSL_UNPACK_HALF_2X16: u32 = 62;
const GLSL_FIND_U_MSB: u32 = 75;

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
    uvec2: u32,
    image_t: u32,
    sampled_image: u32,
    p_sampled_image: u32,
    /// struct { uint, uint } (extended multiply and carry results).
    uint_pair: u32,
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

/// Most storage buffers (guest memory reached by LDG/STG) a stage binds.
pub const MAX_STORAGE: usize = 8;

/// Descriptor binding of the `i`-th storage buffer of a stage.
pub fn storage_binding(stage: Stage, i: u32) -> u32 {
    match stage {
        Stage::Pixel => 100 + MAX_STORAGE as u32 + i,
        _ => 100 + i,
    }
}

/// A storage buffer a shader reaches: guest memory at the 64-bit
/// pointer in constant buffer `cbuf` at byte `offset` (Lichii
/// global_memory_to_storage_buffer_pass), bound from the pointer rounded
/// down to 4 bytes; its size word follows the pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageBuffer {
    pub cbuf: u8,
    pub offset: u32,
}

/// The constant buffer words a program reads that may hold storage
/// buffer pointers: 8-byte aligned, the NVN range (bank 0, 0x100..0x700,
/// Lichii's bias) first.
fn storage_candidates(program: &[Inst]) -> Vec<StorageBuffer> {
    let mut found = BTreeSet::new();
    let mut note = |o: Operand| {
        if let Operand::Cbuf { index, offset } = o {
            if offset.is_multiple_of(8) {
                found.insert((index, offset));
            }
        }
    };
    let mut global = false;
    for i in program {
        match i.op {
            Op::Ldc {
                index,
                base,
                offset,
                ..
            } if base == RZ => note(Operand::Cbuf { index, offset }),
            Op::Mov { src, .. } => note(src),
            Op::Iadd3 { b, c, .. } => {
                note(b);
                note(c);
            }
            Op::Iscadd { b, .. } | Op::Lea { b, .. } => note(b),
            Op::Ldg { .. } | Op::Stg { .. } => global = true,
            Op::Atom {
                space: Space::Global,
                ..
            } => global = true,
            _ => {}
        }
    }
    if !global {
        return Vec::new();
    }
    let biased = |&(b, o): &(u8, u32)| b == 0 && (0x100..0x700).contains(&o) && o % 16 == 0;
    let mut list: Vec<(u8, u32)> = found.iter().copied().filter(biased).collect();
    list.extend(found.iter().copied().filter(|c| !biased(c)));
    list.truncate(MAX_STORAGE);
    list.into_iter()
        .map(|(cbuf, offset)| StorageBuffer { cbuf, offset })
        .collect()
}

/// The image view a sampled texture binding needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TexKind {
    /// 2D (1D textures are 2D images of height 1).
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

impl TexKind {
    /// The kind for a TEX dimension field.
    pub fn from_dim(dim: u8) -> Option<TexKind> {
        Some(match dim {
            0 | 1 => TexKind::D2,
            4 | 5 => TexKind::D2Array,
            2 => TexKind::D3,
            3 => TexKind::Cube,
            7 => TexKind::CubeArray,
            _ => return None,
        })
    }

    /// The kind that holds a texture of TIC type `ty`.
    pub fn for_tic_type(ty: u32) -> TexKind {
        use crate::texture::ty;
        match ty {
            ty::D1_ARRAY | ty::D2_ARRAY => TexKind::D2Array,
            ty::D3 => TexKind::D3,
            ty::CUBE => TexKind::Cube,
            ty::CUBE_ARRAY => TexKind::CubeArray,
            _ => TexKind::D2,
        }
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
    /// The image view kind of each texture binding.
    pub texture_kinds: Vec<TexKind>,
    /// Storage buffers, the `i`-th at [`storage_binding`]`(stage, i)`.
    pub storage: Vec<StorageBuffer>,
    /// The shader writes guest memory (results must be written back).
    pub storage_written: bool,
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
    /// Texture variables by handle, in binding order, with their kind.
    textures: Vec<(u32, u32, TexKind)>,
    /// Image, sampled image and pointer types by kind (2D: the common
    /// ones), and lazily declared vector types.
    image_types: BTreeMap<u8, (u32, u32, u32)>,
    vec3: Option<u32>,
    uvec3: Option<u32>,
    /// Storage buffer candidates and, once declared, their variables
    /// and the pointer type to their words.
    storage_slots: Vec<StorageBuffer>,
    storage_vars: Vec<u32>,
    p_storage_uint: u32,
    storage_written: bool,
    /// Compute: built-in input variables by BuiltIn number, the shared
    /// memory array and its size in words.
    builtins: BTreeMap<u32, u32>,
    shared: Option<u32>,
    shared_words: u32,
    shared_p_word: u32,
    stage: Stage,
    /// Local memory (LDL/STL): a private array of words.
    local: Option<u32>,
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

    /// Words of local memory a translated shader has.
    const LOCAL_WORDS: u32 = 1024;

    fn local_var(&mut self) -> u32 {
        if let Some(v) = self.local {
            return v;
        }
        let (len, arr, ptr, null, v) = (
            self.m.id(),
            self.m.id(),
            self.m.id(),
            self.m.id(),
            self.m.id(),
        );
        let g = &mut self.m.globals;
        inst(g, op::CONSTANT, &[self.c.uint, len, Self::LOCAL_WORDS]);
        inst(g, op::TYPE_ARRAY, &[arr, self.c.uint, len]);
        inst(g, op::TYPE_POINTER, &[ptr, PRIVATE, arr]);
        inst(g, op::CONSTANT_NULL, &[arr, null]);
        inst(g, op::VARIABLE, &[ptr, v, PRIVATE, null]);
        self.local = Some(v);
        v
    }

    /// Pointer to local word `index` (clamped into the array).
    fn local_word(&mut self, index: u32) -> u32 {
        let var = self.local_var();
        let last = self.constant_u(Self::LOCAL_WORDS - 1);
        let i = self.glsl(self.c.uint, GLSL_UMIN, &[index, last]);
        let p = self.m.id();
        inst(
            &mut self.m.body,
            op::ACCESS_CHAIN,
            &[self.c.p_private_uint, p, var, i],
        );
        p
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

    /// A logical shift that gives 0 for amounts of 32 or more (SPIR-V
    /// leaves those undefined).
    fn safe_shift(&mut self, opcode: u16, v: u32, s: u32) -> u32 {
        let uint = self.c.uint;
        let shifted = self.op2(opcode, uint, v, s);
        let w = self.constant_u(32);
        let big = self.op2(op::U_GREATER_THAN_EQUAL, self.c.bool_t, s, w);
        self.select(uint, big, self.c.zero_u, shifted)
    }

    /// Adds 32-bit `terms` and boolean `ones`, returning the low word and
    /// the number of carries out of bit 31 (a uint, 0..=len-1).
    fn add_chain(&mut self, terms: &[u32], ones: &[u32]) -> (u32, u32) {
        let uint = self.c.uint;
        let one = self.constant_u(1);
        let mut sum = self.c.zero_u;
        let mut carries = self.c.zero_u;
        let addends: Vec<u32> = terms
            .iter()
            .copied()
            .chain(
                ones.iter()
                    .map(|&b| self.select(uint, b, one, self.c.zero_u))
                    .collect::<Vec<_>>(),
            )
            .collect();
        for v in addends {
            let next = self.op2(op::I_ADD, uint, sum, v);
            let wrapped = self.op2(op::U_LESS_THAN, self.c.bool_t, next, sum);
            let c = self.select(uint, wrapped, one, self.c.zero_u);
            carries = self.op2(op::I_ADD, uint, carries, c);
            sum = next;
        }
        (sum, carries)
    }

    /// Writes bits 0 and 1 of the carry count to two predicates.
    fn store_carries(&mut self, co: [u8; 2], carries: u32) {
        let uint = self.c.uint;
        for (k, &p) in co.iter().enumerate() {
            if p == 7 {
                continue;
            }
            let bit = self.constant_u(1 << k);
            let m = self.op2(op::BITWISE_AND, uint, carries, bit);
            let v = self.op2(op::I_NOT_EQUAL, self.c.bool_t, m, self.c.zero_u);
            self.store_pred(p, v);
        }
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
        self.texture_var_of(handle, TexKind::D2)
    }

    /// A uvec3 built-in input (compute ids), component `k` as a uint.
    fn builtin_component(&mut self, builtin: u32, k: u32) -> u32 {
        let var = match self.builtins.get(&builtin) {
            Some(&v) => v,
            None => {
                let ty = self.uvec3();
                let (ptr, v) = (self.m.id(), self.m.id());
                inst(&mut self.m.globals, op::TYPE_POINTER, &[ptr, INPUT, ty]);
                inst(&mut self.m.globals, op::VARIABLE, &[ptr, v, INPUT]);
                inst(
                    &mut self.m.annotations,
                    op::DECORATE,
                    &[v, BUILTIN, builtin],
                );
                self.builtins.insert(builtin, v);
                v
            }
        };
        let ty = self.uvec3();
        let vec = self.load(ty, var);
        self.extract(self.c.uint, vec, k)
    }

    /// The shared memory array (Workgroup words).
    fn shared_var(&mut self) -> Result<u32, String> {
        if let Some(v) = self.shared {
            return Ok(v);
        }
        if self.shared_words == 0 {
            return Err("shared memory without a size".into());
        }
        const WORKGROUP: u32 = 4;
        let (len, arr, ptr, v) = (self.m.id(), self.m.id(), self.m.id(), self.m.id());
        let p_word = self.m.id();
        let g = &mut self.m.globals;
        inst(g, op::CONSTANT, &[self.c.uint, len, self.shared_words]);
        inst(g, op::TYPE_ARRAY, &[arr, self.c.uint, len]);
        inst(g, op::TYPE_POINTER, &[ptr, WORKGROUP, arr]);
        inst(g, op::TYPE_POINTER, &[p_word, WORKGROUP, self.c.uint]);
        inst(g, op::VARIABLE, &[ptr, v, WORKGROUP]);
        self.shared = Some(v);
        self.shared_p_word = p_word;
        Ok(v)
    }

    /// Pointer to shared word `index` (clamped into the array).
    fn shared_word(&mut self, index: u32) -> Result<u32, String> {
        let var = self.shared_var()?;
        let last = self.constant_u(self.shared_words - 1);
        let i = self.glsl(self.c.uint, GLSL_UMIN, &[index, last]);
        let p = self.m.id();
        inst(
            &mut self.m.body,
            op::ACCESS_CHAIN,
            &[self.shared_p_word, p, var, i],
        );
        Ok(p)
    }

    /// The byte address of a shared access.
    fn shared_address(&mut self, addr: u8, uaddr: u8, offset: i32) -> u32 {
        let uint = self.c.uint;
        let a = self.load_reg(addr);
        let u = self.load_reg(uaddr);
        let o = self.constant_u(offset as u32);
        let at = self.op2(op::I_ADD, uint, a, u);
        self.op2(op::I_ADD, uint, at, o)
    }

    /// Declares the storage buffer variables (once).
    fn storage_vars(&mut self) -> Result<(), String> {
        if !self.storage_vars.is_empty() {
            return Ok(());
        }
        if self.storage_slots.is_empty() {
            return Err("global memory without a constant buffer pointer".into());
        }
        const STORAGE_BUFFER: u32 = 12;
        let (rta, block, p_block, p_uint) = (self.m.id(), self.m.id(), self.m.id(), self.m.id());
        let g = &mut self.m.globals;
        inst(g, op::TYPE_RUNTIME_ARRAY, &[rta, self.c.uint]);
        inst(g, op::TYPE_STRUCT, &[block, rta]);
        inst(g, op::TYPE_POINTER, &[p_block, STORAGE_BUFFER, block]);
        inst(g, op::TYPE_POINTER, &[p_uint, STORAGE_BUFFER, self.c.uint]);
        let a = &mut self.m.annotations;
        inst(a, op::DECORATE, &[rta, 6, 4]); // ArrayStride 4
        inst(a, op::DECORATE, &[block, 2]); // Block
        inst(a, op::MEMBER_DECORATE, &[block, 0, 35, 0]); // Offset 0
        self.p_storage_uint = p_uint;
        for i in 0..self.storage_slots.len() {
            let v = self.m.id();
            inst(
                &mut self.m.globals,
                op::VARIABLE,
                &[p_block, v, STORAGE_BUFFER],
            );
            let binding = storage_binding(self.stage, i as u32);
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
            self.storage_vars.push(v);
        }
        Ok(())
    }

    /// 64-bit add: (lo, hi) + (b_lo, b_hi).
    fn add64(&mut self, lo: u32, hi: u32, b_lo: u32, b_hi: u32) -> (u32, u32) {
        let uint = self.c.uint;
        let r = self.m.id();
        inst(
            &mut self.m.body,
            op::I_ADD_CARRY,
            &[self.c.uint_pair, r, lo, b_lo],
        );
        let sum = self.extract(uint, r, 0);
        let carry = self.extract(uint, r, 1);
        let h = self.op2(op::I_ADD, uint, hi, b_hi);
        let h = self.op2(op::I_ADD, uint, h, carry);
        (sum, h)
    }

    /// The 64-bit address of a global access.
    fn global_address(&mut self, addr: u8, wide: bool, uaddr: u8, offset: i32) -> (u32, u32) {
        let lo = self.load_reg(addr);
        let hi = if wide && addr != RZ {
            self.load_reg(addr + 1)
        } else {
            self.c.zero_u
        };
        let (mut lo, mut hi) = (lo, hi);
        if uaddr != RZ {
            let (ul, uh) = (self.load_reg(uaddr), self.load_reg(uaddr + 1));
            (lo, hi) = self.add64(lo, hi, ul, uh);
        }
        if offset != 0 {
            let ol = self.constant_u(offset as u32);
            let oh = self.constant_u(if offset < 0 { u32::MAX } else { 0 });
            (lo, hi) = self.add64(lo, hi, ol, oh);
        }
        (lo, hi)
    }

    /// For each storage buffer: whether the word at (lo, hi) is inside
    /// it, and a pointer to that word (clamped into the buffer).
    fn global_word(&mut self, lo: u32, hi: u32) -> Result<Vec<(u32, u32)>, String> {
        self.storage_vars()?;
        let (uint, bool_t) = (self.c.uint, self.c.bool_t);
        let mut out = Vec::new();
        for i in 0..self.storage_slots.len() {
            let sb = self.storage_slots[i];
            let var = self.storage_vars[i];
            let base_lo = self.operand_uint(Operand::Cbuf {
                index: sb.cbuf,
                offset: sb.offset,
            })?;
            let base_hi = self.operand_uint(Operand::Cbuf {
                index: sb.cbuf,
                offset: sb.offset + 4,
            })?;
            let mask = self.constant_u(!3);
            let base_lo = self.op2(op::BITWISE_AND, uint, base_lo, mask);
            let r = self.m.id();
            inst(
                &mut self.m.body,
                op::I_SUB_BORROW,
                &[self.c.uint_pair, r, lo, base_lo],
            );
            let off = self.extract(uint, r, 0);
            let borrow = self.extract(uint, r, 1);
            let oh = self.op2(op::I_SUB, uint, hi, base_hi);
            let oh = self.op2(op::I_SUB, uint, oh, borrow);
            let len = self.m.id();
            inst(&mut self.m.body, op::ARRAY_LENGTH, &[uint, len, var, 0]);
            let two = self.constant_u(2);
            let idx = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, off, two);
            let high_ok = self.op2(op::I_EQUAL, bool_t, oh, self.c.zero_u);
            let in_len = self.op2(op::U_LESS_THAN, bool_t, idx, len);
            let ok = self.op2(op::LOGICAL_AND, bool_t, high_ok, in_len);
            let one = self.c.component[1];
            let last = self.op2(op::I_SUB, uint, len, one);
            let idx = self.glsl(uint, GLSL_UMIN, &[idx, last]);
            let p = self.m.id();
            inst(
                &mut self.m.body,
                op::ACCESS_CHAIN,
                &[self.p_storage_uint, p, var, self.c.zero_u, idx],
            );
            out.push((ok, p));
        }
        Ok(out)
    }

    /// Loads the word at (lo, hi) (0 outside every storage buffer).
    fn global_load_word(&mut self, lo: u32, hi: u32) -> Result<u32, String> {
        let uint = self.c.uint;
        let mut v = self.c.zero_u;
        for (ok, p) in self.global_word(lo, hi)? {
            let w = self.load(uint, p);
            v = self.select(uint, ok, w, v);
        }
        Ok(v)
    }

    /// Runs `f(pointer)` for the storage buffer holding (lo, hi).
    fn global_at(
        &mut self,
        lo: u32,
        hi: u32,
        mut f: impl FnMut(&mut Self, u32) -> Result<(), String>,
    ) -> Result<(), String> {
        for (ok, p) in self.global_word(lo, hi)? {
            let (then, merge) = (self.m.id(), self.m.id());
            inst(&mut self.m.body, op::SELECTION_MERGE, &[merge, 0]);
            inst(&mut self.m.body, op::BRANCH_CONDITIONAL, &[ok, then, merge]);
            self.label(then);
            f(self, p)?;
            self.branch(merge);
            self.label(merge);
        }
        Ok(())
    }

    /// An atomic on a storage buffer word: the old value.
    fn atomic(&mut self, opcode: u16, p: u32, v: u32) -> u32 {
        self.atomic_scoped(opcode, p, v, 1)
    }

    /// An atomic with a memory scope (1 device, 2 workgroup).
    fn atomic_scoped(&mut self, opcode: u16, p: u32, v: u32, scope: u32) -> u32 {
        let scope = self.constant_u(scope);
        let relaxed = self.c.zero_u;
        let r = self.m.id();
        inst(
            &mut self.m.body,
            opcode,
            &[self.c.uint, r, p, scope, relaxed, v],
        );
        r
    }

    /// OpImageSample{Implicit,Explicit}Lod (explicit: level 0) into
    /// `texel`.
    fn sample_op(&mut self, texel: u32, sampled: u32, coord: u32, auto_lod: bool) {
        if auto_lod && self.stage == Stage::Pixel {
            inst(
                &mut self.m.body,
                op::IMAGE_SAMPLE_IMPLICIT_LOD,
                &[self.c.vec4, texel, sampled, coord],
            );
        } else {
            inst(
                &mut self.m.body,
                op::IMAGE_SAMPLE_EXPLICIT_LOD,
                &[self.c.vec4, texel, sampled, coord, 0x2, self.c.zero_f],
            );
        }
    }

    /// (image, sampled image, pointer to sampled image) types of a kind.
    fn image_types(&mut self, kind: TexKind) -> (u32, u32, u32) {
        if kind == TexKind::D2 {
            return (self.c.image_t, self.c.sampled_image, self.c.p_sampled_image);
        }
        if let Some(&t) = self.image_types.get(&(kind as u8)) {
            return t;
        }
        let (dim, arrayed) = match kind {
            TexKind::D2Array => (1, 1),
            TexKind::D3 => (2, 0),
            TexKind::Cube => (3, 0),
            _ => (3, 1),
        };
        let (img, sampled, ptr) = (self.m.id(), self.m.id(), self.m.id());
        let g = &mut self.m.globals;
        inst(
            g,
            op::TYPE_IMAGE,
            &[img, self.c.float, dim, 0, arrayed, 0, 1, 0],
        );
        inst(g, op::TYPE_SAMPLED_IMAGE, &[sampled, img]);
        inst(g, op::TYPE_POINTER, &[ptr, UNIFORM_CONSTANT, sampled]);
        self.image_types.insert(kind as u8, (img, sampled, ptr));
        (img, sampled, ptr)
    }

    fn vec3(&mut self) -> u32 {
        if let Some(v) = self.vec3 {
            return v;
        }
        let v = self.m.id();
        inst(&mut self.m.globals, op::TYPE_VECTOR, &[v, self.c.float, 3]);
        self.vec3 = Some(v);
        v
    }

    fn uvec3(&mut self) -> u32 {
        if let Some(v) = self.uvec3 {
            return v;
        }
        let v = self.m.id();
        inst(&mut self.m.globals, op::TYPE_VECTOR, &[v, self.c.uint, 3]);
        self.uvec3 = Some(v);
        v
    }

    fn texture_var_of(&mut self, handle: u32, kind: TexKind) -> Result<u32, String> {
        if let Some(&(_, v, k)) = self.textures.iter().find(|t| t.0 == handle) {
            if k != kind {
                return Err(format!("texture {handle:#x} used as {k:?} and {kind:?}"));
            }
            return Ok(v);
        }
        if self.textures.len() == MAX_TEXTURES {
            return Err("too many textures".into());
        }
        let (_, _, ptr) = self.image_types(kind);
        let v = self.m.id();
        inst(
            &mut self.m.globals,
            op::VARIABLE,
            &[ptr, v, UNIFORM_CONSTANT],
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
        self.textures.push((handle, v, kind));
        Ok(v)
    }

    /// TEX/TLD of any dimension (level 0).
    fn texture_any(
        &mut self,
        dim: u8,
        handle: u32,
        coords: [u8; 3],
        layer: u8,
        fetch: bool,
        auto_lod: bool,
    ) -> Result<[u32; 4], String> {
        let kind = TexKind::from_dim(dim).ok_or("texture dimension")?;
        let var = self.texture_var_of(handle, kind)?;
        let (img_t, sampled_t, _) = self.image_types(kind);
        let sampled = self.load(sampled_t, var);
        let float = self.c.float;
        let texel = self.m.id();
        if fetch {
            if matches!(kind, TexKind::Cube | TexKind::CubeArray) {
                return Err("cube texel fetch".into());
            }
            let x = self.load_reg(coords[0]);
            let y = if dim == 0 || dim == 4 {
                self.c.zero_u
            } else {
                self.load_reg(coords[1])
            };
            let coord = self.m.id();
            match kind {
                TexKind::D2 => inst(
                    &mut self.m.body,
                    op::COMPOSITE_CONSTRUCT,
                    &[self.c.uvec2, coord, x, y],
                ),
                _ => {
                    let z = if kind == TexKind::D3 {
                        self.load_reg(coords[2])
                    } else {
                        self.load_reg(layer)
                    };
                    let t = self.uvec3();
                    inst(
                        &mut self.m.body,
                        op::COMPOSITE_CONSTRUCT,
                        &[t, coord, x, y, z],
                    );
                }
            }
            let img = self.m.id();
            inst(&mut self.m.body, op::IMAGE, &[img_t, img, sampled]);
            inst(
                &mut self.m.body,
                op::IMAGE_FETCH,
                &[self.c.vec4, texel, img, coord, 0x2, self.c.zero_u],
            );
        } else {
            let no = Mods::default();
            let mut parts: Vec<u32> = Vec::new();
            let n = [1, 2, 3, 3, 1, 2, 0, 3][dim as usize];
            for &r in coords.iter().take(n) {
                parts.push(self.reg_float(r, no));
            }
            if dim == 0 || dim == 4 {
                // 1D as a 2D image of height 1: sample its middle row.
                parts.push(self.constant_f(0.5));
            }
            if matches!(kind, TexKind::D2Array | TexKind::CubeArray) {
                let l = self.load_reg(layer);
                parts.push(self.op1(op::CONVERT_U_TO_F, float, l));
            }
            let ty = match parts.len() {
                2 => self.c.vec2,
                3 => self.vec3(),
                _ => self.c.vec4,
            };
            let coord = self.m.id();
            let mut args = vec![ty, coord];
            args.extend(&parts);
            inst(&mut self.m.body, op::COMPOSITE_CONSTRUCT, &args);
            self.sample_op(texel, sampled, coord, auto_lod);
        }
        Ok([0, 1, 2, 3].map(|k| self.extract(float, texel, k)))
    }

    /// Samples level 0 of a texture; returns the RGBA components (floats).
    /// With `auto_lod` in a pixel shader the level comes from the
    /// derivatives (implicit LOD); otherwise level 0.
    fn sample(&mut self, handle: u32, u: u32, v: u32, auto_lod: bool) -> Result<[u32; 4], String> {
        let var = self.texture_var(handle)?;
        let img = self.load(self.c.sampled_image, var);
        let coord = self.m.id();
        inst(
            &mut self.m.body,
            op::COMPOSITE_CONSTRUCT,
            &[self.c.vec2, coord, u, v],
        );
        let texel = self.m.id();
        self.sample_op(texel, img, coord, auto_lod);
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

    /// Fetches texel (x, y) of level 0 (OpImageFetch on the sampled
    /// image's image).
    fn fetch(&mut self, handle: u32, x: u32, y: u32) -> Result<[u32; 4], String> {
        let var = self.texture_var(handle)?;
        let sampled = self.load(self.c.sampled_image, var);
        let img = self.m.id();
        inst(&mut self.m.body, op::IMAGE, &[self.c.image_t, img, sampled]);
        let coord = self.m.id();
        inst(
            &mut self.m.body,
            op::COMPOSITE_CONSTRUCT,
            &[self.c.uvec2, coord, x, y],
        );
        let texel = self.m.id();
        // Image operand Lod (0x2) = 0.
        inst(
            &mut self.m.body,
            op::IMAGE_FETCH,
            &[self.c.vec4, texel, img, coord, 0x2, self.c.zero_u],
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

    fn extract(&mut self, ty: u32, v: u32, k: u32) -> u32 {
        let id = self.m.id();
        inst(&mut self.m.body, op::COMPOSITE_EXTRACT, &[ty, id, v, k]);
        id
    }

    /// The image of a texture's sampled image.
    fn texture_image(&mut self, handle: u32) -> Result<(u32, u32), String> {
        let var = self.texture_var(handle)?;
        let sampled = self.load(self.c.sampled_image, var);
        let img = self.m.id();
        inst(&mut self.m.body, op::IMAGE, &[self.c.image_t, img, sampled]);
        Ok((sampled, img))
    }

    /// The local word index of register + uniform + offset (bytes).
    fn local_index(&mut self, addr: u8, uaddr: u8, offset: i32) -> u32 {
        let uint = self.c.uint;
        let a = self.load_reg(addr);
        let u = self.load_reg(uaddr);
        let o = self.constant_u(offset as u32);
        let at = self.op2(op::I_ADD, uint, a, u);
        let at = self.op2(op::I_ADD, uint, at, o);
        let two = self.constant_u(2);
        self.op2(op::SHIFT_RIGHT_LOGICAL, uint, at, two)
    }

    /// The two lanes (floats) of a packed half operand.
    fn half_lanes(&mut self, s: HalfSrc) -> Result<[u32; 2], String> {
        let v = self.operand_uint(s.src)?;
        let lanes = if s.swizzle == 1 {
            let f = self.bitcast_float(v);
            [f, f]
        } else {
            let pair = self.glsl(self.c.vec2, GLSL_UNPACK_HALF_2X16, &[v]);
            let lo = self.extract(self.c.float, pair, 0);
            let hi = self.extract(self.c.float, pair, 1);
            match s.swizzle {
                0 => [lo, hi],
                2 => [lo, lo],
                _ => [hi, hi],
            }
        };
        let m = Mods {
            neg: s.neg,
            abs: s.abs,
        };
        Ok(lanes.map(|f| self.mods(f, m)))
    }

    /// Two floats packed as halves.
    fn pack_halves(&mut self, lo: u32, hi: u32) -> u32 {
        let v = self.m.id();
        inst(
            &mut self.m.body,
            op::COMPOSITE_CONSTRUCT,
            &[self.c.vec2, v, lo, hi],
        );
        self.glsl(self.c.uint, GLSL_PACK_HALF_2X16, &[v])
    }

    /// Truth table of three booleans (bit i for (a, b, c) = (i >> 2, i >> 1, i) & 1).
    fn bool_lut(&mut self, lut: u8, a: u32, b: u32, c: u32) -> u32 {
        let bool_t = self.c.bool_t;
        let inv = [
            self.op1(op::LOGICAL_NOT, bool_t, a),
            self.op1(op::LOGICAL_NOT, bool_t, b),
            self.op1(op::LOGICAL_NOT, bool_t, c),
        ];
        let mut r = self.c.false_c;
        for i in 0..8u32 {
            if lut >> i & 1 == 0 {
                continue;
            }
            let pa = if i & 4 != 0 { a } else { inv[0] };
            let pb = if i & 2 != 0 { b } else { inv[1] };
            let pc = if i & 1 != 0 { c } else { inv[2] };
            let t = self.op2(op::LOGICAL_AND, bool_t, pa, pb);
            let t = self.op2(op::LOGICAL_AND, bool_t, t, pc);
            r = self.op2(op::LOGICAL_OR, bool_t, r, t);
        }
        r
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
            Op::Iadd3 {
                dst,
                a,
                neg_a,
                b,
                neg_b,
                c,
                neg_c,
                x: ext,
                cin,
                co,
            } if !ext && co == [7, 7] => {
                let _ = cin;
                let x = self.load_reg(a);
                let x = self.negate_int(x, neg_a);
                let y = self.operand_uint(b)?;
                let y = self.negate_int(y, neg_b);
                let z = self.operand_uint(c)?;
                let z = self.negate_int(z, neg_c);
                let r = self.op2(op::I_ADD, uint, x, y);
                let r = self.op2(op::I_ADD, uint, r, z);
                self.store_reg(dst, r);
            }
            Op::Iadd3 {
                dst,
                a,
                neg_a,
                b,
                neg_b,
                c,
                neg_c,
                x: ext,
                cin,
                co,
            } => {
                let mut terms = vec![];
                let mut ones = vec![];
                let va = self.load_reg(a);
                let vb = self.operand_uint(b)?;
                let vc = self.operand_uint(c)?;
                for (v, neg) in [(va, neg_a), (vb, neg_b), (vc, neg_c)] {
                    if neg {
                        terms.push(self.op1(op::NOT, uint, v));
                        if !ext {
                            ones.push(self.c.true_c);
                        }
                    } else {
                        terms.push(v);
                    }
                }
                if ext {
                    for p in cin {
                        ones.push(self.load_pred(p));
                    }
                }
                let (sum, carries) = self.add_chain(&terms, &ones);
                self.store_reg(dst, sum);
                self.store_carries(co, carries);
            }
            Op::Lea {
                dst,
                a,
                neg_a,
                b,
                c,
                shift,
                hi,
                x: ext,
                cin,
                co,
            } => {
                let va = self.load_reg(a);
                let s = self.constant_u(shift);
                let mut shifted = self.op2(op::SHIFT_LEFT_LOGICAL, uint, va, s);
                if hi {
                    // (c << shift) | (a >> (32 - shift)); shift 0 gives c.
                    let vc = self.load_reg(c);
                    let lo = self.constant_u(32 - shift);
                    let top = self.safe_shift(op::SHIFT_RIGHT_LOGICAL, va, lo);
                    shifted = self.op2(op::SHIFT_LEFT_LOGICAL, uint, vc, s);
                    shifted = self.op2(op::BITWISE_OR, uint, shifted, top);
                }
                let mut ones = vec![];
                if neg_a {
                    shifted = self.op1(op::NOT, uint, shifted);
                    if !ext {
                        ones.push(self.c.true_c);
                    }
                }
                if ext {
                    ones.push(self.load_pred(cin));
                }
                let vb = self.operand_uint(b)?;
                let (sum, carries) = self.add_chain(&[shifted, vb], &ones);
                self.store_reg(dst, sum);
                self.store_carries([co, 7], carries);
            }
            Op::Lop3 {
                dst,
                a,
                b,
                c,
                lut,
                pred,
            } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let z = self.operand_uint(c)?;
                let inv = [
                    self.op1(op::NOT, uint, x),
                    self.op1(op::NOT, uint, y),
                    self.op1(op::NOT, uint, z),
                ];
                let mut r = self.c.zero_u;
                for i in 0..8u32 {
                    if lut >> i & 1 == 0 {
                        continue;
                    }
                    let pa = if i & 4 != 0 { x } else { inv[0] };
                    let pb = if i & 2 != 0 { y } else { inv[1] };
                    let pc = if i & 1 != 0 { z } else { inv[2] };
                    let t = self.op2(op::BITWISE_AND, uint, pa, pb);
                    let t = self.op2(op::BITWISE_AND, uint, t, pc);
                    r = self.op2(op::BITWISE_OR, uint, r, t);
                }
                self.store_reg(dst, r);
                if pred != 7 {
                    let nz = self.op2(op::I_NOT_EQUAL, self.c.bool_t, r, self.c.zero_u);
                    self.store_pred(pred, nz);
                }
            }
            Op::Imad {
                dst,
                a,
                b,
                c,
                neg_c,
            } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let z = self.operand_uint(c)?;
                let z = self.negate_int(z, neg_c);
                let p = self.op2(op::I_MUL, uint, x, y);
                let r = self.op2(op::I_ADD, uint, p, z);
                self.store_reg(dst, r);
            }
            Op::Shf {
                dst,
                a,
                b,
                c,
                right,
                hi,
                wrap,
                wide,
                signed,
            } => {
                if wide {
                    return Err("SHF with a 64-bit type".into());
                }
                let low = self.load_reg(a);
                let amount = self.operand_uint(b)?;
                let high = self.operand_uint(c)?;
                let s = if wrap {
                    let m = self.constant_u(31);
                    self.op2(op::BITWISE_AND, uint, amount, m)
                } else {
                    let w = self.constant_u(32);
                    self.glsl(uint, GLSL_UMIN, &[amount, w])
                };
                let w = self.constant_u(32);
                let rest = self.op2(op::I_SUB, uint, w, s);
                let r = match (right, hi) {
                    (false, false) => self.safe_shift(op::SHIFT_LEFT_LOGICAL, low, s),
                    (false, true) => {
                        let h = self.safe_shift(op::SHIFT_LEFT_LOGICAL, high, s);
                        let l = self.safe_shift(op::SHIFT_RIGHT_LOGICAL, low, rest);
                        self.op2(op::BITWISE_OR, uint, h, l)
                    }
                    (true, false) => {
                        let l = self.safe_shift(op::SHIFT_RIGHT_LOGICAL, low, s);
                        let h = self.safe_shift(op::SHIFT_LEFT_LOGICAL, high, rest);
                        self.op2(op::BITWISE_OR, uint, l, h)
                    }
                    (true, true) if signed => {
                        let m = self.constant_u(31);
                        let s = self.glsl(uint, GLSL_UMIN, &[s, m]);
                        self.op2(op::SHIFT_RIGHT_ARITHMETIC, uint, high, s)
                    }
                    (true, true) => self.safe_shift(op::SHIFT_RIGHT_LOGICAL, high, s),
                };
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
                lz,
            } => {
                let u = self.reg_float(a, no);
                let v = self.reg_float(b, no);
                let t = self.sample(handle, u, v, !lz)?;
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.store_float(super::interp::texs_dst(dst0, dst1, k), t[c], false);
                }
            }
            Op::Tld {
                dst0,
                dst1,
                a,
                b,
                handle,
                comps,
            } => {
                let x = self.load_reg(a);
                let y = self.load_reg(b);
                let t = self.fetch(handle, x, y)?;
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.store_float(super::interp::texs_dst(dst0, dst1, k), t[c], false);
                }
            }
            Op::Tex {
                dst,
                a,
                handle,
                comps,
                lz,
            } => {
                let u = self.reg_float(a, no);
                let v = self.reg_float(a.wrapping_add(1), no);
                let t = self.sample(handle, u, v, !lz)?;
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    if dst != RZ {
                        self.store_float(dst.wrapping_add(k as u8), t[c], false);
                    }
                }
            }
            Op::S2r { dst, sr } if self.stage == Stage::Compute => {
                use super::decode::sr as s;
                // LocalInvocationId 27, WorkgroupId 26.
                let v = match sr {
                    s::TID_X | s::TID_Y | s::TID_Z => {
                        self.builtin_component(27, u32::from(sr - s::TID_X))
                    }
                    s::CTAID_X | s::CTAID_Y | s::CTAID_Z => {
                        self.builtin_component(26, u32::from(sr - s::CTAID_X))
                    }
                    // SR_TID packed: x | y << 16 | z << 26.
                    32 => {
                        let x = self.builtin_component(27, 0);
                        let y = self.builtin_component(27, 1);
                        let z = self.builtin_component(27, 2);
                        let (s16, s26) = (self.constant_u(16), self.constant_u(26));
                        let y = self.op2(op::SHIFT_LEFT_LOGICAL, uint, y, s16);
                        let z = self.op2(op::SHIFT_LEFT_LOGICAL, uint, z, s26);
                        let xy = self.op2(op::BITWISE_OR, uint, x, y);
                        self.op2(op::BITWISE_OR, uint, xy, z)
                    }
                    _ => {
                        let v = s::constant(sr).unwrap_or(0);
                        self.constant_u(v)
                    }
                };
                self.store_reg(dst, v);
            }
            Op::S2r { dst, sr } => {
                // One invocation of a graphics stage: ids are zero.
                let v = super::decode::sr::constant(sr).unwrap_or(0);
                let v = self.constant_u(v);
                self.store_reg(dst, v);
            }
            Op::Lds {
                space: Space::Shared,
                dst,
                addr,
                uaddr,
                offset,
                size,
            } => {
                let at = self.shared_address(addr, uaddr, offset);
                let two = self.constant_u(2);
                let base = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, at, two);
                let bytes = super::interp::access_bytes(size) as u32;
                if bytes >= 4 {
                    for k in 0..bytes / 4 {
                        let kk = self.constant_u(k);
                        let i = self.op2(op::I_ADD, uint, base, kk);
                        let p = self.shared_word(i)?;
                        let v = self.load(uint, p);
                        if dst != RZ {
                            self.store_reg(dst + k as u8, v);
                        }
                    }
                } else {
                    let p = self.shared_word(base)?;
                    let w = self.load(uint, p);
                    let three = self.constant_u(3);
                    let b = self.op2(op::BITWISE_AND, uint, at, three);
                    let eight = self.constant_u(8);
                    let shift = self.op2(op::I_MUL, uint, b, eight);
                    let count = self.constant_u(bytes * 8);
                    let opcode = if size == 1 || size == 3 {
                        op::BIT_FIELD_S_EXTRACT
                    } else {
                        op::BIT_FIELD_U_EXTRACT
                    };
                    let r = self.m.id();
                    inst(&mut self.m.body, opcode, &[uint, r, w, shift, count]);
                    self.store_reg(dst, r);
                }
            }
            Op::Sts {
                space: Space::Shared,
                src,
                addr,
                uaddr,
                offset,
                size,
            } => {
                let at = self.shared_address(addr, uaddr, offset);
                let two = self.constant_u(2);
                let base = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, at, two);
                let bytes = super::interp::access_bytes(size) as u32;
                if bytes >= 4 {
                    for k in 0..bytes / 4 {
                        let kk = self.constant_u(k);
                        let i = self.op2(op::I_ADD, uint, base, kk);
                        let p = self.shared_word(i)?;
                        let r = if src == RZ { RZ } else { src + k as u8 };
                        let v = self.load_reg(r);
                        self.store(p, v);
                    }
                } else {
                    let v = self.load_reg(src);
                    let three = self.constant_u(3);
                    let b = self.op2(op::BITWISE_AND, uint, at, three);
                    let eight = self.constant_u(8);
                    let shift = self.op2(op::I_MUL, uint, b, eight);
                    let field = self.constant_u(if bytes == 1 { 0xFF } else { 0xFFFF });
                    let mask = self.op2(op::SHIFT_LEFT_LOGICAL, uint, field, shift);
                    let keep = self.op1(op::NOT, uint, mask);
                    let v = self.op2(op::BITWISE_AND, uint, v, field);
                    let v = self.op2(op::SHIFT_LEFT_LOGICAL, uint, v, shift);
                    let p = self.shared_word(base)?;
                    self.atomic_scoped(op::ATOMIC_AND, p, keep, 2);
                    self.atomic_scoped(op::ATOMIC_OR, p, v, 2);
                }
            }
            Op::Atom {
                space: Space::Shared,
                dst,
                addr,
                uaddr,
                offset,
                src,
                op: aop,
                signed,
                ..
            } => {
                let opcode = atomic_opcode(aop, signed)?;
                let at = self.shared_address(addr, uaddr, offset);
                let two = self.constant_u(2);
                let i = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, at, two);
                let p = self.shared_word(i)?;
                let v = self.load_reg(src);
                let r = self.atomic_scoped(opcode, p, v, 2);
                self.store_reg(dst, r);
            }
            Op::Bar if self.stage == Stage::Compute => {
                // Workgroup execution and memory barrier
                // (AcquireRelease | WorkgroupMemory).
                let (wg, sem) = (self.constant_u(2), self.constant_u(0x108));
                inst(&mut self.m.body, op::CONTROL_BARRIER, &[wg, wg, sem]);
            }
            Op::Ldc {
                dst,
                index,
                base,
                offset,
                wide,
            } => {
                let var = self.cbuf_var(index);
                let b = self.load_reg(base);
                let off = self.constant_u(offset);
                let at = self.op2(op::I_ADD, uint, b, off);
                for k in 0..if wide { 2 } else { 1 } {
                    let target = if dst == RZ { RZ } else { dst + k };
                    let four = self.constant_u(4 * u32::from(k));
                    let at = self.op2(op::I_ADD, uint, at, four);
                    let (s4, s2, m3, last) = (
                        self.constant_u(4),
                        self.constant_u(2),
                        self.constant_u(3),
                        self.constant_u(4095),
                    );
                    let vec = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, at, s4);
                    let vec = self.glsl(uint, GLSL_UMIN, &[vec, last]);
                    let word = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, at, s2);
                    let comp = self.op2(op::BITWISE_AND, uint, word, m3);
                    let p = self.m.id();
                    inst(
                        &mut self.m.body,
                        op::ACCESS_CHAIN,
                        &[self.c.p_uniform_uint, p, var, self.c.zero_u, vec, comp],
                    );
                    let v = self.load(uint, p);
                    self.store_reg(target, v);
                }
            }
            Op::Lds {
                space: Space::Local,
                dst,
                addr,
                uaddr,
                offset,
                size,
            } if size >= 4 => {
                let words = super::interp::access_bytes(size) as u32 / 4;
                let base = self.local_index(addr, uaddr, offset);
                for k in 0..words {
                    let kk = self.constant_u(k);
                    let i = self.op2(op::I_ADD, uint, base, kk);
                    let p = self.local_word(i);
                    let v = self.load(uint, p);
                    if dst != RZ {
                        self.store_reg(dst + k as u8, v);
                    }
                }
            }
            Op::Sts {
                space: Space::Local,
                src,
                addr,
                uaddr,
                offset,
                size,
            } if size >= 4 => {
                let words = super::interp::access_bytes(size) as u32 / 4;
                let base = self.local_index(addr, uaddr, offset);
                for k in 0..words {
                    let kk = self.constant_u(k);
                    let i = self.op2(op::I_ADD, uint, base, kk);
                    let p = self.local_word(i);
                    let r = if src == RZ { RZ } else { src + k as u8 };
                    let v = self.load_reg(r);
                    self.store(p, v);
                }
            }
            Op::Lds { .. }
            | Op::Sts { .. }
            | Op::Bar
            | Op::Atom {
                space: Space::Local,
                ..
            } => {
                return Err("this memory access or barrier is not translated in this stage".into());
            }
            Op::Half2 {
                dst,
                op: hop,
                a,
                b,
                c,
                sat,
            } => {
                let x = self.half_lanes(a)?;
                let y = self.half_lanes(b)?;
                let z = self.half_lanes(c)?;
                let mut r = [0u32; 2];
                for k in 0..2 {
                    let v = match hop {
                        HalfOp::Add => self.op2(op::F_ADD, float, x[k], y[k]),
                        HalfOp::Mul => self.op2(op::F_MUL, float, x[k], y[k]),
                        HalfOp::Fma => self.glsl(float, GLSL_FMA, &[x[k], y[k], z[k]]),
                        HalfOp::Mnmx(p) => {
                            let lo = self.glsl(float, GLSL_NMIN, &[x[k], y[k]]);
                            let hi = self.glsl(float, GLSL_NMAX, &[x[k], y[k]]);
                            let c = self.load_pred(p);
                            self.select(float, c, lo, hi)
                        }
                    };
                    r[k] = if sat {
                        let (zero, one) = (self.c.zero_f, self.c.one_f);
                        self.glsl(float, GLSL_FCLAMP, &[v, zero, one])
                    } else {
                        v
                    };
                }
                let packed = self.pack_halves(r[0], r[1]);
                self.store_reg(dst, packed);
            }
            Op::Hsetp2 {
                dst,
                dst2,
                cmp,
                a,
                b,
                bop,
                c,
                h_and,
            } => {
                let x = self.half_lanes(a)?;
                let y = self.half_lanes(b)?;
                let r0 = self.fcmp(cmp, x[0], y[0]);
                let r1 = self.fcmp(cmp, x[1], y[1]);
                let c = self.load_pred(c);
                let bool_t = self.c.bool_t;
                let (p, q) = if h_and {
                    let both = self.op2(op::LOGICAL_AND, bool_t, r0, r1);
                    let not = self.op1(op::LOGICAL_NOT, bool_t, both);
                    (self.bop(bop, both, c), self.bop(bop, not, c))
                } else {
                    (self.bop(bop, r0, c), self.bop(bop, r1, c))
                };
                self.store_pred(dst, p);
                self.store_pred(dst2, q);
            }
            Op::Hset2 {
                dst,
                cmp,
                a,
                b,
                bop,
                c,
                bf,
            } => {
                let x = self.half_lanes(a)?;
                let y = self.half_lanes(b)?;
                let c = self.load_pred(c);
                let on = self.constant_u(if bf { 0x3C00 } else { 0xFFFF });
                let mut lanes = [0u32; 2];
                for k in 0..2 {
                    let r = self.fcmp(cmp, x[k], y[k]);
                    let r = self.bop(bop, r, c);
                    lanes[k] = self.select(uint, r, on, self.c.zero_u);
                }
                let sixteen = self.constant_u(16);
                let hi = self.op2(op::SHIFT_LEFT_LOGICAL, uint, lanes[1], sixteen);
                let v = self.op2(op::BITWISE_OR, uint, lanes[0], hi);
                self.store_reg(dst, v);
            }
            Op::F2fp { dst, a, b, merge } => {
                let fb = self.operand_float(b, no)?;
                let v = if merge {
                    let raw = self.operand_uint(a)?;
                    let low = self.pack_halves(fb, self.c.zero_f);
                    let m = self.constant_u(0xFFFF);
                    let ra = self.op2(op::BITWISE_AND, uint, raw, m);
                    let sixteen = self.constant_u(16);
                    let hi = self.op2(op::SHIFT_LEFT_LOGICAL, uint, ra, sixteen);
                    self.op2(op::BITWISE_OR, uint, low, hi)
                } else {
                    let fa = self.operand_float(a, no)?;
                    self.pack_halves(fb, fa)
                };
                self.store_reg(dst, v);
            }
            Op::F2h { dst, src, mods } => {
                let f = self.operand_float(src, mods)?;
                let v = self.pack_halves(f, self.c.zero_f);
                self.store_reg(dst, v);
            }
            Op::P2r {
                dst,
                a,
                mask,
                shift,
            } => {
                let mut preds = self.c.zero_u;
                for i in 0..7u8 {
                    let p = self.load_pred(Pred {
                        index: i,
                        neg: false,
                    });
                    let bit = self.constant_u(1 << i);
                    let v = self.select(uint, p, bit, self.c.zero_u);
                    preds = self.op2(op::BITWISE_OR, uint, preds, v);
                }
                let m = self.constant_u(mask);
                let keep = self.constant_u(!(mask << shift));
                let sh = self.constant_u(shift);
                let x = self.load_reg(a);
                let x = self.op2(op::BITWISE_AND, uint, x, keep);
                let p = self.op2(op::BITWISE_AND, uint, preds, m);
                let p = self.op2(op::SHIFT_LEFT_LOGICAL, uint, p, sh);
                let v = self.op2(op::BITWISE_OR, uint, x, p);
                self.store_reg(dst, v);
            }
            Op::R2p { a, mask, shift } => {
                let x = self.load_reg(a);
                for i in 0..7u32 {
                    if mask >> i & 1 == 0 {
                        continue;
                    }
                    let sh = self.constant_u(shift + i);
                    let one = self.constant_u(1);
                    let v = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, x, sh);
                    let v = self.op2(op::BITWISE_AND, uint, v, one);
                    let p = self.op2(op::I_NOT_EQUAL, self.c.bool_t, v, self.c.zero_u);
                    self.store_pred(i as u8, p);
                }
            }
            Op::TexN {
                dst0,
                dst1,
                dim,
                coords,
                layer,
                handle,
                comps,
                fetch,
                auto_lod,
            } => {
                let t = self.texture_any(dim, handle, coords, layer, fetch, auto_lod)?;
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.store_float(super::interp::texs_dst(dst0, dst1, k), t[c], false);
                }
            }
            Op::Ldg {
                dst,
                addr,
                wide,
                uaddr,
                offset,
                size,
            } => {
                let (lo, hi) = self.global_address(addr, wide, uaddr, offset);
                let bytes = super::interp::access_bytes(size) as u32;
                if bytes >= 4 {
                    for k in 0..bytes / 4 {
                        let kk = self.constant_u(4 * k);
                        let (l, h) = self.add64(lo, hi, kk, self.c.zero_u);
                        let v = self.global_load_word(l, h)?;
                        if dst != RZ {
                            self.store_reg(dst + k as u8, v);
                        }
                    }
                } else {
                    let w = self.global_load_word(lo, hi)?;
                    let three = self.constant_u(3);
                    let b = self.op2(op::BITWISE_AND, uint, lo, three);
                    let eight = self.constant_u(8);
                    let shift = self.op2(op::I_MUL, uint, b, eight);
                    let count = self.constant_u(bytes * 8);
                    let signed = size == 1 || size == 3;
                    let v = if signed {
                        let r = self.m.id();
                        inst(
                            &mut self.m.body,
                            op::BIT_FIELD_S_EXTRACT,
                            &[uint, r, w, shift, count],
                        );
                        r
                    } else {
                        let r = self.m.id();
                        inst(
                            &mut self.m.body,
                            op::BIT_FIELD_U_EXTRACT,
                            &[uint, r, w, shift, count],
                        );
                        r
                    };
                    self.store_reg(dst, v);
                }
            }
            Op::Stg {
                src,
                addr,
                wide,
                uaddr,
                offset,
                size,
            } => {
                self.storage_written = true;
                let (lo, hi) = self.global_address(addr, wide, uaddr, offset);
                let bytes = super::interp::access_bytes(size) as u32;
                if bytes >= 4 {
                    for k in 0..bytes / 4 {
                        let kk = self.constant_u(4 * k);
                        let (l, h) = self.add64(lo, hi, kk, self.c.zero_u);
                        let r = if src == RZ { RZ } else { src + k as u8 };
                        let v = self.load_reg(r);
                        self.global_at(l, h, |t, p| {
                            t.store(p, v);
                            Ok(())
                        })?;
                    }
                } else {
                    // Sub-word: clear the bytes, then OR the value in.
                    let v = self.load_reg(src);
                    let three = self.constant_u(3);
                    let b = self.op2(op::BITWISE_AND, uint, lo, three);
                    let eight = self.constant_u(8);
                    let shift = self.op2(op::I_MUL, uint, b, eight);
                    let field = self.constant_u(if bytes == 1 { 0xFF } else { 0xFFFF });
                    let mask = self.op2(op::SHIFT_LEFT_LOGICAL, uint, field, shift);
                    let keep = self.op1(op::NOT, uint, mask);
                    let v = self.op2(op::BITWISE_AND, uint, v, field);
                    let v = self.op2(op::SHIFT_LEFT_LOGICAL, uint, v, shift);
                    self.global_at(lo, hi, |t, p| {
                        t.atomic(op::ATOMIC_AND, p, keep);
                        t.atomic(op::ATOMIC_OR, p, v);
                        Ok(())
                    })?;
                }
            }
            Op::Atom {
                space: Space::Global,
                dst,
                addr,
                wide,
                uaddr,
                offset,
                src,
                op: aop,
                signed,
            } => {
                let opcode = atomic_opcode(aop, signed)?;
                self.storage_written = true;
                let (lo, hi) = self.global_address(addr, wide, uaddr, offset);
                let v = self.load_reg(src);
                let old = self.private_var(self.c.p_private_uint, self.c.zero_u);
                self.global_at(lo, hi, |t, p| {
                    let r = t.atomic(opcode, p, v);
                    t.store(old, r);
                    Ok(())
                })?;
                let r = self.load(uint, old);
                self.store_reg(dst, r);
            }
            Op::Plop3 {
                dst,
                dst2,
                a,
                b,
                c,
                lut,
                lut2,
            } => {
                let (x, y, z) = (self.load_pred(a), self.load_pred(b), self.load_pred(c));
                let r = self.bool_lut(lut, x, y, z);
                let r2 = self.bool_lut(lut2, x, y, z);
                self.store_pred(dst, r);
                self.store_pred(dst2, r2);
            }
            Op::IntUnary { dst, func, src } => {
                let x = self.operand_uint(src)?;
                let r = match func {
                    IntFunc::Popc => self.op1(op::BIT_COUNT, uint, x),
                    IntFunc::Brev => self.op1(op::BIT_REVERSE, uint, x),
                    IntFunc::Iabs => self.glsl(uint, GLSL_SABS, &[x]),
                    IntFunc::Flo { signed, shift } => {
                        let f = if signed {
                            GLSL_FIND_S_MSB
                        } else {
                            GLSL_FIND_U_MSB
                        };
                        let r = self.glsl(uint, f, &[x]);
                        if shift {
                            let none = self.constant_u(u32::MAX);
                            let is_none = self.op2(op::I_EQUAL, self.c.bool_t, r, none);
                            let k = self.constant_u(31);
                            let s = self.op2(op::I_SUB, uint, k, r);
                            self.select(uint, is_none, none, s)
                        } else {
                            r
                        }
                    }
                };
                self.store_reg(dst, r);
            }
            Op::Prmt { dst, a, sel, c } => {
                let x = self.load_reg(a);
                let s = self.operand_uint(sel)?;
                let z = self.operand_uint(c)?;
                let bool_t = self.c.bool_t;
                let mut r = self.c.zero_u;
                for i in 0..4u32 {
                    let (sh, m15, m3, m4, m8, ff, eight, seven) = (
                        self.constant_u(4 * i),
                        self.constant_u(15),
                        self.constant_u(3),
                        self.constant_u(4),
                        self.constant_u(8),
                        self.constant_u(0xFF),
                        self.constant_u(8),
                        self.constant_u(7),
                    );
                    let n = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, s, sh);
                    let n = self.op2(op::BITWISE_AND, uint, n, m15);
                    let which = self.op2(op::BITWISE_AND, uint, n, m4);
                    let hi = self.op2(op::I_NOT_EQUAL, bool_t, which, self.c.zero_u);
                    let src = self.select(uint, hi, z, x);
                    let byte = self.op2(op::BITWISE_AND, uint, n, m3);
                    let bits = self.op2(op::I_MUL, uint, byte, eight);
                    let b = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, src, bits);
                    let b = self.op2(op::BITWISE_AND, uint, b, ff);
                    let sign_bit = self.op2(op::SHIFT_RIGHT_LOGICAL, uint, b, seven);
                    let rep = self.op2(op::I_MUL, uint, sign_bit, ff);
                    let sign = self.op2(op::BITWISE_AND, uint, n, m8);
                    let signed = self.op2(op::I_NOT_EQUAL, bool_t, sign, self.c.zero_u);
                    let b = self.select(uint, signed, rep, b);
                    let at = self.constant_u(8 * i);
                    let b = self.op2(op::SHIFT_LEFT_LOGICAL, uint, b, at);
                    r = self.op2(op::BITWISE_OR, uint, r, b);
                }
                self.store_reg(dst, r);
            }
            Op::ImadWide {
                dst,
                a,
                b,
                c,
                signed,
            } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let pair = self.c.uint_pair;
                let mul = if signed {
                    op::S_MUL_EXTENDED
                } else {
                    op::U_MUL_EXTENDED
                };
                let p = self.op2(mul, pair, x, y);
                let (lo, hi) = (self.extract(uint, p, 0), self.extract(uint, p, 1));
                let (clo, chi) = if c == RZ {
                    (self.c.zero_u, self.c.zero_u)
                } else {
                    (self.load_reg(c), self.load_reg(c + 1))
                };
                let sum = self.op2(op::I_ADD_CARRY, pair, lo, clo);
                let (rlo, carry) = (self.extract(uint, sum, 0), self.extract(uint, sum, 1));
                let rhi = self.op2(op::I_ADD, uint, hi, chi);
                let rhi = self.op2(op::I_ADD, uint, rhi, carry);
                self.store_reg(dst, rlo);
                if dst != RZ {
                    self.store_reg(dst + 1, rhi);
                }
            }
            Op::ImadHi {
                dst,
                a,
                b,
                c,
                signed,
            } => {
                let x = self.load_reg(a);
                let y = self.operand_uint(b)?;
                let z = self.operand_uint(c)?;
                let mul = if signed {
                    op::S_MUL_EXTENDED
                } else {
                    op::U_MUL_EXTENDED
                };
                let p = self.op2(mul, self.c.uint_pair, x, y);
                let hi = self.extract(uint, p, 1);
                let r = self.op2(op::I_ADD, uint, hi, z);
                self.store_reg(dst, r);
            }
            Op::Vote { dst, pdst, src, eq } => {
                let p = self.load_pred(src);
                let one = self.constant_u(1);
                let v = self.select(uint, p, one, self.c.zero_u);
                self.store_reg(dst, v);
                let out = if eq { self.c.true_c } else { p };
                self.store_pred(pdst, out);
            }
            Op::Txq {
                dst,
                dst1,
                lod,
                handle,
                comps,
            } => {
                let (_, img) = self.texture_image(handle)?;
                let l = self.load_reg(lod);
                let size = self.op2(op::IMAGE_QUERY_SIZE_LOD, self.c.uvec2, img, l);
                let levels = self.op1(op::IMAGE_QUERY_LEVELS, uint, img);
                let one = self.constant_u(1);
                let vals = [
                    self.extract(uint, size, 0),
                    self.extract(uint, size, 1),
                    one,
                    levels,
                ];
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.store_reg(super::interp::texs_dst(dst, dst1, k), vals[c]);
                }
            }
            Op::Tld4 {
                dst0,
                dst1,
                a,
                b,
                handle,
                comp,
                comps,
            } => {
                let u = self.reg_float(a, no);
                let v = self.reg_float(b, no);
                let var = self.texture_var(handle)?;
                let img = self.load(self.c.sampled_image, var);
                let coord = self.m.id();
                inst(
                    &mut self.m.body,
                    op::COMPOSITE_CONSTRUCT,
                    &[self.c.vec2, coord, u, v],
                );
                let which = self.constant_u(u32::from(comp));
                let texel = self.m.id();
                inst(
                    &mut self.m.body,
                    op::IMAGE_GATHER,
                    &[self.c.vec4, texel, img, coord, which],
                );
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    let f = self.extract(float, texel, c as u32);
                    self.store_float(super::interp::texs_dst(dst0, dst1, k), f, false);
                }
            }
            Op::Nop => {}
            Op::Bra { .. } | Op::Brx { .. } | Op::Ssy { .. } | Op::Sync | Op::Kil | Op::Exit => {
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
                    Op::Kil if self.stage == Stage::Compute => {
                        return Err("KIL in a compute program".into());
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

/// The SPIR-V atomic for an ATOM operation.
fn atomic_opcode(aop: AtomOp, signed: bool) -> Result<u16, String> {
    Ok(match (aop, signed) {
        (AtomOp::Add, _) => op::ATOMIC_I_ADD,
        (AtomOp::Min, true) => op::ATOMIC_S_MIN,
        (AtomOp::Min, false) => op::ATOMIC_U_MIN,
        (AtomOp::Max, true) => op::ATOMIC_S_MAX,
        (AtomOp::Max, false) => op::ATOMIC_U_MAX,
        (AtomOp::And, _) => op::ATOMIC_AND,
        (AtomOp::Or, _) => op::ATOMIC_OR,
        (AtomOp::Xor, _) => op::ATOMIC_XOR,
        (AtomOp::Exch, _) => op::ATOMIC_EXCHANGE,
        (AtomOp::Inc | AtomOp::Dec, _) => {
            return Err("atomic INC/DEC are not translated yet".into())
        }
    })
}

/// Block number meaning "the program has ended".
const EXIT: u32 = u32::MAX;

/// Translates a vertex or pixel program to a SPIR-V module with entry
/// point "main".
pub fn translate(program: &[Inst], header: &Header) -> Result<Translated, String> {
    translate_stage(program, header.stage, Some(header), None)
}

/// Translates a compute program for workgroups of `block` invocations
/// with `shared_bytes` of shared memory (entry point "main").
pub fn translate_compute(
    program: &[Inst],
    block: [u32; 3],
    shared_bytes: u32,
) -> Result<Translated, String> {
    translate_stage(program, Stage::Compute, None, Some((block, shared_bytes)))
}

fn translate_stage(
    program: &[Inst],
    stage: Stage,
    header: Option<&Header>,
    compute: Option<([u32; 3], u32)>,
) -> Result<Translated, String> {
    let model = match stage {
        Stage::Vertex => 0,
        Stage::Pixel => 4,
        Stage::Compute => 5,
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
    let (uvec2, uint_pair) = (m.id(), m.id());
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
    inst(g, op::TYPE_VECTOR, &[uvec2, uint, 2]);
    inst(g, op::TYPE_STRUCT, &[uint_pair, uint, uint]);
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
        uvec2,
        image_t,
        sampled_image,
        p_sampled_image,
        uint_pair,
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
        image_types: BTreeMap::new(),
        vec3: None,
        uvec3: None,
        storage_slots: storage_candidates(program),
        storage_vars: Vec::new(),
        p_storage_uint: 0,
        storage_written: false,
        builtins: BTreeMap::new(),
        shared: None,
        shared_words: compute.map_or(0, |(_, b)| b.div_ceil(4)),
        shared_p_word: 0,
        local: None,
        stage,
    };
    let main = t.m.id();
    let label = t.m.id();
    t.translate(program, stage)?;

    // Pixel shader outputs: each target's enabled components from the
    // next registers (r0, r1, ... across targets in order), at the
    // target's location.
    let mut color_out = Vec::new();
    if let (Stage::Pixel, Some(header)) = (stage, header) {
        let mut next = 0u8;
        for (k, &mask) in header.omap_targets.iter().enumerate() {
            if mask == 0 {
                continue;
            }
            let out = t.m.id();
            inst(
                &mut t.m.globals,
                op::VARIABLE,
                &[t.c.p_out_vec4, out, OUTPUT],
            );
            inst(
                &mut t.m.annotations,
                op::DECORATE,
                &[out, LOCATION, k as u32],
            );
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
            color_out.push(out);
        }
    }
    for &v in &t.flat {
        inst(&mut t.m.annotations, op::DECORATE, &[v, FLAT]);
    }

    let mut words = Vec::new();
    let bound = t.m.next_id + 1;
    words.extend([0x0723_0203, 0x0001_0300, 0, bound, 0]);
    inst(&mut words, op::CAPABILITY, &[1]); // Shader
    inst(&mut words, op::CAPABILITY, &[50]); // ImageQuery
    if t.textures.iter().any(|t| t.2 == TexKind::CubeArray) {
        inst(&mut words, op::CAPABILITY, &[45]); // SampledCubeArray
    }
    let mut ext = vec![glsl];
    ext.extend(string("GLSL.std.450"));
    inst(&mut words, op::EXT_INST_IMPORT, &ext);
    inst(&mut words, op::MEMORY_MODEL, &[0, 1]);
    let mut entry = vec![model, main];
    entry.extend(string("main"));
    entry.extend(t.inputs.values().chain(t.outputs.values()).copied());
    entry.extend(color_out);
    entry.extend(t.builtins.values().copied());
    inst(&mut words, op::ENTRY_POINT, &entry);
    if stage == Stage::Pixel {
        inst(&mut words, op::EXECUTION_MODE, &[main, 7]); // OriginUpperLeft
    }
    if let Some((block, _)) = compute {
        // LocalSize.
        inst(
            &mut words,
            op::EXECUTION_MODE,
            &[main, 17, block[0].max(1), block[1].max(1), block[2].max(1)],
        );
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
        texture_kinds: t.textures.iter().map(|t| t.2).collect(),
        storage: if t.storage_vars.is_empty() {
            Vec::new()
        } else {
            t.storage_slots.clone()
        },
        storage_written: t.storage_written,
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
    fn switch2_vertex_shader_validates() {
        use crate::shader::sm86::asm::{exit, inst};
        // ALD 4 words; IADD3, LOP3, IMAD, SHF (both directions), FFMA, MUFU;
        // AST position.
        let insts = [
            inst(
                0x321,
                0,
                &[(24, 8, 255), (32, 8, 255), (40, 10, 0x80), (74, 2, 3)],
            ),
            inst(
                0x810,
                4,
                &[(24, 8, 0), (32, 32, 3), (64, 8, 1), (81, 6, 0x3F)],
            ),
            inst(
                0x212,
                5,
                &[
                    (24, 8, 4),
                    (32, 8, 1),
                    (64, 8, 2),
                    (72, 8, 0x96),
                    (81, 3, 1),
                    (87, 4, 15),
                ],
            ),
            inst(0x224, 6, &[(24, 8, 5), (32, 8, 4), (64, 8, 3)]),
            inst(
                0x819,
                7,
                &[(24, 8, 6), (32, 32, 5), (64, 8, 5), (73, 2, 3), (80, 1, 1)],
            ),
            inst(
                0x819,
                8,
                &[
                    (24, 8, 7),
                    (32, 32, 3),
                    (64, 8, 6),
                    (73, 2, 2),
                    (76, 1, 1),
                    (80, 1, 1),
                ],
            ),
            inst(0x223, 0, &[(24, 8, 0), (32, 8, 1), (64, 8, 2)]),
            inst(0x308, 1, &[(32, 8, 1), (74, 6, 1)]),
            inst(
                0x322,
                0,
                &[
                    (24, 8, 255),
                    (32, 8, 0),
                    (40, 10, 0x70),
                    (64, 8, 255),
                    (74, 2, 3),
                ],
            ),
            exit(),
        ];
        let mut code = crate::shader::sm86::asm::program(&insts);
        code[..0x50].copy_from_slice(&vertex_header());
        let h = super::super::Header::parse(&code);
        let p =
            crate::shader::sm86::decode_program(&code[crate::shader::sm86::HEADER_SIZE..]).unwrap();
        let t = translate(&p, &h).unwrap();
        validate(&t.words);
    }

    #[test]
    fn switch2_computed_branch_validates() {
        use crate::shader::sm86::asm::{exit, inst};
        let set = |v: u32| {
            inst(
                0x810,
                1,
                &[
                    (24, 8, 255),
                    (32, 32, v),
                    (64, 8, 255),
                    (81, 3, 7),
                    (84, 3, 7),
                ],
            )
        };
        let brx = inst(
            0x949,
            0,
            &[(24, 8, 0), (32, 32, (-16i32) as u32), (64, 18, 0x3FFFF)],
        );
        let insts = [brx, set(1), exit(), set(2), exit()];
        let mut code = crate::shader::sm86::asm::program(&insts);
        code[..0x50].copy_from_slice(&vertex_header());
        let h = super::super::Header::parse(&code);
        let p = crate::shader::sm86::decode_program_ext(
            &code[crate::shader::sm86::HEADER_SIZE..],
            true,
        )
        .unwrap();
        let t = translate(&p, &h).unwrap();
        validate(&t.words);
    }

    #[test]
    fn switch2_integer_and_uniform_ops_validate() {
        use crate::shader::sm86::asm::{exit, inst};
        let insts = [
            inst(0x309, 2, &[(32, 8, 0)]),
            inst(0x300, 4, &[(32, 8, 0), (74, 1, 1), (81, 3, 7)]),
            inst(0x816, 5, &[(24, 8, 0), (32, 32, 0x7610), (64, 8, 1)]),
            inst(0x225, 6, &[(24, 8, 0), (32, 8, 1), (64, 8, 8), (81, 3, 7)]),
            inst(
                0x227,
                9,
                &[(24, 8, 0), (32, 8, 1), (64, 8, 255), (73, 1, 1), (81, 3, 7)],
            ),
            inst(
                0x88C,
                0,
                &[
                    (24, 8, 4),
                    (32, 32, 7),
                    (76, 3, 6),
                    (81, 3, 1),
                    (84, 3, 7),
                    (87, 4, 7),
                ],
            ),
            inst(0x887, 5, &[(24, 8, 4), (32, 32, 99), (87, 4, 1)]),
            inst(
                0xB82,
                2,
                &[(24, 8, 0), (38, 16, 0x10), (54, 5, 3), (73, 3, 5)],
            ),
            inst(0x919, 4, &[(72, 8, 56)]),
            inst(0x806, 10, &[(81, 3, 1), (87, 4, 7)]),
            inst(
                0x210,
                5,
                &[(24, 8, 1), (32, 8, 3), (64, 8, 255), (81, 3, 2), (84, 3, 3)],
            ),
            inst(
                0x210,
                6,
                &[
                    (24, 8, 1),
                    (72, 1, 1),
                    (32, 8, 3),
                    (64, 8, 255),
                    (74, 1, 1),
                    (87, 4, 2),
                    (77, 4, 0xF),
                ],
            ),
            inst(
                0x211,
                7,
                &[
                    (24, 8, 0),
                    (32, 8, 3),
                    (64, 8, 1),
                    (75, 5, 4),
                    (80, 1, 1),
                    (74, 1, 1),
                    (81, 3, 4),
                    (87, 4, 3),
                ],
            ),
            inst(
                0x81C,
                0,
                &[
                    (87, 4, 1),
                    (77, 4, 7),
                    (68, 4, 7),
                    (64, 3, 7),
                    (72, 5, 1),
                    (81, 3, 2),
                    (84, 3, 7),
                ],
            ),
            inst(
                0x322,
                0,
                &[
                    (24, 8, 255),
                    (32, 8, 0),
                    (40, 10, 0x70),
                    (64, 8, 255),
                    (74, 2, 3),
                ],
            ),
            exit(),
        ];
        let mut code = crate::shader::sm86::asm::program(&insts);
        code[..0x50].copy_from_slice(&vertex_header());
        let h = super::super::Header::parse(&code);
        let p =
            crate::shader::sm86::decode_program(&code[crate::shader::sm86::HEADER_SIZE..]).unwrap();
        let t = translate(&p, &h).unwrap();
        validate(&t.words);
    }

    #[test]
    fn compute_validates() {
        use super::super::decode::sr;
        let p = vec![
            Inst::always(Op::S2r {
                dst: 0,
                sr: sr::TID_X,
            }),
            Inst::always(Op::S2r {
                dst: 1,
                sr: sr::CTAID_Y,
            }),
            Inst::always(Op::S2r { dst: 6, sr: 32 }),
            Inst::always(Op::Sts {
                space: Space::Shared,
                src: 1,
                addr: 0,
                uaddr: RZ,
                offset: 0,
                size: 4,
            }),
            Inst::always(Op::Sts {
                space: Space::Shared,
                src: 1,
                addr: 0,
                uaddr: RZ,
                offset: 1,
                size: 0,
            }),
            Inst::always(Op::Bar),
            Inst::always(Op::Lds {
                space: Space::Shared,
                dst: 2,
                addr: 0,
                uaddr: RZ,
                offset: 4,
                size: 5,
            }),
            Inst::always(Op::Atom {
                space: Space::Shared,
                dst: 4,
                addr: 0,
                wide: false,
                uaddr: RZ,
                offset: 0,
                src: 1,
                op: AtomOp::Max,
                signed: true,
            }),
            Inst::always(Op::Ldc {
                dst: 8,
                index: 0,
                base: RZ,
                offset: 0x120,
                wide: true,
            }),
            Inst::always(Op::Stg {
                src: 2,
                addr: 8,
                wide: true,
                uaddr: RZ,
                offset: 0,
                size: 4,
            }),
            Inst::always(Op::Exit),
        ];
        let t = translate_compute(&p, [64, 2, 1], 1024).unwrap();
        assert_eq!(t.storage.len(), 1);
        validate(&t.words);
    }

    #[test]
    fn global_memory_validates() {
        let p = vec![
            Inst::always(Op::Ldc {
                dst: 2,
                index: 0,
                base: RZ,
                offset: 0x110,
                wide: true,
            }),
            Inst::always(Op::Ldg {
                dst: 4,
                addr: 2,
                wide: true,
                uaddr: RZ,
                offset: 8,
                size: 6,
            }),
            Inst::always(Op::Ldg {
                dst: 8,
                addr: 2,
                wide: true,
                uaddr: RZ,
                offset: -3,
                size: 1,
            }),
            Inst::always(Op::Stg {
                src: 4,
                addr: 2,
                wide: true,
                uaddr: RZ,
                offset: 0,
                size: 5,
            }),
            Inst::always(Op::Stg {
                src: 4,
                addr: 2,
                wide: true,
                uaddr: RZ,
                offset: 2,
                size: 2,
            }),
            Inst::always(Op::Atom {
                space: Space::Global,
                dst: 9,
                addr: 2,
                wide: true,
                uaddr: RZ,
                offset: 0,
                src: 4,
                op: AtomOp::Add,
                signed: false,
            }),
            Inst::always(Op::Exit),
        ];
        let mut code = vec![0u8; 0x50];
        code.copy_from_slice(&vertex_header());
        let h = super::super::Header::parse(&code);
        let t = translate(&p, &h).unwrap();
        assert_eq!(
            t.storage,
            [StorageBuffer {
                cbuf: 0,
                offset: 0x110
            }]
        );
        assert!(t.storage_written);
        validate(&t.words);
    }

    #[test]
    fn texture_kinds_validate() {
        let mut p = Vec::new();
        for (k, (dim, fetch)) in [
            (0u8, false),
            (2, false),
            (3, false),
            (5, false),
            (7, false),
            (5, true),
            (2, true),
            (4, true),
        ]
        .into_iter()
        .enumerate()
        {
            p.push(Inst::always(Op::TexN {
                dst0: 8,
                dst1: 10,
                dim,
                coords: [0, 1, 2],
                layer: 3,
                handle: 0x100 + 4 * k as u32,
                comps: 0xF,
                fetch,
                auto_lod: true,
            }));
        }
        p.push(Inst::always(Op::Exit));
        let mut code = vec![0u8; 0x50];
        code.copy_from_slice(&vertex_header());
        let h = super::super::Header::parse(&code);
        let t = translate(&p, &h).unwrap();
        assert_eq!(
            t.texture_kinds,
            [
                TexKind::D2,
                TexKind::D3,
                TexKind::Cube,
                TexKind::D2Array,
                TexKind::CubeArray,
                TexKind::D2Array,
                TexKind::D3,
                TexKind::D2Array
            ]
        );
        validate(&t.words);
    }

    #[test]
    fn switch2_half_ops_validate() {
        use crate::shader::sm86::asm::{exit, inst};
        let insts = [
            inst(0x230, 2, &[(24, 8, 0), (32, 8, 1)]),
            inst(0x231, 4, &[(24, 8, 0), (32, 8, 1), (64, 8, 0), (77, 1, 1)]),
            inst(
                0x234,
                0,
                &[
                    (24, 8, 0),
                    (32, 8, 1),
                    (76, 4, 4),
                    (81, 3, 0),
                    (84, 3, 1),
                    (87, 4, 7),
                ],
            ),
            inst(
                0x233,
                6,
                &[(24, 8, 0), (32, 8, 1), (76, 4, 1), (87, 4, 7), (71, 1, 1)],
            ),
            inst(0x240, 7, &[(24, 8, 0), (32, 8, 1), (87, 4, 7)]),
            inst(0x23E, 5, &[(24, 8, 6), (32, 8, 7)]),
            inst(0x803, 8, &[(24, 8, 9), (32, 32, 3)]),
            inst(0x804, 0, &[(24, 8, 9), (32, 32, 3)]),
            inst(
                0x322,
                0,
                &[
                    (24, 8, 255),
                    (32, 8, 0),
                    (40, 10, 0x70),
                    (64, 8, 255),
                    (74, 2, 3),
                ],
            ),
            exit(),
        ];
        let mut code = crate::shader::sm86::asm::program(&insts);
        code[..0x50].copy_from_slice(&vertex_header());
        let h = super::super::Header::parse(&code);
        let p =
            crate::shader::sm86::decode_program(&code[crate::shader::sm86::HEADER_SIZE..]).unwrap();
        let t = translate(&p, &h).unwrap();
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
