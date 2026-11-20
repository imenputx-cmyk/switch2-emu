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
//! Straight-line programs only for now (no branches). Anything
//! unsupported returns an error so the caller can fall back to the
//! software renderer.

use super::decode::{Inst, IpaMode, Mods, Operand, RZ};
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
}

// Storage classes, decorations, built-ins, GLSL.std.450 instructions.
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
}

/// Descriptor binding of constant buffer `index` for a stage.
pub fn cbuf_binding(stage: Stage, index: u32) -> u32 {
    match stage {
        Stage::Pixel => 18 + index,
        _ => index,
    }
}

/// A translated shader.
pub struct Translated {
    pub words: Vec<u32>,
    /// Bit `i` set: constant buffer `i` is read.
    pub cbufs: u32,
}

struct Translator {
    m: Module,
    c: Common,
    regs: BTreeMap<u8, u32>,
    inputs: BTreeMap<u32, u32>,
    outputs: BTreeMap<u32, u32>,
    flat: BTreeSet<u32>,
    cbufs: BTreeMap<u8, u32>,
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

    fn translate(&mut self, program: &[Inst], stage: Stage) -> Result<(), String> {
        for &i in program {
            match i {
                Inst::Mov { dst, src } => {
                    let v = self.operand_uint(src)?;
                    self.store_reg(dst, v);
                }
                Inst::Fadd {
                    dst,
                    a,
                    a_mods,
                    b,
                    b_mods,
                    sat,
                } => {
                    let x = self.load_reg(a);
                    let x = self.bitcast_float(x);
                    let x = self.mods(x, a_mods);
                    let y = self.operand_uint(b)?;
                    let y = self.bitcast_float(y);
                    let y = self.mods(y, b_mods);
                    let r = self.binary(op::F_ADD, x, y);
                    self.store_float(dst, r, sat);
                }
                Inst::Fmul {
                    dst,
                    a,
                    b,
                    neg_b,
                    sat,
                } => {
                    let x = self.load_reg(a);
                    let x = self.bitcast_float(x);
                    let y = self.operand_uint(b)?;
                    let y = self.bitcast_float(y);
                    let y = self.mods(
                        y,
                        Mods {
                            neg: neg_b,
                            abs: false,
                        },
                    );
                    let r = self.binary(op::F_MUL, x, y);
                    self.store_float(dst, r, sat);
                }
                Inst::Ffma {
                    dst,
                    a,
                    b,
                    neg_b,
                    c,
                    neg_c,
                    sat,
                } => {
                    let x = self.load_reg(a);
                    let x = self.bitcast_float(x);
                    let y = self.operand_uint(b)?;
                    let y = self.bitcast_float(y);
                    let y = self.mods(
                        y,
                        Mods {
                            neg: neg_b,
                            abs: false,
                        },
                    );
                    let z = self.operand_uint(c)?;
                    let z = self.bitcast_float(z);
                    let z = self.mods(
                        z,
                        Mods {
                            neg: neg_c,
                            abs: false,
                        },
                    );
                    let r = self.glsl(self.c.float, GLSL_FMA, &[x, y, z]);
                    self.store_float(dst, r, sat);
                }
                Inst::Ald {
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
                        let f = self.m.id();
                        inst(&mut self.m.body, op::LOAD, &[self.c.float, f, p]);
                        let u = self.bitcast(self.c.uint, f);
                        self.store_reg(dst.wrapping_add(k as u8), u);
                    }
                }
                Inst::Ast {
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
                        inst(&mut self.m.body, op::STORE, &[p, f]);
                    }
                }
                Inst::Ipa {
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
                    let f = self.m.id();
                    inst(&mut self.m.body, op::LOAD, &[self.c.float, f, p]);
                    self.store_float(dst, f, sat);
                }
                Inst::Nop => {}
                Inst::Exit => break,
            }
        }
        Ok(())
    }
}

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
    };
    let mut t = Translator {
        m,
        c,
        regs: BTreeMap::new(),
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
        flat: BTreeSet::new(),
        cbufs: BTreeMap::new(),
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
}
