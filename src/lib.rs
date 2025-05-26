pub mod decoder;
use anyhow::Result;
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

pub type Word = i64;
pub type Pid = usize;

const REG_COUNT: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct Reg(pub u8);

#[derive(Clone, Copy, Debug)]
pub struct Addr(pub usize);

struct HandlerOptions<'a> {
    proc: &'a mut Process,
    process_table: &'a mut BTreeMap<Pid, Rc<RefCell<Process>>>,
    haulted: &'a mut Vec<Rc<RefCell<Process>>>,
    run_queue: &'a mut VecDeque<Rc<RefCell<Process>>>,
    modules: &'a BTreeMap<String, Module>,
    next_pid: &'a mut usize,
}

type OpHandler = fn(ho: HandlerOptions);

static DISPATCH_TABLE: &[OpHandler] = &[
    handle_noop,
    handle_halt,
    handle_load_imm,
    handle_add,
    handle_sub,
    handle_mul,
    handle_cmp_le,
    handle_mov,
    handle_jump_if,
    handle_jump,
    handle_spawn,
    handle_print,
    handle_send,
    handle_recv,
];

#[inline(always)]
fn handle_noop(_ho: HandlerOptions) {}

#[inline(always)]
fn handle_halt(ho: HandlerOptions) {
    ho.proc.state = ProcessState::Halted;
}

#[inline(always)]
fn handle_load_imm(ho: HandlerOptions) {
    let Ok(dst) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(value) = ho.proc.imm_i64() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    ho.proc.regs[dst] = value;
}

#[inline(always)]
fn handle_add(ho: HandlerOptions) {
    let Ok(dst) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(lhs) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(rhs) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    ho.proc.regs[dst] = ho.proc.regs[lhs] + ho.proc.regs[rhs];
}

#[inline(always)]
fn handle_sub(ho: HandlerOptions) {
    let Ok(dst) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(lhs) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(rhs) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    ho.proc.regs[dst] = ho.proc.regs[lhs] - ho.proc.regs[rhs];
}

#[inline(always)]
fn handle_mul(ho: HandlerOptions) {
    let Ok(dst) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(lhs) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(rhs) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    ho.proc.regs[dst] = ho.proc.regs[lhs] * ho.proc.regs[rhs];
}

#[inline(always)]
fn handle_cmp_le(ho: HandlerOptions) {
    let Ok(dst) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(lhs) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(rhs) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    ho.proc.regs[dst] = (ho.proc.regs[lhs] <= ho.proc.regs[rhs]) as i64;
}

#[inline(always)]
fn handle_mov(ho: HandlerOptions) {
    let Ok(dst) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(src) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    ho.proc.regs[dst] = ho.proc.regs[src] as i64;
}

#[inline(always)]
fn handle_jump_if(ho: HandlerOptions) {
    let Ok(cmp) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(target) = ho.proc.imm_u32() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    if ho.proc.regs[cmp] != 0 {
        ho.proc.ip = target as usize;
    }
}

#[inline(always)]
fn handle_jump(ho: HandlerOptions) {
    let Ok(target) = ho.proc.imm_u32() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    ho.proc.ip = target as usize;
}

#[inline(always)]
fn handle_spawn(ho: HandlerOptions) {
    let Ok(module_name) = ho.proc.string() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(function_name) = ho.proc.string() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(args) = ho.proc.args() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(dst) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Some(module) = ho.modules.get(module_name.as_str()) else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Some(function) = module.functions.get(function_name.as_str()) else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let new_pid = *ho.next_pid;
    *ho.next_pid += 1;
    let mut args: Vec<Word> = args.iter().map(|&arg| ho.proc.regs[arg as usize]).collect();
    if function.returns {
        args.insert(0, ho.proc.pid as Word);
    }
    let new_proc = if let Some(proc_rc) = ho.haulted.pop() {
        proc_rc
            .borrow_mut()
            .reset(new_pid, Arc::clone(function), &args);
        proc_rc
    } else {
        Rc::new(RefCell::new(Process::new(
            new_pid,
            Arc::clone(function),
            &args,
        )))
    };
    ho.proc.regs[dst] = new_pid as i64;

    ho.process_table.insert(new_pid, Rc::clone(&new_proc));
    ho.run_queue.push_back(new_proc);
}

#[inline(always)]
fn handle_print(ho: HandlerOptions) {
    let Ok(arg) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    println!("{}", ho.proc.regs[arg]);
}

#[inline(always)]
fn handle_recv(ho: HandlerOptions) {
    let start = ho.proc.ip;
    let Ok(dst) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };

    let Some(msg) = ho.proc.mailbox.pop_front() else {
        ho.proc.ip = start - 1;
        ho.proc.state = ProcessState::Waiting;
        return;
    };

    ho.proc.regs[dst] = msg;
    ho.proc.state = ProcessState::Running;
}

#[inline(always)]
fn handle_send(ho: HandlerOptions) {
    let Ok(dst_pid) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let Ok(src_reg) = ho.proc.reg() else {
        ho.proc.state = ProcessState::Crashed;
        return;
    };
    let pid = ho.proc.regs[dst_pid] as Pid;
    if let Some(dst_proc) = ho.process_table.get(&pid) {
        let mut p = dst_proc.borrow_mut();
        p.mailbox.push_back(ho.proc.regs[src_reg]);
        ho.run_queue.push_back(Rc::clone(dst_proc));
    }
}

#[derive(Clone, Debug)]
pub enum Instruction {
    Noop,
    Halt,
    LoadImm {
        dst: Reg,
        value: Word,
    },
    Add {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Sub {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Mul {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    CmpLE {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Mov {
        dst: Reg,
        src: Reg,
    },
    JumpIf {
        cmp: Reg,
        target: String,
    },
    Jump {
        target: String,
    },
    Spawn {
        module: Box<String>,
        function: Box<String>,
        args: Vec<Reg>,
        dst: Reg,
    },
    Print {
        reg: Reg,
    },
    Send {
        dst_pid: Reg,
        src_reg: Reg,
    },
    Recv {
        dst_reg: Reg,
    },
    Label {
        name: String,
    },
}

impl Instruction {
    pub fn opcode(&self) -> u8 {
        match self {
            Instruction::Noop => 0,
            Instruction::Halt => 1,
            Instruction::LoadImm { .. } => 2,
            Instruction::Add { .. } => 3,
            Instruction::Sub { .. } => 4,
            Instruction::Mul { .. } => 5,
            Instruction::CmpLE { .. } => 6,
            Instruction::Mov { .. } => 7,
            Instruction::JumpIf { .. } => 8,
            Instruction::Jump { .. } => 9,
            Instruction::Spawn { .. } => 10,
            Instruction::Print { .. } => 11,
            Instruction::Send { .. } => 12,
            Instruction::Recv { .. } => 13,
            Instruction::Label { .. } => panic!("Label has no opcode"),
        }
    }

    pub fn into_bytes(self, index: usize, backpatch: &mut Vec<(usize, Instruction)>) -> Vec<u8> {
        let opcode = self.opcode();
        let mut bytes = vec![opcode];
        match self {
            Instruction::Noop | Instruction::Halt => {}
            Instruction::LoadImm { dst, value } => {
                bytes.push(dst.0);
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Instruction::Add { dst, lhs, rhs }
            | Instruction::Sub { dst, lhs, rhs }
            | Instruction::Mul { dst, lhs, rhs }
            | Instruction::CmpLE { dst, lhs, rhs } => {
                bytes.push(dst.0);
                bytes.push(lhs.0);
                bytes.push(rhs.0);
            }
            Instruction::Mov { dst, src } => {
                bytes.push(dst.0);
                bytes.push(src.0);
            }
            Instruction::JumpIf { cmp, target } => {
                bytes.push(cmp.0);
                bytes.extend_from_slice(&u32::MAX.to_le_bytes());
                backpatch.push((index, Instruction::JumpIf { cmp, target }));
            }
            Instruction::Jump { target } => {
                bytes.extend_from_slice(&u32::MAX.to_le_bytes());
                backpatch.push((index, Instruction::Jump { target }));
            }
            Instruction::Spawn {
                module,
                function,
                args,
                dst,
            } => {
                bytes.extend_from_slice(&(module.len() as u32).to_le_bytes());
                bytes.extend_from_slice(module.as_bytes());
                bytes.extend_from_slice(&(function.len() as u32).to_le_bytes());
                bytes.extend_from_slice(function.as_bytes());
                bytes.extend_from_slice(&(args.len() as u32).to_le_bytes());
                for arg in args {
                    bytes.push(arg.0);
                }
                bytes.push(dst.0);
            }
            Instruction::Print { reg } => {
                bytes.push(reg.0);
            }
            Instruction::Send { dst_pid, src_reg } => {
                bytes.push(dst_pid.0);
                bytes.push(src_reg.0);
            }
            Instruction::Recv { dst_reg } => {
                bytes.push(dst_reg.0);
            }
            Instruction::Label { .. } => panic!("Label has no bytes"),
        }
        bytes
    }
}

pub struct InstructionBuilder<'a> {
    bytecode: &'a mut Vec<u8>,
    instructions: Vec<Instruction>,
}

impl Drop for InstructionBuilder<'_> {
    fn drop(&mut self) {
        let instructions = std::mem::take(&mut self.instructions);
        let mut labels = std::collections::BTreeMap::new();
        let mut backpatch = Vec::new();
        for instruction in instructions {
            if let Instruction::Label { name } = instruction {
                labels.insert(name, self.bytecode.len());
                continue;
            }
            self.bytecode
                .extend_from_slice(&instruction.into_bytes(self.bytecode.len(), &mut backpatch));
        }

        for (instruction_index, instruction) in backpatch {
            match instruction {
                Instruction::JumpIf { target, .. } => {
                    let Some(target) = labels.get(&target) else {
                        panic!("Label {target} not found");
                    };
                    let bytes = (*target as u32).to_le_bytes();
                    for (i, byte) in bytes.iter().enumerate() {
                        self.bytecode[instruction_index + i + 2] = *byte;
                    }
                }
                Instruction::Jump { target } => {
                    let Some(target) = labels.get(&target) else {
                        panic!("Label {target} not found");
                    };
                    let bytes = (*target as u32).to_le_bytes();
                    for (i, byte) in bytes.iter().enumerate() {
                        self.bytecode[instruction_index + i + 1] = *byte;
                    }
                }
                _ => panic!("Instruction is not JumpIf or Jump"),
            }
        }
    }
}

impl<'a> InstructionBuilder<'a> {
    pub fn new(bytecode: &'a mut Vec<u8>) -> Self {
        Self {
            instructions: Vec::new(),
            bytecode,
        }
    }

    pub fn noop(&mut self) -> &mut Self {
        self.instructions.push(Instruction::Noop);
        self
    }

    pub fn load_imm(&mut self, dst: Reg, value: Word) -> &mut Self {
        self.instructions.push(Instruction::LoadImm { dst, value });
        self
    }

    pub fn add(&mut self, dst: Reg, lhs: Reg, rhs: Reg) -> &mut Self {
        self.instructions.push(Instruction::Add { dst, lhs, rhs });
        self
    }

    pub fn sub(&mut self, dst: Reg, lhs: Reg, rhs: Reg) -> &mut Self {
        self.instructions.push(Instruction::Sub { dst, lhs, rhs });
        self
    }

    pub fn mul(&mut self, dst: Reg, lhs: Reg, rhs: Reg) -> &mut Self {
        self.instructions.push(Instruction::Mul { dst, lhs, rhs });
        self
    }

    pub fn cmp_le(&mut self, dst: Reg, lhs: Reg, rhs: Reg) -> &mut Self {
        self.instructions.push(Instruction::CmpLE { dst, lhs, rhs });
        self
    }

    pub fn mov(&mut self, dst: Reg, src: Reg) -> &mut Self {
        self.instructions.push(Instruction::Mov { dst, src });
        self
    }

    pub fn jump_if(&mut self, cmp: Reg, target: impl Into<String>) -> &mut Self {
        let target = target.into();
        self.instructions.push(Instruction::JumpIf { cmp, target });
        self
    }

    pub fn jump(&mut self, target: impl Into<String>) -> &mut Self {
        let target = target.into();
        self.instructions.push(Instruction::Jump { target });
        self
    }

    pub fn spawn(
        &mut self,
        module: impl Into<String>,
        function: impl Into<String>,
        args: Vec<Reg>,
        dst: Reg,
    ) -> &mut Self {
        self.instructions.push(Instruction::Spawn {
            module: Box::new(module.into()),
            function: Box::new(function.into()),
            args,
            dst,
        });
        self
    }

    pub fn print(&mut self, reg: Reg) -> &mut Self {
        self.instructions.push(Instruction::Print { reg });
        self
    }

    pub fn send(&mut self, dst_pid: Reg, src_reg: Reg) -> &mut Self {
        self.instructions
            .push(Instruction::Send { dst_pid, src_reg });
        self
    }

    pub fn recv(&mut self, dst_reg: Reg) -> &mut Self {
        self.instructions.push(Instruction::Recv { dst_reg });
        self
    }

    pub fn halt(&mut self) -> &mut Self {
        self.instructions.push(Instruction::Halt);
        self
    }

    pub fn label(&mut self, name: impl Into<String>) -> &mut Self {
        self.instructions
            .push(Instruction::Label { name: name.into() });
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessState {
    Running,
    Halted,
    Waiting,
    Crashed,
}

#[derive(Debug)]
pub struct Module {
    name: String,
    functions: BTreeMap<String, Arc<Function>>,
}

impl Module {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            functions: BTreeMap::new(),
        }
    }

    pub fn add_function(&mut self, function: Function) {
        let name = function.name.clone();
        self.functions.insert(name, Arc::new(function));
    }

    pub fn get_function(&self, name: &str) -> Option<Arc<Function>> {
        self.functions.get(name).cloned()
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    name: String,
    arity: usize,
    code: Vec<u8>,
    returns: bool,
}

impl Function {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arity: 0,
            code: Vec::new(),
            returns: false,
        }
    }

    pub fn returns(mut self) -> Self {
        self.returns = true;
        self
    }

    pub fn arity(mut self, arity: usize) -> Self {
        self.arity = arity;
        self
    }

    pub fn instructions(&mut self) -> InstructionBuilder<'_> {
        InstructionBuilder::new(&mut self.code)
    }
}

#[derive(Debug)]
struct Process {
    pid: Pid,
    regs: [Word; REG_COUNT],
    ip: usize,
    function: Arc<Function>,
    mailbox: VecDeque<Word>,
    state: ProcessState,
}

impl Process {
    pub fn new(pid: Pid, function: Arc<Function>, args: &[Word]) -> Self {
        let mut regs = [0; REG_COUNT];
        for (i, arg) in args.iter().enumerate() {
            regs[i] = *arg;
        }

        Self {
            pid,
            regs,
            ip: 0,
            function,
            mailbox: VecDeque::new(),
            state: ProcessState::Running,
        }
    }

    pub fn reset(&mut self, pid: Pid, function: Arc<Function>, args: &[Word]) {
        self.regs[..].fill(0);
        for (i, arg) in args.iter().enumerate() {
            self.regs[i] = *arg;
        }

        self.pid = pid;
        self.ip = 0;
        self.function = function;
        self.mailbox.clear();
        self.state = ProcessState::Running;
    }

    fn step(
        &mut self,
        process_table: &mut BTreeMap<Pid, Rc<RefCell<Process>>>,
        haulted: &mut Vec<Rc<RefCell<Process>>>,
        run_queue: &mut VecDeque<Rc<RefCell<Process>>>,
        modules: &BTreeMap<String, Module>,
        next_pid: &mut usize,
    ) {
        if self.ip >= self.function.code.len()
            || matches!(self.state, ProcessState::Halted | ProcessState::Crashed)
        {
            self.state = ProcessState::Halted;
            return;
        }

        let opcode = self.opcode();
        let ho = HandlerOptions {
            proc: self,
            process_table,
            haulted,
            run_queue,
            modules,
            next_pid,
        };
        DISPATCH_TABLE[opcode](ho);
    }

    #[inline(always)]
    fn opcode(&mut self) -> usize {
        let opcode = &self.function.code[self.ip];
        self.ip += 1;
        *opcode as usize
    }

    #[inline(always)]
    fn reg(&mut self) -> Result<usize> {
        if self.ip >= self.function.code.len() {
            anyhow::bail!("UnexpectedEOF".to_string());
        }
        let ip = self.ip;
        self.ip += 1;
        Ok(self.function.code[ip] as usize)
    }

    #[inline(always)]
    fn imm_u32(&mut self) -> Result<u32> {
        let bytes = self.get_n_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into()?))
    }

    #[inline(always)]
    fn imm_i64(&mut self) -> Result<i64> {
        let bytes = self.get_n_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into()?))
    }

    #[inline(always)]
    fn get_n_bytes(&mut self, n: usize) -> Result<&[u8]> {
        if self.ip + n > self.function.code.len() {
            anyhow::bail!("UnexpectedEOF".to_string());
        }
        let slice = &self.function.code[self.ip..self.ip + n];
        self.ip += n;
        Ok(slice)
    }

    #[inline(always)]
    fn string(&mut self) -> Result<String> {
        let len = self.imm_u32()? as usize;
        let bytes = self.function.code[self.ip..self.ip + len].to_vec();
        self.ip += len;
        match String::from_utf8(bytes) {
            Ok(string) => Ok(string),
            Err(err) => panic!("{}", err),
        }
    }

    #[inline(always)]
    fn args(&mut self) -> Result<Vec<u8>> {
        let len = self.imm_u32()? as usize;
        let bytes = self.function.code[self.ip..self.ip + len].to_vec();
        self.ip += len;
        Ok(bytes)
    }
}

#[derive(Debug, Default)]
pub struct Machine {
    next_pid: usize,
    modules: BTreeMap<String, Module>,
    processes: BTreeMap<Pid, Rc<RefCell<Process>>>,
    haulted: Vec<Rc<RefCell<Process>>>,
    run_queue: VecDeque<Rc<RefCell<Process>>>,
}

impl Machine {
    pub fn register_module(&mut self, module: Module) -> &mut Self {
        self.modules.insert(module.name.clone(), module);
        self
    }

    pub fn spawn(&mut self, module_name: &str, function_name: &str, args: &[Word]) -> Option<Pid> {
        let module = self.modules.get(module_name)?;
        let function = module.get_function(function_name)?;
        if function.arity != args.len() {
            eprintln!("Incorrect arity for function `{function_name}`");
            return None;
        }

        let pid = self.next_pid;
        self.next_pid += 1;

        let process = Rc::new(RefCell::new(Process::new(pid, function, args)));

        self.processes.insert(pid, Rc::clone(&process));
        self.run_queue.push_back(process);
        Some(pid)
    }

    pub fn step(&mut self) {
        let Some(proc_rc) = self.run_queue.pop_front() else {
            return;
        };

        let state = {
            let mut proc = proc_rc.borrow_mut();

            proc.step(
                &mut self.processes,
                &mut self.haulted,
                &mut self.run_queue,
                &self.modules,
                &mut self.next_pid,
            );

            proc.state
        };

        if state == ProcessState::Running {
            self.run_queue.push_back(proc_rc);
        }
    }

    pub fn send(&mut self, pid: Pid, msg: Word) {
        if let Some(proc_rc) = self.processes.get_mut(&pid) {
            let mut proc = proc_rc.borrow_mut();
            proc.mailbox.push_back(msg);

            if proc.state == ProcessState::Waiting {
                proc.state = ProcessState::Running;
                self.run_queue.push_back(Rc::clone(proc_rc));
            }
        }
    }

    pub fn run(&mut self) {
        while let Some(proc_rc) = self.run_queue.pop_front() {
            let mut proc = proc_rc.borrow_mut();

            loop {
                proc.step(
                    &mut self.processes,
                    &mut self.haulted,
                    &mut self.run_queue,
                    &self.modules,
                    &mut self.next_pid,
                );

                if proc.state != ProcessState::Running {
                    break;
                }
            }

            match proc.state {
                ProcessState::Running => {
                    self.run_queue.push_back(Rc::clone(&proc_rc));
                }
                ProcessState::Halted => {
                    if let Some(proc) = self.processes.remove(&proc.pid) {
                        if self.haulted.len() < 20 {
                            self.haulted.push(proc);
                        }
                    }
                }
                ProcessState::Crashed => println!("[PID {}] Crashed", proc.pid),
                _ => {}
            }
        }
    }
}

#[test]
fn bytecode_noop() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.noop();
    }

    assert_eq!(code.len(), 1);
    assert_eq!(code[0], 0);
}

#[test]
fn bytecode_halt() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.halt();
    }

    assert_eq!(code.len(), 1);
    assert_eq!(code[0], 1);
}

#[test]
fn bytecode_load_imm() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.load_imm(Reg(10), 123);
    }

    assert_eq!(code.len(), 10);
    assert_eq!(code[0], 2);
    assert_eq!(code[1], 10);
    assert_eq!(code[2], 123);
    assert_eq!(code[3], 0);
    assert_eq!(code[4], 0);
    assert_eq!(code[5], 0);
    assert_eq!(code[6], 0);
    assert_eq!(code[7], 0);
    assert_eq!(code[8], 0);
    assert_eq!(code[9], 0);
}

#[test]
fn bytecode_add() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.add(Reg(10), Reg(2), Reg(3));
    }

    assert_eq!(code.len(), 4);
    assert_eq!(code[0], 3); // opcode
    assert_eq!(code[1], 10); // dest
    assert_eq!(code[2], 2); // lhs
    assert_eq!(code[3], 3); // rhs
}

#[test]
fn bytecode_sub() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.sub(Reg(10), Reg(2), Reg(3));
    }

    assert_eq!(code.len(), 4);
    assert_eq!(code[0], 4); // opcode
    assert_eq!(code[1], 10); // dest
    assert_eq!(code[2], 2); // lhs
    assert_eq!(code[3], 3); // rhs
}

#[test]
fn bytecode_cmp_le() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.cmp_le(Reg(10), Reg(2), Reg(3));
    }

    assert_eq!(code.len(), 4);
    assert_eq!(code[0], 5); // opcode
    assert_eq!(code[1], 10); // dest
    assert_eq!(code[2], 2); // lhs
    assert_eq!(code[3], 3); // rhs
}

#[test]
fn bytecode_jump_if() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.jump_if(Reg(10), "main").label("main");
    }

    assert_eq!(code.len(), 6);
    assert_eq!(code[0], 6); // opcode
    assert_eq!(code[1], 10); // cmp
    assert_eq!(code[2], 6); // target
    assert_eq!(code[3], 0);
    assert_eq!(code[4], 0);
    assert_eq!(code[5], 0);
}

#[test]
fn bytecode_spawn() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.spawn("main", "main", vec![], Reg(10));
    }

    assert_eq!(code.len(), 0x16);
    assert_eq!(code[0], 0x7);
    assert_eq!(code[1], 0x4);
    assert_eq!(code[2], 0x0);
    assert_eq!(code[3], 0x0);
    assert_eq!(code[4], 0x0);
    assert_eq!(code[5], 0x6d);
    assert_eq!(code[6], 0x61);
    assert_eq!(code[7], 0x69);
    assert_eq!(code[8], 0x6e);
    assert_eq!(code[9], 0x4);
    assert_eq!(code[10], 0x0);
    assert_eq!(code[11], 0x0);
    assert_eq!(code[12], 0x0);
    assert_eq!(code[13], 0x6d);
    assert_eq!(code[14], 0x61);
    assert_eq!(code[15], 0x69);
    assert_eq!(code[16], 0x6e);
    assert_eq!(code[17], 0x0);
    assert_eq!(code[18], 0x0);
    assert_eq!(code[19], 0x0);
    assert_eq!(code[20], 0x0);
    assert_eq!(code[21], 0xa);
}

#[test]
fn bytecode_print() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.print(Reg(10));
    }

    assert_eq!(code.len(), 2);
    assert_eq!(code[0], 8); // opcode
    assert_eq!(code[1], 10); // reg
}

#[test]
fn bytecode_send() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.send(Reg(200), Reg(10));
    }

    assert_eq!(code.len(), 3);
    assert_eq!(code[0], 9); // opcode
    assert_eq!(code[1], 200); // reg for pid
    assert_eq!(code[2], 10); // data reg
}

#[test]
fn bytecode_recv() {
    let mut code = Vec::new();
    {
        let mut assembler = InstructionBuilder::new(&mut code);
        assembler.recv(Reg(200));
    }

    assert_eq!(code.len(), 2);
    assert_eq!(code[0], 10); // opcode
    assert_eq!(code[1], 200); // reg
}
