pub mod decoder;
use anyhow::Result;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

pub type Word = i64;
pub type Pid = usize;

const REG_COUNT: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct Reg(pub usize);

#[derive(Clone, Copy, Debug)]
pub struct Addr(pub usize);

type OpHandler = fn(
    &mut Process,
    &mut HashMap<Pid, Rc<RefCell<Process>>>,
    &HashMap<String, Module>,
    &mut usize,
) -> Option<Process>;

static DISPATCH_TABLE: &[OpHandler] = &[
    handle_noop,
    handle_halt,
    handle_load_imm,
    handle_add,
    handle_sub,
    handle_cmp_le,
    handle_jump_if,
    handle_spawn,
    handle_print,
    handle_send,
    handle_recv,
];

#[inline(always)]
fn handle_noop(
    _proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    None
}

#[inline(always)]
fn handle_halt(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    proc.state = ProcessState::Halted;
    None
}

#[inline(always)]
fn handle_load_imm(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    let Ok(dst) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(value) = proc.imm_i64() else {
        proc.state = ProcessState::Crashed;
        return None;
    };

    proc.regs[dst as usize] = value;
    None
}

#[inline(always)]
fn handle_add(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    let Ok(dst) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(lhs) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(rhs) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };

    proc.regs[dst as usize] = proc.regs[lhs as usize] + proc.regs[rhs as usize];
    None
}

#[inline(always)]
fn handle_sub(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    let Ok(dst) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(lhs) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(rhs) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };

    proc.regs[dst as usize] = proc.regs[lhs as usize] - proc.regs[rhs as usize];
    None
}

#[inline(always)]
fn handle_cmp_le(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    let Ok(dst) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(lhs) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(rhs) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };

    proc.regs[dst as usize] = (proc.regs[lhs as usize] <= proc.regs[rhs as usize]) as i64;
    None
}

#[inline(always)]
fn handle_jump_if(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    let Ok(cmp) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(target) = proc.imm_u32() else {
        proc.state = ProcessState::Crashed;
        return None;
    };

    if proc.regs[cmp as usize] != 0 {
        proc.ip = target as usize;
    }
    None
}

#[inline(always)]
fn handle_spawn(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    modules: &HashMap<String, Module>,
    next_pid: &mut usize,
) -> Option<Process> {
    let Ok(module_name) = proc.string() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(function_name) = proc.string() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(args) = proc.args() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(dst) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Some(module) = modules.get(module_name.as_str()) else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Some(function) = module.functions.get(function_name.as_str()) else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let new_pid = *next_pid;
    *next_pid += 1;
    let mut args: Vec<Word> = args.iter().map(|&arg| proc.regs[arg as usize]).collect();
    if function.returns {
        args.insert(0, proc.pid as Word);
    }
    let new_proc = Process::new(new_pid, Arc::clone(function), &args);
    proc.regs[dst as usize] = new_pid as i64;

    Some(new_proc)
}

#[inline(always)]
fn handle_print(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    let Ok(arg) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    println!("{}", proc.regs[arg as usize]);
    None
}

#[inline(always)]
fn handle_recv(
    proc: &mut Process,
    _process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    let start = proc.ip;
    let Ok(dst) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };

    if let Some(msg) = proc.mailbox.pop_front() {
        proc.regs[dst as usize] = msg;
        proc.state = ProcessState::Running;
    } else {
        proc.ip = start - 1;
        proc.state = ProcessState::Waiting;
        return None;
    }
    None
}

#[inline(always)]
fn handle_send(
    proc: &mut Process,
    process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
    _modules: &HashMap<String, Module>,
    _next_pid: &mut usize,
) -> Option<Process> {
    let Ok(dst_pid) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let Ok(src_reg) = proc.reg() else {
        proc.state = ProcessState::Crashed;
        return None;
    };
    let pid = proc.regs[dst_pid as usize] as Pid;
    if let Some(dst_proc) = process_table.get_mut(&pid) {
        let mut p = dst_proc.borrow_mut();
        p.mailbox.push_back(proc.regs[src_reg as usize]);
    }
    None
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
    CmpLE {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    JumpIf {
        cmp: Reg,
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
            Instruction::CmpLE { .. } => 5,
            Instruction::JumpIf { .. } => 6,
            Instruction::Spawn { .. } => 7,
            Instruction::Print { .. } => 8,
            Instruction::Send { .. } => 9,
            Instruction::Recv { .. } => 10,
            Instruction::Label { .. } => panic!("Label has no opcode"),
        }
    }

    pub fn into_bytes(self, index: usize, backpatch: &mut Vec<(usize, Instruction)>) -> Vec<u8> {
        let opcode = self.opcode();
        let mut bytes = vec![opcode];
        match self {
            Instruction::Noop | Instruction::Halt => {}
            Instruction::LoadImm { dst, value } => {
                bytes.push(dst.0 as u8);
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Instruction::Add { dst, lhs, rhs }
            | Instruction::Sub { dst, lhs, rhs }
            | Instruction::CmpLE { dst, lhs, rhs } => {
                bytes.push(dst.0 as u8);
                bytes.push(lhs.0 as u8);
                bytes.push(rhs.0 as u8);
            }
            Instruction::JumpIf { cmp, target } => {
                bytes.push(cmp.0 as u8);
                bytes.extend_from_slice(&u32::MAX.to_le_bytes());
                backpatch.push((index, Instruction::JumpIf { cmp, target }));
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
                    bytes.push(arg.0 as u8);
                }
                bytes.push(dst.0 as u8);
            }
            Instruction::Print { reg } => {
                bytes.push(reg.0 as u8);
            }
            Instruction::Send { dst_pid, src_reg } => {
                bytes.push(dst_pid.0 as u8);
                bytes.push(src_reg.0 as u8);
            }
            Instruction::Recv { dst_reg } => {
                bytes.push(dst_reg.0 as u8);
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
        let mut labels = std::collections::HashMap::new();
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
            let Instruction::JumpIf {
                cmp: _,
                target: name,
            } = instruction
            else {
                panic!("Instruction is not JumpIf");
            };
            let Some(target) = labels.get(&name) else {
                panic!("Label {name} not found");
            };
            let bytes = (*target as u32).to_le_bytes();
            for (i, byte) in bytes.iter().enumerate() {
                self.bytecode[instruction_index + i + 2] = *byte;
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

    pub fn cmp_le(&mut self, dst: Reg, lhs: Reg, rhs: Reg) -> &mut Self {
        self.instructions.push(Instruction::CmpLE { dst, lhs, rhs });
        self
    }

    pub fn jump_if(&mut self, cmp: Reg, target: impl Into<String>) -> &mut Self {
        let target = target.into();
        self.instructions.push(Instruction::JumpIf { cmp, target });
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
    functions: HashMap<String, Arc<Function>>,
}

impl Module {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            functions: HashMap::new(),
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

    fn step(
        &mut self,
        process_table: &mut HashMap<Pid, Rc<RefCell<Process>>>,
        modules: &HashMap<String, Module>,
        next_pid: &mut usize,
    ) -> Option<Process> {
        if self.ip >= self.function.code.len()
            || matches!(self.state, ProcessState::Halted | ProcessState::Crashed)
        {
            self.state = ProcessState::Halted;
            return None;
        }

        let opcode = self.opcode();
        DISPATCH_TABLE[opcode](self, process_table, modules, next_pid)
    }

    #[inline(always)]
    fn opcode(&mut self) -> usize {
        let opcode = &self.function.code[self.ip];
        self.ip += 1;
        *opcode as usize
    }

    #[inline(always)]
    fn reg(&mut self) -> Result<u8> {
        if self.ip >= self.function.code.len() {
            anyhow::bail!("UnexpectedEOF".to_string());
        }
        let ip = self.ip;
        self.ip += 1;
        Ok(self.function.code[ip])
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
    modules: HashMap<String, Module>,
    processes: HashMap<Pid, Rc<RefCell<Process>>>,
    run_queue: VecDeque<Pid>,
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

        let process = Process::new(pid, function, args);
        self.processes.insert(pid, Rc::new(RefCell::new(process)));
        self.run_queue.push_back(pid);
        Some(pid)
    }

    pub fn step(&mut self) {
        if let Some(pid) = self.run_queue.pop_front() {
            let Some(proc_rc) = self.processes.get(&pid).cloned() else {
                return;
            };

            let still_running = {
                let mut proc = proc_rc.borrow_mut();

                if let Some(new_proc) =
                    proc.step(&mut self.processes, &self.modules, &mut self.next_pid)
                {
                    let new_pid = new_proc.pid;
                    self.processes
                        .insert(new_pid, Rc::new(RefCell::new(new_proc)));
                    self.run_queue.push_back(new_pid);
                }

                matches!(proc.state, ProcessState::Running)
            };

            if still_running {
                self.run_queue.push_back(pid);
            }
        }
    }

    pub fn send(&mut self, pid: Pid, msg: Word) {
        if let Some(proc) = self.processes.get_mut(&pid) {
            let mut proc = proc.borrow_mut();
            proc.mailbox.push_back(msg);

            if matches!(proc.state, ProcessState::Waiting) {
                proc.state = ProcessState::Running;
                self.run_queue.push_back(pid);
            }
        }
    }

    pub fn run(&mut self) {
        while let Some(pid) = self.run_queue.pop_front() {
            let proc_rc = match self.processes.get(&pid) {
                Some(p) => Rc::clone(p),
                None => continue,
            };

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut proc = proc_rc.borrow_mut();

                loop {
                    if let Some(new_proc) =
                        proc.step(&mut self.processes, &self.modules, &mut self.next_pid)
                    {
                        let new_pid = new_proc.pid;
                        self.processes
                            .insert(new_pid, Rc::new(RefCell::new(new_proc)));
                        self.run_queue.push_back(new_pid);
                    }

                    if matches!(
                        proc.state,
                        ProcessState::Halted | ProcessState::Crashed | ProcessState::Waiting
                    ) {
                        break;
                    }
                }
            }));

            let mut proc = proc_rc.borrow_mut();

            match result {
                Ok(_) => match proc.state {
                    ProcessState::Running | ProcessState::Waiting => {
                        self.run_queue.push_back(pid);
                    }
                    ProcessState::Halted => {}
                    ProcessState::Crashed => println!("[PID {}] Crashed", pid),
                },
                Err(_) => {
                    println!("[PID {}] Panicked!", pid);
                    proc.state = ProcessState::Crashed;
                }
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
