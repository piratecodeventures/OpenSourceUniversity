This section takes a deep dive into **how a function call actually works**—from Python source code, through the interpreter, down to CPU instructions and memory. We explore it from three complementary perspectives:

- **Computer Organization**: How the CPU manages the stack, registers, and memory during a function call.
    
- **Compiler & Interpreter Internals**: What CPython (the standard Python implementation) does when you define and invoke a function.
    
- **Kernel / OS Perspective**: How operating system mechanisms (processes, virtual memory, system calls) underpin the runtime.
    

We conclude with methods to _trace and verify_ these steps, including disassembly of Python bytecode and native machine instructions.

---

## 1. High-Level Flow

When you write and call a Python function:

```python
def add(a, b):
    return a + b

result = add(2, 3)
```

The journey of `add(2, 3)` is roughly:

1. **Parsing & Compilation**: Source code → Python bytecode (`.pyc`) by CPython’s compiler.
    
2. **Interpreter Dispatch**: CPython Virtual Machine (VM) reads bytecode instructions.
    
3. **Frame & Stack Setup**: The VM creates a _call frame_ with local variables and pushes it on the call stack.
    
4. **Execution**: The VM executes bytecode ops (`LOAD_FAST`, `BINARY_ADD`, etc.) until `RETURN_VALUE`.
    
5. **Teardown**: Frame pops, return value passed back to caller.
    

Each of these maps to lower-level CPU and OS mechanisms.

---

## 2. CPython Internals

### Compilation to Bytecode

Running `python -m dis myscript.py` shows the bytecode:

```python
>>> import dis
>>> def add(a,b): return a+b
>>> dis.dis(add)
  2           0 LOAD_FAST                0 (a)
              2 LOAD_FAST                1 (b)
              4 BINARY_ADD
              6 RETURN_VALUE
```

The CPython compiler builds a code object (`PyCodeObject`) with instructions and metadata.

### Function Call Mechanics

- A Python function is a `PyFunctionObject` wrapping a `PyCodeObject`.
    
- Calling it triggers `PyEval_EvalFrameEx` (in CPython’s `ceval.c`).
    
- A `PyFrameObject` is created:
    
    - Holds local/global namespaces, stack, instruction pointer.
        
    - Linked list forming the call stack.
        
- The interpreter loop fetches and dispatches bytecode instructions (the “ceval loop”).
    

### Return Path

`RETURN_VALUE` pops the top-of-stack value and destroys the frame.  
Reference counts are updated; garbage collection may free objects.

---

## 3. Computer Organization View

At the hardware level, the Python process is just native machine code executing in user space:

1. **Process & Memory Layout**: Code, heap, stack segments in virtual memory.
    
2. **C Function Calls**: CPython itself is written in C; each Python frame often maps to a C function call.
    
    - `PyEval_EvalFrameEx` uses the C stack for its own frames.
        
3. **CPU Stack Frames**:
    
    - Call instructions (`CALL` on x86_64) push the return address.
        
    - Callee saves registers, sets up a stack frame (`push rbp; mov rbp, rsp`).
        
    - Local variables stored relative to base pointer (RBP).
        
    - `ret` pops the return address to continue execution.
        
4. **Registers**:
    
    - General purpose registers hold arguments (per ABI, e.g., first args in RDI/RSI for x86_64 System V).
        
    - Instruction Pointer (RIP) tracks current instruction.
        

Python adds extra layers, but at bottom, it’s CPU instructions manipulating stack and registers.

---

## 4. Kernel / OS Perspective

- **Process Scheduling**: The OS kernel schedules the Python process.
    
- **Virtual Memory**: Each Python frame, object allocation, and C stack lives in user-space virtual memory managed by the kernel.
    
- **System Calls**: Function calls that involve I/O (file reads, networking) trap into the kernel (e.g., `read()`, `write()`). Pure computation stays in user space.
    
- **Signals & Exceptions**: If a segfault occurs, the kernel delivers a signal; Python converts some signals to exceptions.
    

---

## 5. Tracing and Verifying

### Disassembling Python Bytecode

Use the `dis` module (as above) to see VM instructions.

### Native Machine Code

To observe real CPU instructions:

```bash
gcc -S example.c   # from C to assembly (for comparison)
objdump -d $(which python) | less   # inspect CPython binary
```

Attach a debugger:

```bash
gdb --args python script.py
(gdb) break PyEval_EvalFrameEx
(gdb) run
```

Step through C code and examine the stack with `info frame`, `disassemble`.

### sys.settrace & Profilers

Set a trace function to log every Python call/return:

```python
import sys

def tracer(frame, event, arg):
    print(event, frame.f_code.co_name)
    return tracer

sys.settrace(tracer)
```

### perf / strace

- `strace python script.py` shows system calls.
    
- `perf record -g` and `perf report` profile CPU-level call graphs.
    

---

## 6. Example: From Python to Assembly

Let’s compile and inspect a simple C extension to observe raw assembly of a function call.

```c
// add.c
int add(int a, int b) {
    return a + b;
}
```

Compile and dump assembly:

```bash
gcc -O2 -S add.c -o add.s
cat add.s
```

Sample x86-64 output:

```asm
add:
    leal    (%rdi,%rsi), %eax  # eax = rdi + rsi
    ret
```

This is analogous to what the CPU ultimately executes when CPython calls into a C-implemented builtin.

---

## 7. Performance Considerations

- Python function calls are heavier than C due to frame creation, reference counting, dynamic lookup.
    
- Inline functions or using builtins can avoid Python-level overhead.
    
- Tools like `cython` or `numba` compile Python to native code for speed.
    

---

## 8. Best Practices for Low-Level Tracing

- Build CPython in debug mode for more symbols.
    
- Use `gdb` or `lldb` with Python’s debugging macros (`py-bt` etc.).
    
- Monitor reference counts with `sys.getrefcount`.
    
- Use deterministic test cases to avoid JIT noise (note: standard CPython has no JIT, but PyPy does).
    

---

## Exercises

1. Write a Python function and disassemble its bytecode.
    
2. Trace its execution with `sys.settrace` and confirm call/return events.
    
3. Attach `gdb` to a running Python process and break at `PyEval_EvalFrameEx`.
    
4. Compare a Python function and an equivalent C function compiled to assembly.
    
5. Explore how arguments are passed in registers vs stack on your CPU architecture.
    

---

## References

- CPython Internals: [https://devguide.python.org/internals/](https://devguide.python.org/internals/)
    
- Python dis module: [https://docs.python.org/3/library/dis.html](https://docs.python.org/3/library/dis.html)
    
- "Inside the Python Virtual Machine" by Nilton Volpato
    
- Computer Organization textbooks (Patterson & Hennessy)
    
- gdb Python extensions: [https://wiki.python.org/moin/DebuggingWithGdb](https://wiki.python.org/moin/DebuggingWithGdb)
    
- Linux perf: [https://perf.wiki.kernel.org](https://perf.wiki.kernel.org/)

[[Under the Hood of Function Calls - Hardware-Level Deep Dive]]
