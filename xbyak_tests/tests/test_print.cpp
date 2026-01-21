#include "defs.h"

typedef void (*PrintPtr)();

struct PrintJit : Xbyak::CodeGenerator 
{
    void operator=(const PrintJit&);
public:
    PrintJit()
    {
        mov(rax, 0);
        add(rax, 1);

        Print(" rax: %ld ", rax);

        Print(" jit_message ");

        Print(" rax: %ld ", rax);

        ret();
    }

private:
    void Print(const char* message)
    {
        push(rax);
        push(rdi);
        push(rsi);
        push(rbp);
        mov(rax, (size_t)printf);
        mov(rdi, (size_t)message);
        mov(rsi, 0);
        call(rax);
        pop(rbp);
        pop(rdi);
        pop(rsi);
        pop(rax);
    }

    void Print(const char* format, const Xbyak::Reg64 &reg)
    {
        push(reg);        
        push(rax);
        push(rdi);
        push(rsi);
        push(rdx);
        push(rbp);
        mov(rsi, reg);
        mov(rdi, (size_t)format);
        mov(rdx, 0);
        mov(rax, (size_t)printf);
        call(rax);
        pop(rbp);
        pop(rdx);
        pop(rsi);
        pop(rdi);
        pop(rax);
        pop(reg);
    }
};

bool TestPrint()
{
    std::cout << " TestPrint: ";

    PrintJit printJit;

    printJit.getCode<PrintPtr>()();

    std::cout << " OK." << std::endl;

    return true;
}