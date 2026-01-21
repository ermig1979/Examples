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

        Print(" jit_message ");

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
};

bool TestPrint()
{
    std::cout << " TestPrint: ";

    PrintJit printJit;

    printJit.getCode<PrintPtr>()();

    std::cout << " OK." << std::endl;

    return true;
}