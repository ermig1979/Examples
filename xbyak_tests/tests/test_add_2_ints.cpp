#include "defs.h"

typedef int (*Add2IntsPtr)(int a, int b);

int Add2IntRef(int a, int b)
{
    return a + b;
}

struct Add2IntsJit : Xbyak::CodeGenerator 
{
    void operator=(const Add2IntsJit&);
public:
    Add2IntsJit()
    {
        mov(rax, rdi);
        add(rax, rsi);
        ret();
    }
};

bool TestAdd2Ints()
{
    std::cout << " TestAdd2Ints:";

    int a = 10, b = 20;

    Add2IntsPtr add2IntsRef = Add2IntRef;

    Add2IntsJit add2IntsGen;

    Add2IntsPtr add2IntsJit = add2IntsGen.getCode<Add2IntsPtr>();

    int cRef = add2IntsRef(a, b);

    int cJit = add2IntsJit(a, b);

    if (cRef == cJit)
    {
        std::cout << " OK." << std::endl;
        return true;
    }
    else
    {
        std::cout << " Error! " << cRef << " != " << cJit << std::endl;
        return false;
    }
}