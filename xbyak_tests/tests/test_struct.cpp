#include "defs.h"

struct S
{
    int32_t a;
    int64_t b;
    int16_t c;
    int8_t d;
};

typedef void (*ModifyPtr)(S * s);

void ModifyRef(S* s)
{
    s->a = 10;
    s->b = s->c + s->d;
}

struct ModifyJit : Xbyak::CodeGenerator
{
    void operator=(const ModifyJit&);
public:
    ModifyJit()
    {
        mov(eax, 10);
        mov(ptr[rdi + 0], eax);
        mov(ax, ptr[rdi + (offsetof(S, c))]);
        mov(cl, ptr[rdi + offsetof(S, d)]);
        add(rax, rcx);
        mov(ptr[rdi + offsetof(S, b)], rax);
        ret();
    }
};

bool TestStruct()
{
    std::cout << " TestStruct:" << std::flush;

    S sRef = { 0, 1, 2, 3 }, sJit = sRef;

    ModifyRef(&sRef);

    ModifyJit modifyGen;
    ModifyPtr modifyJit = modifyGen.getCode<ModifyPtr>();

    modifyJit(&sJit);

    if (memcmp(&sRef, &sJit, sizeof(S)) == 0)
    {
        std::cout << " OK." << std::endl;
        return true;
    }
    else
    {
        std::cout << " Error! a = " << sJit.a << " b = " << sJit.b << " c = " << (int)sJit.c << " d = " << (int)sJit.d << std::endl;
        return false;
    }
}