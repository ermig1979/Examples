#include "defs.h"

#include <cmath>

typedef void (*Add2Fp32VecsPtr)(const float *a, const float *b, size_t size, float *c);

void Add2Fp32VecsRef(const float* a, const float* b, size_t size, float* c)
{
    for (size_t i = 0; i < size; ++i)
        c[i] = a[i] + b[i];
}

struct Add2Fp32VecsJit : Xbyak::CodeGenerator
{
    void operator=(const Add2Fp32VecsJit&);
public:
    Add2Fp32VecsJit()
    {
    }

    bool Generate()
    {
        resetSize();

        test(rdx, rdx);
        je("LOOP_END");

        mov(r10, rdx);
        and_(r10, ~7);

        mov(r11, rdx);
        and_(r11, ~3);

        xor_(rbx, rbx);

        L("LOOP_BEG_8");
        cmp(rbx, r10);
        jnl("LOOP_END_8");
        vmovups(ymm0, ptr[rdi + rbx * 4]);
        vmovups(ymm1, ptr[rsi + rbx * 4]);
        vaddps(ymm2, ymm0, ymm1);
        vmovups(ptr[rcx + rbx * 4], ymm2);
        add(rbx, 8);
        jmp("LOOP_BEG_8");
        L("LOOP_END_8");

        L("LOOP_BEG_4");
        cmp(rbx, r11);
        jnl("LOOP_END_4");
        vmovups(xmm0, ptr[rdi + rbx * 4]);
        vmovups(xmm1, ptr[rsi + rbx * 4]);
        vaddps(xmm0, xmm1);
        vmovups(ptr[rcx + rbx * 4], xmm0);
        add(rbx, 4);
        jmp("LOOP_BEG_4");
        L("LOOP_END_4");

        L("LOOP_BEG_1");
        cmp(rbx, rdx);
        jnl("LOOP_END_1");
        vmovss(xmm0, ptr[rdi + rbx * 4]);
        vmovss(xmm1, ptr[rsi + rbx * 4]);
        vaddps(xmm0, xmm1);
        vmovss(ptr[rcx + rbx * 4], xmm0);
        add(rbx, 1);
        jmp("LOOP_BEG_1");
        L("LOOP_END_1");

        ret();

        return true;
    }
};

bool TestAdd2Fp32Vecs()
{
    std::cout << " TestAdd2Fp32Vecs:";

    const size_t n = 1023;
    Buf a(n), b(n), cRef(n), cJit(n);
    Init(a);
    Init(b);

    Add2Fp32VecsPtr add2Fp32VecsRef = Add2Fp32VecsRef;

    Add2Fp32VecsJit add2Fp32VecsGen;
    add2Fp32VecsGen.Generate();
    std::cout << " Jit size: " << add2Fp32VecsGen.getSize();

    Add2Fp32VecsPtr add2Fp32VecsJit = add2Fp32VecsGen.getCode<Add2Fp32VecsPtr>();

    add2Fp32VecsRef(a.p, b.p, n, cRef.p);

    add2Fp32VecsJit(a.p, b.p, n, cJit.p);

    if (!Check(cRef, cJit, "", 0.001f))
        return false;

    std::cout << " OK." << std::endl;
    return true;
}