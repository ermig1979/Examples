#include "defs.h"

#include <cmath>

typedef void (*GemmPtr)(int M, int N, int K, const float* A, const float* B, float* C);

int S = 96;
int M = S, N = S, K = S, L = 0;
double TIME = 0.1;

bool Test(GemmPtr gemm, const std::string& desc, const Buf& a, const Buf& b, const Buf& control)
{
    Buf current(M * N);

    double t = 0;
    int n = 0;
    while (t < TIME)
    {
        double start = Time();
        gemm(M, N, K, a.p, b.p, current.p);
        t += Time() - start;
        n++;
    }
    double gflops = 2 * double(M * N) * K * n / t / (1024 * 1024 * 1024);

    std::cout << "  " << desc << " : " << std::setprecision(3) << std::fixed << gflops << " GFLOPS; t = " << t / n * 1000.0f << " msec." << std::endl;

    return Check(control, current, desc);
}

//--------------------------------------------------------------------------------------------------

void GemmV0(int M, int N, int K, const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i * N + j] = 0;
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    }
}

//--------------------------------------------------------------------------------------------------

void Micro6x16(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();
    const int offset0 = lda * 0;
    const int offset1 = lda * 1;
    const int offset2 = lda * 2;
    const int offset3 = lda * 3;
    const int offset4 = lda * 4;
    const int offset5 = lda * 5;
    __m256 b0, b1, a0, a1;
    for (int k = 0; k < K; k++)
    {
        b0 = _mm256_loadu_ps(B + 0);
        b1 = _mm256_loadu_ps(B + 8);
        a0 = _mm256_set1_ps(A[offset0]);
        a1 = _mm256_set1_ps(A[offset1]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        a0 = _mm256_set1_ps(A[offset2]);
        a1 = _mm256_set1_ps(A[offset3]);
        c20 = _mm256_fmadd_ps(a0, b0, c20);
        c21 = _mm256_fmadd_ps(a0, b1, c21);
        c30 = _mm256_fmadd_ps(a1, b0, c30);
        c31 = _mm256_fmadd_ps(a1, b1, c31);
        a0 = _mm256_set1_ps(A[offset4]);
        a1 = _mm256_set1_ps(A[offset5]);
        c40 = _mm256_fmadd_ps(a0, b0, c40);
        c41 = _mm256_fmadd_ps(a0, b1, c41);
        c50 = _mm256_fmadd_ps(a1, b0, c50);
        c51 = _mm256_fmadd_ps(a1, b1, c51);
        B += ldb; A += step;
    }
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c21, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c31, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c40, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c41, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c50, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c51, _mm256_loadu_ps(C + 8)));
}

void ReorderB16(int K, const float* B, int ldb, float* bufB)
{
    for (int k = 0; k < K; ++k, B += ldb, bufB += 16)
    {
        _mm256_storeu_ps(bufB + 0, _mm256_loadu_ps(B + 0));
        _mm256_storeu_ps(bufB + 8, _mm256_loadu_ps(B + 8));
    }
}

void ReorderA6(const float* A, int lda, int M, int K, float* bufA)
{
    for (int i = 0; i < M; i += 6)
    {
        for (int k = 0; k < K; k += 4)
        {
            const float* pA = A + k;
            __m128 a0 = _mm_loadu_ps(pA + 0 * lda);
            __m128 a1 = _mm_loadu_ps(pA + 1 * lda);
            __m128 a2 = _mm_loadu_ps(pA + 2 * lda);
            __m128 a3 = _mm_loadu_ps(pA + 3 * lda);
            __m128 a4 = _mm_loadu_ps(pA + 4 * lda);
            __m128 a5 = _mm_loadu_ps(pA + 5 * lda);
            __m128 a00 = _mm_unpacklo_ps(a0, a2);
            __m128 a01 = _mm_unpacklo_ps(a1, a3);
            __m128 a10 = _mm_unpackhi_ps(a0, a2);
            __m128 a11 = _mm_unpackhi_ps(a1, a3);
            __m128 a20 = _mm_unpacklo_ps(a4, a5);
            __m128 a21 = _mm_unpackhi_ps(a4, a5);
            _mm_storeu_ps(bufA + 0, _mm_unpacklo_ps(a00, a01));
            _mm_storel_pi((__m64*)(bufA + 4), a20);
            _mm_storeu_ps(bufA + 6, _mm_unpackhi_ps(a00, a01));
            _mm_storeh_pi((__m64*)(bufA + 10), a20);
            _mm_storeu_ps(bufA + 12, _mm_unpacklo_ps(a10, a11));
            _mm_storel_pi((__m64*)(bufA + 16), a21);
            _mm_storeu_ps(bufA + 18, _mm_unpackhi_ps(a10, a11));
            _mm_storeh_pi((__m64*)(bufA + 22), a21);
            bufA += 24;
        }
        A += 6 * lda;
    }
}

void InitC(int M, int N, float* C, int ldc)
{
    for (int i = 0; i < M; ++i, C += ldc)
        for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(C + j, _mm256_setzero_ps());
}

void MacroV1(int M, int N, int K, const float* A,
    const float* B, int ldb, float* bufB, bool reorderB, float* C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        if (reorderB)
            ReorderB16(K, B + j, ldb, bufB + K * j);
        for (int i = 0; i < M; i += 6)
            Micro6x16(K, A + i * K, 1, 6, bufB + K * j, 16, C + i * ldc + j, ldc);
    }
}

void GemmV1(int M, int N, int K, const float* A, const float* B, float* C)
{
    const int L1 = 32 * 1024, L2 = 256 * 1024, L3 = 2 * 1024 * 1024;
    int mK = std::min(L1 / 4 / 16, K) / 4 * 4;
    int mM = std::min(L2 / 4 / mK, M) / 6 * 6;
    int mN = std::min(L3 / 4 / mK, N) / 16 * 16;
    Buf bufB(mN * mK);
    Buf bufA(mK * mM);
    for (int j = 0; j < N; j += mN)
    {
        int dN = std::min(N, j + mN) - j;
        for (int k = 0; k < K; k += mK)
        {
            int dK = std::min(K, k + mK) - k;
            for (int i = 0; i < M; i += mM)
            {
                int dM = std::min(M, i + mM) - i;
                if (k == 0)
                    InitC(dM, dN, C + i * N + j, N);
                ReorderA6(A + i * K + k, K, dM, dK, bufA.p);
                MacroV1(dM, dN, dK, bufA.p, B + k * N + j, N, bufB.p, i == 0, C + i * N + j, N);
            }
        }
    }
}

//--------------------------------------------------------------------------------------------------

typedef void (*Micro6x16Ptr)(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc);

struct Micro6x16Jit : Xbyak::CodeGenerator
{
    void operator=(const Micro6x16Jit&);

    int _m, _n;
public:
    Micro6x16Jit(int m, int n)
        : _m(m)
        , _n(n)
    {
    }

    bool Generate()
    {
        resetSize();

        const Xbyak::Reg64& K = rdi;
        const Xbyak::Reg64& A = rsi;
        const Xbyak::Reg64& lda = rdx;
        const Xbyak::Reg64& step = rcx;
        const Xbyak::Reg64& B = r8;
        const Xbyak::Reg64& ldb = r9;

        _C = r10;
        _ldc = r11;

        mov(_C, ptr[rbp + 0x10]);
        mov(_ldc, ptr[rbp + 0x18]);

        ResetC();

        xor_(rbx, rbx);
        L("LOOP_BEG_K_1");
        cmp(rbx, K);
        jnl("LOOP_END_K_1");

        //vmovss(xmm0, ptr[rdi + rbx * 4]);
        //vmovss(xmm1, ptr[rsi + rbx * 4]);
        //vaddps(xmm0, xmm1);
        //vmovss(ptr[rcx + rbx * 4], xmm0);
        add(rbx, 1);
        jmp("LOOP_BEG_K_1");
        L("LOOP_END_K_1");

        StoreC();

        ret();

        return true;
    }

private:
    Xbyak::Reg64 _C, _ldc;

    void ResetC()
    {
        for (int i = 0; i < _m; ++i)
        {
            Xbyak::Ymm c0(i * 2 + 0), c1(i * 2 + 1);
            vxorps(c0, c0, c0);
            vxorps(c1, c1, c1);
        }
    }

    void StoreC()
    {
        for (int i = 0; i < _m; ++i)
        {
            Xbyak::Ymm c0(i * 2 + 0), c1(i * 2 + 1);
            //std::cout << "_C " << _C.toString() << std::endl;
            vmovups(ptr[_C + 00], c0);
            //vmovups(ptr[_C + 32], c1);
            //add(_C, _ldc);
        }
    }
};

void MacroV2(int M, int N, int K, const float* A, const float* B, int ldb, float* bufB, bool reorderB, float* C, int ldc, Micro6x16Ptr micro6x16)
{
    for (int j = 0; j < N; j += 16)
    {
        if (reorderB)
            ReorderB16(K, B + j, ldb, bufB + K * j);
        for (int i = 0; i < M; i += 6)
            micro6x16(K, A + i * K, 1, 6, bufB + K * j, 16, C + i * ldc + j, ldc);
    }
}

int echo = 1;

void GemmV2(int M, int N, int K, const float* A, const float* B, float* C)
{
    const int L1 = 32 * 1024, L2 = 256 * 1024, L3 = 2 * 1024 * 1024;
    int mK = std::min(L1 / 4 / 16, K) / 4 * 4;
    int mM = std::min(L2 / 4 / mK, M) / 6 * 6;
    int mN = std::min(L3 / 4 / mK, N) / 16 * 16;
    Buf bufB(mN * mK);
    Buf bufA(mK * mM);

    Micro6x16Ptr micro6x16 = Micro6x16;

    Micro6x16Jit micro6x16Jit(6, 16);
    micro6x16Jit.Generate();
    if (echo)
    {
        std::cout << "Micro6x16 size is " << micro6x16Jit.getSize() << std::endl;
        echo = 0;
    }
    micro6x16 = micro6x16Jit.getCode<Micro6x16Ptr>();

    for (int j = 0; j < N; j += mN)
    {
        int dN = std::min(N, j + mN) - j;
        for (int k = 0; k < K; k += mK)
        {
            int dK = std::min(K, k + mK) - k;
            for (int i = 0; i < M; i += mM)
            {
                int dM = std::min(M, i + mM) - i;
                if (k == 0)
                    InitC(dM, dN, C + i * N + j, N);
                ReorderA6(A + i * K + k, K, dM, dK, bufA.p);
                MacroV2(dM, dN, dK, bufA.p, B + k * N + j, N, bufB.p, i == 0, C + i * N + j, N, micro6x16);
            }
        }
    }
}

//--------------------------------------------------------------------------------------------------

bool TestSgemm()
{
    std::cout << " TestSgemm:" << std::endl;

    Buf a(M * K), b(K * N), c(M * N);
    Init(a);
    Init(b);
    GemmV0(M, N, K, a.p, b.p, c.p);

    if (L <= 0 && !Test(GemmV0, "GemmV0", a, b, c)) return false;
    if (L <= 1 && !Test(GemmV1, "GemmV1", a, b, c)) return false;
    if (L <= 2 && !Test(GemmV2, "GemmV2", a, b, c)) return false;

    std::cout << " OK." << std::endl;
    return true;
}