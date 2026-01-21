#include "defs.h"

#include <cmath>

typedef void (*GemmPtr)(int M, int N, int K, const float* A, const float* B, float* C);

int S = 96;
int M = S, N = S, K = S, L = 0;
double TIME = 0.9;

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

typedef void (*MicroPtr)(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc);

struct MicroJit : Xbyak::CodeGenerator
{
    void operator=(const MicroJit&);

public:
    MicroJit()
    {
    }

    static int MaxM()
    {
#ifdef __AVX512BW__
        return 6;
#else
        return 6;
#endif
    }

    static int MaxN()
    {
#ifdef __AVX512BW__
        return 16;
#else
        return 16;
#endif
    }

    bool Generate(int m, int n)
    {
        resetSize();

        const Xbyak::Reg64& K = rdi;
        _A = rsi;
        _lda = rdx;
        const Xbyak::Reg64& step = rcx;
        _B = r8;
        _ldb = r9;
        _C = r10;
        _ldc = r11;

        mov(_C, ptr[rsp + 0x08]);
        imul(_ldb, _ldb, 4);
        imul(_ldc, ptr[rsp + 0x10], 4);

        ResetC(m);

        xor_(rbx, rbx);
        L("LOOP_BEG_K_1");
        cmp(rbx, K);
        jnl("LOOP_END_K_1", T_NEAR);

        LoopBody(m);

        add(rbx, 1);
        jmp("LOOP_BEG_K_1", T_NEAR);
        L("LOOP_END_K_1");

        StoreC(m);

        ret();

        return true;
    }

private:
    Xbyak::Reg64 _A, _lda, _B, _ldb, _C, _ldc;

    void ResetC(int m)
    {
#ifdef __AVX512BW__
        for (int i = 0; i < m; ++i)
        {
            Xbyak::Zmm c0(i * 2 + 0), c1(i * 2 + 1);
            vxorps(c0, c0, c0);
            //vxorps(c1, c1, c1);
        }
#else
        for (int i = 0; i < m; ++i)
        {
            Xbyak::Ymm c0(i * 2 + 0), c1(i * 2 + 1);
            vxorps(c0, c0, c0);
            vxorps(c1, c1, c1);
        }
#endif
    }

    void LoopBody(int m)
    {
#ifdef __AVX512BW__
        Xbyak::Zmm b0(m * 2 + 0), b1(m * 2 + 1), a0(m * 2 + 2), a1(m * 2 + 3);
        vmovups(b0, ptr[_B + 00]);
        //vmovups(b1, ptr[_B + 32]);
        for (int i = 0; i < m; ++i)
        {
            Xbyak::Zmm c0(i * 2 + 0), c1(i * 2 + 1);
            vbroadcastss(a0, ptr[_A + i * 4]);
            vfmadd231ps(c0, b0, a0);
            //vfmadd231ps(c1, b1, a0);
        }
        add(_B, _ldb);
        add(_A, m * 4);
#else
        Xbyak::Ymm b0(m * 2 + 0), b1(m * 2 + 1), a0(m * 2 + 2), a1(m * 2 + 3);
        vmovups(b0, ptr[_B + 00]);
        vmovups(b1, ptr[_B + 32]);
        for (int i = 0; i < m; ++i)
        {
            Xbyak::Ymm c0(i * 2 + 0), c1(i * 2 + 1);
            vbroadcastss(a0, ptr[_A + i * 4]);
            vfmadd231ps(c0, b0, a0);
            vfmadd231ps(c1, b1, a0);
        }
        add(_B, _ldb);
        add(_A, m * 4);
#endif
    }

    void StoreC(int m)
    {
#ifdef __AVX512BW__
        for (int i = 0; i < m; ++i)
        {
            Xbyak::Zmm c0(i * 2 + 0), c1(i * 2 + 1);
            vmovups(ptr[_C + 00], c0);
            //vmovups(ptr[_C + 32], c1);
            add(_C, _ldc);
        }
#else
        for (int i = 0; i < m; ++i)
        {
            Xbyak::Ymm c0(i * 2 + 0), c1(i * 2 + 1);
            vmovups(ptr[_C + 00], c0);
            vmovups(ptr[_C + 32], c1);
            add(_C, _ldc);
        }
#endif
    }
};

void MacroV2(int M, int N, int K, const float* A, const float* B, int ldb, float* bufB, bool reorderB, float* C, int ldc, MicroPtr micro)
{
    int MM = MicroJit::MaxM();
    for (int j = 0; j < N; j += 16)
    {
        if (reorderB)
            ReorderB16(K, B + j, ldb, bufB + K * j);
        for (int i = 0; i < M; i += MicroJit::MaxM())
            micro(K, A + i * K, 1, MM, bufB + K * j, 16, C + i * ldc + j, ldc);
    }
}

int echo = 1;

void GemmV2(int M, int N, int K, const float* A, const float* B, float* C)
{
    const int L1 = 32 * 1024, L2 = 256 * 1024, L3 = 2 * 1024 * 1024, MM = MicroJit::MaxM(), MN = MicroJit::MaxN();
    int mK = std::min(L1 / 4 / 16, K) / 4 * 4;
    int mM = std::min(L2 / 4 / mK, M) / MM * MM;
    int mN = std::min(L3 / 4 / mK, N) / MN * MN;
    Buf bufB(mN * mK);
    Buf bufA(mK * mM);

    MicroPtr micro = Micro6x16;

    MicroJit microJit;
    if (echo)
        std::cout << "Micro " << MM << "x" << MN << std::flush;
    microJit.Generate(MM, MN);
    if (echo)
    {
        std::cout << " size is " << microJit.getSize()  << " B." << std::endl;
        echo = 0;
    }
    micro = microJit.getCode<MicroPtr>();

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
                MacroV2(dM, dN, dK, bufA.p, B + k * N + j, N, bufB.p, i == 0, C + i * N + j, N, micro);
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