#include "defs.h"
#include "bf16.h"

void gemm_v0(const mat_t & a, const mat_t& b, mat_t& c)
{
    assert(a.m == c.m && a.n == b.m && b.n == c.n);
    for (int i = 0; i < a.m; ++i)
    {
        for (int j = 0; j < b.n; ++j)
        {
            double sum = 0;
            for (int k = 0; k < a.n; ++k)
                sum += double(a.p[i * a.n + k]) * double(b.p[k * b.n + j]);
            c.p[i * b.n + j] = float(sum);
        }
    }
}

void gemm_v1(const mat_t& a, const mat_t& b, mat_t& c)
{
    assert(a.m == c.m && a.n == b.m && b.n == c.n);
    for (int i = 0; i < a.m; ++i)
    {
        for (int j = 0; j < b.n; ++j)
        {
            float sum = 0;
            for (int k = 0; k < a.n; ++k)
                sum += a.p[i * a.n + k] * b.p[k * b.n + j];
            c.p[i * b.n + j] = sum;
        }
    }
}

void gemm_v2(const mat_t& a, const mat_t& b, mat_t& c)
{
    assert(a.m == c.m && a.n == b.m && b.n == c.n);
    for (int i = 0; i < a.m; ++i)
    {
        for (int j = 0; j < b.n; ++j)
        {
            float sum = 0;
            for (int k = 0; k < a.n; ++k)
            {
                float a16 = bf16_truncate(a.p[i * a.n + k]);
                float b16 = bf16_truncate(b.p[k * b.n + j]);
                sum += a16 * b16;
            }
            c.p[i * b.n + j] = sum;
        }
    }
}

void gemm_v3(const mat_t& a, const mat_t& b, mat_t& c)
{
    assert(a.m == c.m && a.n == b.m && b.n == c.n);
    for (int i = 0; i < a.m; ++i)
    {
        for (int j = 0; j < b.n; ++j)
        {
            float sum = 0;
            for (int k = 0; k < a.n; ++k)
            {
                float a16 = bf16_nearest(a.p[i * a.n + k]);
                float b16 = bf16_nearest(b.p[k * b.n + j]);
                sum += a16 * b16;
            }
            c.p[i * b.n + j] = sum;
        }
    }
}

void gemm_v4(const mat_t& a, const mat_t& b, mat_t& c)
{
    assert(a.m == c.m && a.n == b.m && b.n == c.n);
    for (int i = 0; i < a.m; ++i)
    {
        for (int j = 0; j < b.n; ++j)
        {
            float sum = 0;
            for (int k = 0; k < a.n; ++k)
            {
                float a16 = bf16_nearest_even(a.p[i * a.n + k]);
                float b16 = bf16_nearest_even(b.p[k * b.n + j]);
                sum += a16 * b16;
            }
            c.p[i * b.n + j] = sum;
        }
    }
}

void gemm_v5(const mat_t& a, const mat_t& b, mat_t& c)
{
    assert(a.m == c.m && a.n == b.m && b.n == c.n);
    for (int i = 0; i < a.m; ++i)
    {
        for (int j = 0; j < b.n; ++j)
        {
            float sum = 0;
            for (int k = 0; k < a.n; ++k)
            {
                float a16 = bf16_magic_number(a.p[i * a.n + k]);
                float b16 = bf16_magic_number(b.p[k * b.n + j]);
                sum += a16 * b16;
            }
            c.p[i * b.n + j] = sum;
        }
    }
}

const int S = 128;
int M = S, N = S, K = S;
float HALF = 10.0f;

void test(int M, int N, int K, const std::string& desc, gemm_t gemm, float aMin, float aMax, float bMin, float bMax, int order)
{
    std::cout << "Test a[" << aMin << " .. " << aMax << "], b[" << bMin << " .. " << bMax << "], o: " << order << " :" << std::endl;

    mat_t a(M, K), b(K, N), c0(M, N), c1(M, N);
    srand(0);
    init(a, aMin, aMax, order);
    init(b, bMin, bMax, order);

    gemm_v0(a, b, c0);
    gemm(a, b, c1);

    diff_t d;

    diff(c0, c1, d);

    std::cout << " Orig: " << d.a.info(3) << std::endl;
    //std::cout << " Curr: " << d.b.info(3) << std::endl;
    std::cout << " Diff: " << d.d.info(6) << std::endl;
    std::cout << std::endl;
}

void test(int M, int N, int K, const std::string& desc, gemm_t gemm)
{
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "TEST " << desc << " :" << std::endl;
    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
            for (int o = 1; o <= 1; o++)
                test(M, N, K, desc, gemm, (a - 1) * HALF, (a + 1) * HALF, (b - 1) * HALF, (b + 1) * HALF, o);
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    float a = 1.00390625f;

    if (argc > 1) M = N = K = atoi(argv[1]);
    if (argc > 2) N = K = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    test(M, N, K, "v1 (float origin)", gemm_v1);

    test(M, N, K, "v2 (bf16 truncate)", gemm_v2);

    test(M, N, K, "v3 (bf16 nearest)", gemm_v3);

    test(M, N, K, "v4 (bf16 nearest even)", gemm_v4);

    test(M, N, K, "v5 (bf16 magic number)", gemm_v5);

    return 0;
}