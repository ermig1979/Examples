#include "defs.h"

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

const int S = 128;
int M = S, N = S, K = S;
float HALF = 100.0f;

bool test(int M, int N, int K, const std::string& desc, float aMin, float aMax, float bMin, float bMax, int order)
{
    std::cout << " Test: " << desc << " a[" << aMin << " .. " << aMax << "], b[" << bMin << " .. " << bMax << "], o=" << order << " :" << std::endl;

    mat_t a(M, K), b(K, N), c0(M, N), c1(M, N);
    init(a, aMin, aMax, order);
    init(b, bMin, bMax, order);

    gemm_v0(a, b, c0);
    gemm_v1(a, b, c1);

    diff_t d1;

    diff(c0, c1, d1);

    std::cout << " Diff: min = " << d1.d.min << ", max = " << d1.d.max << std::endl;
    std::cout << std::endl;

    return true;
}

int main(int argc, char* argv[])
{
    if (argc > 1) M = N = K = atoi(argv[1]);
    if (argc > 2) N = K = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    for (int ia = -1; ia <= 1; ia++)
    {
        for (int ib = -1; ib <= 1; ib++)
        {
            test(M, N, K, "stub", (ia - 1) * HALF, (ia + 1) * HALF, (ib - 1) * HALF, (ib + 1) * HALF, 1);
        }
    }

    return 0;
}