#include "defs.h"
#include "bf16.h"

void test(int M, int N, int K, const std::string& desc, gemm_t gemm, float aMin, float aMax, int aOrder, float bMin, float bMax, int bOrder, int orig)
{
    std::cout << "Test a[" << aMin << " .. " << aMax << " ^ " << aOrder << "], b[" << bMin << " .. " << bMax << " ^ " << bOrder << "] : " << std::endl;

    mat_t a(M, K), b(K, N), c0(M, N), c1(M, N);
    srand(0);
    init(a, aMin, aMax, aOrder);
    init(b, bMin, bMax, bOrder);

    gemm_control(a, b, c0);
    gemm(a, b, c1);

    diff_t d;

    diff(c0, c1, d);

    if(orig)
        std::cout << " Orig: " << d.a.info(3) << std::endl;
    std::cout << " Diff: " << d.d.info(6) << std::endl;
    std::cout << std::endl;
}

void test(int M, int N, int K, const std::string& desc, gemm_t gemm, int orig)
{
    const float HALF = 10.0f;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "TEST " << desc << " :" << std::endl;
    test(M, N, K, desc, gemm, 0.0f, 10.0f, 1, -0.3f, 0.3f, 6, orig);
    test(M, N, K, desc, gemm, -1.0f, 9.0f, 1, -0.3f, 0.3f, 6, orig);
    test(M, N, K, desc, gemm, 0.0f, 10.0f, 1, -0.2f, 0.4f, 6, orig);

    std::cout << std::endl;
}

#define TEST(M, N, K, gemm,orig) test(M, N, K, #gemm, gemm, orig)

int main(int argc, char* argv[])
{
    const int S = 100;
    int M = S, N = S, K = S;
    if (argc > 1) M = N = K = atoi(argv[1]);
    if (argc > 2) N = K = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    TEST(M, N, K, ::gemm<bf16_original_t>, 1);

    TEST(M, N, K, ::gemm<bf16_truncate_t>, 0);

    TEST(M, N, K, ::gemm<bf16_nearest_t>, 0);

    TEST(M, N, K, ::gemm<bf16_nearest_even_a_t>, 0);

    TEST(M, N, K, ::gemm<bf16_nearest_even_b_t>, 0);

    TEST(M, N, K, ::gemm<bf16_magic_number_a_t>, 0);

    TEST(M, N, K, ::gemm<bf16_magic_number_b_t>, 0);

    return 0;
}