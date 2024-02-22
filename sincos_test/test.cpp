#include "defs.h"
#include <immintrin.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

static inline float Rcp(float x)
{
    return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
}

struct AlgV0
{
    static inline float Sin(float x)
    {
        return ::sinf(x);
    }

    static inline float Cos(float x)
    {
        return ::cosf(x);
    }
};

struct AlgV1
{
    static float SinPi(float x)
    {
        float coeffs[] = {
        -0.10132118f,          // x
            0.0066208798f,        // x^3
            -0.00017350505f,       // x^5
            0.0000025222919f,     // x^7
            -0.000000023317787f,   // x^9
            0.00000000013291342f // x^11
        };
        float pi_major = 3.1415927f;
        float pi_minor = -0.00000008742278f;
        float x2 = x * x;
        float p11 = coeffs[5];
        float p9 = p11 * x2 + coeffs[4];
        float p7 = p9 * x2 + coeffs[3];
        float p5 = p7 * x2 + coeffs[2];
        float p3 = p5 * x2 + coeffs[1];
        float p1 = p3 * x2 + coeffs[0];
        return (x - pi_major - pi_minor) *
            (x + pi_major + pi_minor) * p1 * x;
    }

    static inline float Sin(float x)
    {
        int n = int(floor(x * M_1_PI));
        float f = x - float(n * M_PI);
        float v = SinPi(f);
        return v * (n & 1 ? -1.0f : 1.0f);
    }

    static inline float Cos(float x)
    {
        x = M_PI_2 - x;
        int n = int(floor(x * M_1_PI));
        float f = x - float(n * M_PI);
        float v = SinPi(f);
        return v * (n & 1 ? -1.0f : 1.0f);
    }
};

struct AlgV2
{
    static float SinPi(float x)
    {
        float coeffs[] = {
            0.999999999973088f, // x
                -0.1666666663960699f, // x^3
                0.00833333287058762f, // x^5
                -0.0001984123883227529f, // x^7,
                2.755627491096882e-6f, // x^9
                -2.503262029557047e-8f, // x^11
                1.58535563425041e-10f // x^13
        };
        float p13 = coeffs[6];
        float p11 = p13 * x * x + coeffs[5];
        float p9 = p11 * x * x + coeffs[4];
        float p7 = p9 * x * x + coeffs[3];
        float p5 = p7 * x * x + coeffs[2];
        float p3 = p5 * x * x + coeffs[1];
        float p1 = p3 * x * x + coeffs[0];
        return p1 * x;
    }

    static inline float Sin(float x)
    {
        int n = int(floor(x * M_1_PI));
        float f = x - float(n * M_PI);
        float v = SinPi(f);
        return v * (n & 1 ? -1.0f : 1.0f);
    }

    static inline float Cos(float x)
    {
        x = M_PI_2 - x;
        int n = int(floor(x * M_1_PI));
        float f = x - float(n * M_PI);
        float v = SinPi(f);
        return v * (n & 1 ? -1.0f : 1.0f);
    }
};

struct AlgV3
{
    static float SinPi(float x)
    {
        float coeffs[] = {
    -0.101321183346709072589001712988183609230944236760490476f, // x
     0.00662087952180793343258682906697112938547424931632185616f, // x^3
    -0.000173505057912483501491115906801116298084629719204655552f, // x^5
     2.52229235749396866288379170129828403876289663605034418e-6f, // x^7
    -2.33177897192836082466066115718536782354224647348350113e-8f, // x^9
     1.32913446369766718120324917415992976452154154051525892e-10f, // x^11
        };
        float x2 = x * x;
        float p11 = coeffs[5];
        float p9 = p11 * x2 + coeffs[4];
        float p7 = p9 * x2 + coeffs[3];
        float p5 = p7 * x2 + coeffs[2];
        float p3 = p5 * x2 + coeffs[1];
        float p1 = p3 * x2 + coeffs[0];
        return (x - 3.1415927410125732f + 0.00000008742277657347586f) *
            (x + 3.1415927410125732f - 0.00000008742277657347586f) * p1 * x;
    }

    static inline float Sin(float x)
    {
        int n = int(floor(x * M_1_PI));
        float f = x - float(n * M_PI);
        float v = SinPi(f);
        return v * (n & 1 ? -1.0f : 1.0f);
    }

    static inline float Cos(float x)
    {
        x = M_PI_2 - x;
        int n = int(floor(x * M_1_PI));
        float f = x - float(n * M_PI);
        float v = SinPi(f);
        return v * (n & 1 ? -1.0f : 1.0f);
    }
};

struct AlgV4
{
    static float SinPiN(float x)
    {
        float coeffs[] = {
            -3.1415926444234477f,   // x
                2.0261194642649887f,   // x^3
                -0.5240361513980939f,   // x^5
                0.0751872634325299f,   // x^7
                -0.006860187425683514f, // x^9
                0.000385937753182769f, // x^11
        };
        float x2 = x * x;
        float p11 = coeffs[5];
        float p9 = p11 * x2 + coeffs[4];
        float p7 = p9 * x2 + coeffs[3];
        float p5 = p7 * x2 + coeffs[2];
        float p3 = p5 * x2 + coeffs[1];
        float p1 = p3 * x2 + coeffs[0];
        return (x - 1.0f)* (x + 1.0f) * p1 * x;
    }

    static inline float Sin(float x)
    {
        float nx = x * M_1_PI;
        float fn = floor(nx);
        float v = SinPiN(nx - fn);
        int n = int(fn);
        return v * (n & 1 ? -1.0f : 1.0f);
    }

    static inline float Cos(float x)
    {
        return Sin(float(M_PI_2) - x);
    }
};

struct AlgV5
{
    static float SinPiN(float x)
    {
        float coeffs[] = {
            -3.1415926444234477f,   // x
                2.0261194642649887f,   // x^3
                -0.5240361513980939f,   // x^5
                0.0751872634325299f,   // x^7
                -0.006860187425683514f, // x^9
                0.000385937753182769f, // x^11
        };
        float x2 = x * x;
        float p11 = coeffs[5];
        float p9 = p11 * x2 + coeffs[4];
        float p7 = p9 * x2 + coeffs[3];
        float p5 = p7 * x2 + coeffs[2];
        float p3 = p5 * x2 + coeffs[1];
        float p1 = p3 * x2 + coeffs[0];
        return (x - 1.0f) * (x + 1.0f) * p1 * x;
    }

    static inline float Sin(float x)
    {
        float nx = x * M_1_PI;
        float fn = floor(nx);
        float v = SinPiN(nx - fn);
        int n = int(fn);
        return v * (n & 1 ? -1.0f : 1.0f);
    }

    static inline float Cos(float x)
    {
        float nx = x * M_1_PI * 0.5;
        float fn = floor(nx);
        float f = nx - fn;
        float a = 0.25 - f;
        return SinPiN(a*2.0);
    }
};

template<class A> void Sin(const vec_t& src, vec_t& dst)
{
    for (int i = 0; i < src.n; ++i)
        dst.p[i] = typename A::Sin(src.p[i]);
}

template<class A> void Cos(const vec_t& src, vec_t& dst)
{
    for (int i = 0; i < src.n; ++i)
        dst.p[i] = typename A::Cos(src.p[i]);
}

template< class A> void test(const std::string& desc, int size)
{
    const float min = -23.0f, max = 23.0f;
    std::cout << "Test " << desc << " for " << size << " [" << min << " .. " << max << "] : " << std::endl;

    vec_t a(size), s0(size), s1(size), c0(size), c1(size);
    srand(0);
    init(a, min, max, false);

    Sin<AlgV0>(a, s0);
    Cos<AlgV0>(a, c0);

    Sin<A>(a, s1);
    Cos<A>(a, c1);

    diff_t ds, dc;
    diff(s0, s1, ds);
    diff(c0, c1, dc);

    std::cout << " Diff sin: " << ds.d.info(8) << " ; cos: " << dc.d.info(8) << std::endl;
    std::cout << std::endl;
}

#define TEST(N, alg) test<alg>(#alg, N)

int main(int argc, char* argv[])
{
    int N = 1024 * 64;
    if (argc > 1) N = atoi(argv[1]);

    TEST(N, AlgV1);
    TEST(N, AlgV2);
    TEST(N, AlgV3);
    TEST(N, AlgV4);
    TEST(N, AlgV5);

    return 0;
}