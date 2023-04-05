#include "defs.h"
#include <immintrin.h>

static inline float Rcp(float x)
{
    return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
}

struct ErfV0
{
    static inline float Erf(float x)
    {
        return ::erf(x);
    }
};

struct ErfV1
{
    static inline float Erf(float x)
    {
        const float a0 = 1.0f, a1 = 0.278393f, a2 = 0.230389f, a3 = 0.000972f, a4 = 0.078108f;
        float sign = x > 0 ? 1.0f : -1.0f;
        float a = abs(x);
        float p = (((a * a4 + a3) * a + a2) * a + a1) * a + a0;
        float p2 = p * p;
        float p4 = p2 * p2;
        return (1.0f - 1.0f / p4) * sign;
    }
};

struct ErfV1r
{
    static inline float Erf(float x)
    {
        const float a0 = 1.0f, a1 = 0.278393f, a2 = 0.230389f, a3 = 0.000972f, a4 = 0.078108f;
        float sign = x > 0 ? 1.0f : -1.0f;
        float a = abs(x);
        float p = (((a * a4 + a3) * a + a2) * a + a1) * a + a0;
        float p2 = p * p;
        float p4 = p2 * p2;
        return (1.0f - Rcp(p4)) * sign;
    }
};

struct ErfV2
{
    static inline float Erf(float x)
    {
        const float p = 0.47047f, a1 = 0.3480242f, a2 = -0.0958798f, a3 = 0.7478556f;
        float sign = x > 0 ? 1.0f : -1.0f;
        float a = abs(x);
        float t = 1.0f / (1.0f + p * a);
        float q = ((t * a3 + a2) * t + a1) * t;
        return (1.0f - q * exp(-x*x)) * sign;
    }
};

struct ErfV3
{
    static inline float Erf(float x)
    {
        const float a0 = 1.0f, a1 = 0.0705230784f, a2 = 0.0422820123f, a3 = 0.0092705272f, a4 = 0.0001520143f, a5 = 0.0002765672f, a6 = 0.0000430638f;
        float sign = x > 0 ? 1.0f : -1.0f;
        float a = abs(x);
        float p = (((((a * a6 + a5) * a + a4) * a + a3) * a + a2) * a + a1) * a + a0;
        float p2 = p * p;
        float p4 = p2 * p2;
        float p8 = p4 * p4;
        float p16 = p8 * p8;
        return (1.0f - 1.0f / p16) * sign;
    }
};

struct ErfV3r
{
    static inline float Erf(float x)
    {
        const float a0 = 1.0f, a1 = 0.0705230784f, a2 = 0.0422820123f, a3 = 0.0092705272f, a4 = 0.0001520143f, a5 = 0.0002765672f, a6 = 0.0000430638f;
        float sign = x > 0 ? 1.0f : -1.0f;
        float a = abs(x);
        float p = (((((a * a6 + a5) * a + a4) * a + a3) * a + a2) * a + a1) * a + a0;
        float p2 = p * p;
        float p4 = p2 * p2;
        float p8 = p4 * p4;
        float p16 = p8 * p8;
        return (1.0f - Rcp(p16)) * sign;
    }
};

struct ErfV3d
{
    static inline float Erf(float x)
    {
        const double a0 = 1.0, a1 = 0.0705230784, a2 = 0.0422820123, a3 = 0.0092705272, a4 = 0.0001520143, a5 = 0.0002765672, a6 = 0.0000430638;
        double sign = x > 0 ? 1.0 : -1.0;
        double a = abs(x);
        double p = (((((a * a6 + a5) * a + a4) * a + a3) * a + a2) * a + a1) * a + a0;
        double p2 = p * p;
        double p4 = p2 * p2;
        double p8 = p4 * p4;
        double p16 = p8 * p8;
        return float((1.0 - 1.0 / p16) * sign);
    }
};

struct ErfV4
{
    static inline float Erf(float x)
    {
        const float p = 0.3275911f, a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
        float sign = x > 0 ? 1.0f : -1.0f;
        float a = abs(x);
        float t = 1.0f / (1.0f + p * a);
        float q = ((((t * a5 + a4) * t + a3) * t + a2) * t + a1) * t;
        return (1.0f - q * exp(-x * x)) * sign;
    }
};

template<class E> void Erf(const vec_t& src, vec_t& dst)
{
    for (int i = 0; i < src.n; ++i)
        dst.p[i] = typename E::Erf(src.p[i]);
}

void test(const std::string& desc, func_t func, int size, float min, float max)
{
    std::cout << "Test " << desc << " for " << size << " [" << min << " .. " << max << "] : " << std::endl;

    vec_t s(size), d0(size), d1(size);
    srand(0);
    init(s, min, max, false);

    Erf<ErfV0>(s, d0);

    func(s, d1);

    diff_t d;
    diff(d0, d1, d);

    std::cout << " Diff: " << d.d.info(8) << std::endl;
    std::cout << std::endl;
}

void test(int size, const std::string& desc, func_t func)
{
    //std::cout << std::fixed << std::setprecision(3);
    //std::cout << "TEST " << desc << " :" << std::endl;
    test(desc, func, size, -0.0f, 9.0f);
    std::cout << std::endl;
}

#define TEST(N, func) test(N, #func, func)

int main(int argc, char* argv[])
{
    int N = 1024 * 64;
    if (argc > 1) N = atoi(argv[1]);

    TEST(N, Erf<ErfV1>);
    TEST(N, Erf<ErfV1r>);
    TEST(N, Erf<ErfV2>);
    TEST(N, Erf<ErfV3>);
    TEST(N, Erf<ErfV3r>);
    TEST(N, Erf<ErfV3d>);
    TEST(N, Erf<ErfV4>);

    return 0;
}