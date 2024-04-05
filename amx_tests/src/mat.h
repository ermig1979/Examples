#pragma once

#include "defs.h"

template<class T> struct Mat
{
    T* p;
    int m, n;

    Mat(int _m, int _n) : m(_m), n(_n), p((T*)_mm_malloc(_m * _n * sizeof(T), 64)) {}
    ~Mat() { _mm_free(p); }
    int Size() const { return m * n; }
};

typedef Mat<float> Mat32f;
typedef Mat<int32_t> Mat32i;
typedef Mat<uint16_t> Mat16b;
typedef Mat<uint8_t> Mat8u;

//-------------------------------------------------------------------------------------------------

union F32
{
    F32(float val) : f32{ val } {   }
    F32(uint32_t val) : u32{ val } {  }

    float f32;
    uint32_t u32;
};

inline float Round(float src)
{
    return F32((F32(src).u32 + 0x8000) & 0xFFFF0000).f32;
}

inline void Convert(float src, uint16_t& dst)
{
    dst = uint16_t((F32(src).u32 + 0x8000) >> 16);
}

inline void Convert(uint16_t src, float& dst)
{
    dst = F32(uint32_t(src) << 16).f32;
}

inline uint16_t To16b(float val)
{
    return uint16_t((F32(val).u32 + 0x8000) >> 16);
}

inline float To32f(uint16_t val)
{
    return F32(uint32_t(val) << 16).f32;
}

//-------------------------------------------------------------------------------------------------

template<class T> inline void Fill(Mat<T>& mat, T val = T(0))
{
    for (int i = 0, n = mat.Size(); i < n; ++i)
        mat.p[i] = val;
}

inline float Random(int order = 1)
{
    float value = 0;
    for (int o = 0; o < order; ++o)
        value += float(rand()) / float(RAND_MAX);
    return value / float(order);
}

inline void Init(Mat32f & mat, float min, float max, int order = 1)
{
    float range = (max - min) / order;
    for (int i = 0, n = mat.Size(); i < n; ++i)
        mat.p[i] = Random(order) * range + min;
}

inline void Init(Mat16b & mat, float min, float max, int order = 1)
{
    float range = (max - min) / order;
    for (int i = 0, n = mat.Size(); i < n; ++i)
        Convert(Random(order) * range + min, mat.p[i]);
}

inline void ConvertA(const Mat32f& src, Mat16b& dst)
{
    assert(src.Size() == dst.Size());
    for (int i = 0, n = src.Size(); i < n; ++i)
        Convert(src.p[i], dst.p[i]);
}

inline void ConvertB(const Mat32f& src, Mat16b& dst)
{
    assert(src.m == dst.m * 2 && src.n * 2 == dst.n);
    for (int r = 0; r < src.m; r += 2)
    {
        for (int c = 0; c < src.n; c += 1)
        {
            Convert(src.p[(r + 0) * src.n + c], dst.p[r * src.n + c * 2 + 0]);
            Convert(src.p[(r + 1) * src.n + c], dst.p[r * src.n + c * 2 + 1]);
        }
    }
}

//-------------------------------------------------------------------------------------------------

struct Stat
{
    uint64_t count;
    double sum, sqsum, min, max;

    Stat()
    {
        count = 0;
        sum = 0;
        sqsum = 0;
        min = DBL_MAX;
        max = -DBL_MAX;
    }

    void Update(double val)
    {
        count++;
        sum += val;
        sqsum += val * val;
        min = std::min(min, val);
        max = std::max(max, val);
    }

    std::string Info(int precision)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(precision);
        ss << "{";
        ss << " min: " << min;
        ss << " max: " << max;
        ss << " avg: " << sum / count;
        ss << " std: " << sqrt(sqsum / count - sum * sum / count / count);
        ss << " }";
        return ss.str();
    }

    double Abs() const
    {
        return std::max(abs(max), abs(min));
    }
};

struct Diff
{
    Stat a, b, d;

    Diff() { }
};

inline bool GetDiff(const Mat32f& a, const Mat32f& b, Diff& d)
{
    if (a.m != b.m || a.n != b.n)
        return false;

    for (int i = 0, n = a.Size(); i < n; ++i)
    {
        d.a.Update(a.p[i]);
        d.b.Update(b.p[i]);
        d.d.Update(a.p[i] - b.p[i]);
    }

    return true;
}