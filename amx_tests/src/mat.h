#pragma once

#include "bf16.h"

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

//-------------------------------------------------------------------------------------------------

template<class T> inline void Fill(Mat<T>& mat, T val = T(0))
{
    for (int i = 0, n = mat.Size(); i < n; ++i)
        mat.p[i] = val;
}

inline float Random(int order = 1)
{
    float val = 0;
    for (int o = 0; o < order; ++o)
        val += float(rand()) / float(RAND_MAX);
    return val;
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
