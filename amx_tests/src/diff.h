#pragma once

#include "mat.h"

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
        ss << " avg: " << sum /count;
        ss << " std: " << sqrt(sqsum/count - sum * sum / count / count) ;
        ss << " }";
        return ss.str();
    }
};  

struct Diff
{
    Stat a, b, d;

    Diff() { }
};

inline bool GetDiff(const Mat32f& a, const Mat32f& b, Diff & d)
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