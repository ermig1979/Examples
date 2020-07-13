#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

struct Buf
{
    float* p;
    int n;

    Buf(int size) : n(size), p((float*)_mm_malloc(size * 4, 64)) {}
    ~Buf() { _mm_free(p); }
};

enum Activation
{
    Identity,
    Relu,
    RestrictRange,
    Prelu
};

struct Param
{
    int srcC, srcH, srcW;
    int dstC, dstH, dstW;
    int kernelY, kernelX;
    int strideY, strideX;
    int padY, padX, padH, padW;
    int group;
    Activation activation;
};

