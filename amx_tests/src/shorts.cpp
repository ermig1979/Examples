#include <immintrin.h>
#include <stdint.h>
#include <iostream>
#if defined(__linux__) && 0
#include <unistd.h>
#include <sys/syscall.h>

const int ARCH_REQ_XCOMP_PERM = 0x1023;
const int XFEATURE_XTILEDATA = 18;

void ConvertA(const float* src, uint16_t* dst)
{
    __m512 s0 = _mm512_loadu_ps(src + 0 * 16);
    __m512 s1 = _mm512_loadu_ps(src + 1 * 16);
    _mm512_storeu_si512(dst, (__m512i)_mm512_cvtne2ps_pbh(s1, s0));
}

void ConvertB(const float* src, int stride, uint16_t* dst)
{
    static const __m512i PERM_IDX = _mm512_set_epi16(
        0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c,
        0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08,
        0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04,
        0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);
    __m512 s0 = _mm512_loadu_ps(src + 0 * stride);
    __m512 s1 = _mm512_loadu_ps(src + 1 * stride);
    __m512i d = (__m512i)_mm512_cvtne2ps_pbh(s1, s0);
    _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM_IDX, d));
} // Конвертация в BF16 с переупорядочиванием четных и нечетных строк.

struct TileConfig
{
    uint8_t paletteId; // должен быть установлен в 1
    uint8_t startRow; // должен быть установлен в 0
    uint8_t reserved[14];
    uint16_t colsb[16]; // актуальная длина строк матриц в байтах
    uint8_t rows[16]; // актуальное число строк в матрицах
};

int main()
{
    // Инициализация AMX в Linux:
    if (syscall(SYS_arch_prctl,
        ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0)
    {
        std::cout << "Can't initialize AMX!" << std::endl;
        return 1;
    }

    float A[16][32], B[32][16], C[16][16];

    uint16_t a[16][32];
    for (int i = 0; i < 16; ++i)
        ConvertA(A[i], a[i]);

    uint16_t b[16][32];
    for (int i = 0; i < 16; ++i)
        ConvertB(B[i * 2], 16, b[i]);

    TileConfig conf = {};
    conf.paletteId = 1;
    conf.rows[0] = 16;
    conf.colsb[0] = 16 * 4;
    conf.rows[1] = 16;
    conf.colsb[1] = 16 * 4;
    conf.rows[2] = 16;
    conf.colsb[2] = 16 * 4;
    _tile_loadconfig(&conf);// Загрузка конфигурации AMX

    _tile_zero(0); // обнуление 0-го рестра

    _tile_loadd(1, a, 64); // загрузка матрицы A в 1-й регистр

    _tile_loadd(2, b, 64); // загрузка матрицы B в 2-й регистр

    _tile_dpbf16ps(0, 1, 2);// непосредственно умножение С += A * B

    _tile_stored(0, C, 64); // сохранение рузультата в матрицу С

    _tile_release(); // очистка AMX конфигурации

    return 0;
}


#endif

//FOR m = 0 TO dst.rows - 1
//    FOR k = 0 TO (a.colsb / 4) - 1
//        FOR n = 0 TO (dst.colsb / 4) - 1
//            dst[m][n] += FP32(a[m][2 * k + 0]) * FP32(b[k][2 * n + 0])
//            dst[m][n] += FP32(a[m][2 * k + 1]) * FP32(b[k][2 * n + 1])


void PerfBf16L0(int count)
{
    for (int i = 0; i < count; i += 4)
    {
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    }
}

void PerfInt8L0(int count)
{
    for (int i = 0; i < count; i += 4)
    {
        _tile_dpbuud(0, 4, 6);
        _tile_dpbuud(1, 4, 7);
        _tile_dpbuud(2, 5, 6);
        _tile_dpbuud(3, 5, 7);
    }
}

void LoadCompact(int count, uint8_t* buf)
{
    for (int i = 0; i < count; i++)
        _tile_loadd(0, buf + i * 1024, 64);
}

void LoadLongRows(int count, uint8_t* buf)
{
    for (int i = 0; i < count; i++)
        _tile_loadd(0, buf + i * 64, 64 * count);
}