#include "defs.h"
#include <immintrin.h>


int main(int argc, char* argv[])
{
    std::cout << "xbyak_tests:" << std::endl;

    TestAdd2Ints();

    TestAdd2Fp32Vecs();

    return 0;
}