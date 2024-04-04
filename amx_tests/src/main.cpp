#include "mat.h"
#include "amx.h"
#include "test.h"

int main(int argc, char* argv[])
{
    const int S = 1536;
    int M = S, N = S, K = S;
    if (argc > 1) M = N = K = atoi(argv[1]);
    if (argc > 2) N = K = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    if (M % 96 || N % 32 || K % 32)
    {
        std::cout << "Wrong input sizes!" << std::endl;
        return 1;
    }

    Amx::InitAmx();

    PrintCurrentFrequency();

    if (M && N && K)
    {
        if (!TestGemm(M, N, K))
            return 1;
    }

    Amx::TestPerf();

    PrintCurrentFrequency();

    return 0;
}