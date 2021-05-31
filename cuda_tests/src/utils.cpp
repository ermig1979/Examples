#include "utils.h"

void print_device_info()
{
    int deviceCount;
    CHECK(cudaGetDeviceCount(&deviceCount));
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, device));
        printf("Device %d: '%s'.\n", device, deviceProp.name);
        printf("Compute capability: %d.%d.\n", deviceProp.major, deviceProp.minor);
        printf("Device global memory: %d MB.\n", int(deviceProp.totalGlobalMem / 1024 / 1024));
        printf("Shared memory per block: %d kB.\n", int(deviceProp.sharedMemPerBlock / 1024));
        printf("Registers per block: %d kB.\n", int(deviceProp.regsPerBlock / 1024));
        printf("\n");
    }
}

//-----------------------------------------------------------------------------

inline float square(float x)
{
    return x * x;
}

inline float invalid(float a, float b, float e2)
{
    float d2 = square(a - b);
    return d2 > e2 && d2 > e2 * (a * a + b * b);
}

bool check(const cpu_tensor_t& control, const cpu_tensor_t& current, const std::string& desc, float eps, int count)
{
    assert(control.size == current.size);
    float e2 = square(eps);
    int errors = 0;
    for (int i = 0; i < control.size; ++i)
    {
        if (invalid(control.data[i], current.data[i], e2))
        {
            std::cout << desc << " : check error at " << i << ": ";
            std::cout << std::setprecision(4) << std::fixed << control.data[i] << " != " << current.data[i] << std::endl;
            errors++;
            if (errors >= count)
                return false;
        }
    }
    return errors == 0;
}

