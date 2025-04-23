#include <tmatrix/common.h>
#include <cuda_runtime.h>
#include <tmatrix/Utils/msg.h>

bool file_exists(const char* filename)
{
    FILE*f = fopen(filename, "r");
    if (f == NULL)
        return false;
    fclose(f);
    return true;
}

void cudaInit(int device)
{
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    echo(info, "Device %d: %s, compute capability: %d.%d", device, prop.name, prop.major, prop.minor);
}

void cudaDebug(std::string&msg)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        echo(error, "%s: %s", msg.c_str(), cudaGetErrorString(err));
        exit(1);
    }
}