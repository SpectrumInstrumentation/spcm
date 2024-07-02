template<typename T> 
__global__ void CudaKernelInvert (T* pcIn, T* pcOut) 
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    pcOut[i] = -1 * pcIn[i]; 
    }