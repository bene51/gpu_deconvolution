#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

void
setErrorHandler(void (*handler)(void *, const char *), void *param);

int
iDivUp(int a, int b);

int
snapTransformSize(int dataSize);

#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
void
__getLastCudaError(const char *errorMessage, const char *file, const int line);

#define checkCudaErrors(ans) {__gpuAssert((ans), __FILE__, __LINE__); }
void
__gpuAssert(unsigned int code, const char *file, int line);

int
getNumCUDADevices();

void
getCudaDeviceName(int dev, char* name);

void
setCudaDevice(int dev);


#endif

