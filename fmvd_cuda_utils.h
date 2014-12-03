#ifndef __FMVD_UTILS_H__
#define __FMVD_UTILS_H__

extern "C" {

inline int
iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
inline void
__getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA error : %s : (%d) %s.\n",
				file, line, errorMessage, (int)err,
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(ans) {__gpuAssert((ans), __FILE__, __LINE__); }
inline void
__gpuAssert(unsigned int code, const char *file, int line, bool abort=true)
{
	if(code != cudaSuccess) {
		const char *str = cudaGetErrorString((cudaError_t)code);
		fprintf(stderr, "GPUAssert: error %d %s %d\n", code, file, line);
		fprintf(stderr, "%s\n", str);
		if(abort)
			exit(code);
	}
}

inline int
getNumCUDADevices()
{
	int count = 0;
	checkCudaErrors(cudaGetDeviceCount(&count));
	return count;
}

inline void
getCudaDeviceName(int dev, char* name)
{
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
	memcpy(name,prop.name,sizeof(char)*256);
}

inline void
setCudaDevice(int dev)
{
	checkCudaErrors(cudaSetDevice(dev));
}


}
#endif

