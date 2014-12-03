#ifndef __FMVD_UTILS_H__
#define __FMVD_UTILS_H__

extern "C" {

static void exit_on_error(void *)
{
	printf("exiting... \n");
	exit(EXIT_FAILURE);
}

static void (*on_error)(void *) = exit_on_error;

static void *hparam = NULL;

inline void setErrorHandler(void (*handler)(void *), void *param)
{
	hparam = param;
	on_error = handler;
	printf("set error handler\n");
}

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
		char message[256];
		sprintf(message, "Cuda error %d: %s in %s (line %i)",
				(int)err, cudaGetErrorString(err),
				file, line);
		fprintf(stderr, "%s\n", message);

		on_error(hparam);
	}
}

#define checkCudaErrors(ans) {__gpuAssert((ans), __FILE__, __LINE__); }
inline void
__gpuAssert(unsigned int code, const char *file, int line)
{
	if(code != cudaSuccess) {
		char message[256];
		sprintf(message, "Cuda error %d: %s in %s (line %i)",
				code, cudaGetErrorString((cudaError_t)code),
				file, line);
		fprintf(stderr, "%s\n", message);
		const char *bla = message;

		on_error(hparam);
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

