#ifndef __FMVD_UTILS_H__
#define __FMVD_UTILS_H__

extern "C" {

inline int
iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
				file, line, errorMessage, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

}
#endif

