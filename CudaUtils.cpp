#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "CudaUtils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

static void exit_on_error(void *, const char *)
{
	printf("exiting... \n");
	exit(EXIT_FAILURE);
}

static void (*bla_on_error)(void *, const char *) = exit_on_error;

static void *hparam = NULL;

void
setErrorHandler(void (*handler)(void *, const char *), void *param)
{
	hparam = param;
	bla_on_error = handler;
}

int
iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

static int
iAlignUp(int a, int b)
{
	//Align a to nearest higher multiple of b
	return (a % b != 0) ? (a - a % b + b) : a;
}

int
snapTransformSize(int dataSize)
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
			break;

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
		return dataSize;

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 1024)
		return hiPOT;
	else
		return iAlignUp(dataSize, 512);
}


void
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
		bla_on_error(hparam, message);
	}
}

void
__gpuAssert(unsigned int code, const char *file, int line)
{
	if(code != cudaSuccess) {
		char message[256];
		sprintf(message, "Cuda error %d: %s in %s (line %i)",
				code, cudaGetErrorString((cudaError_t)code),
				file, line);
		fprintf(stderr, "%s\n", message);
		bla_on_error(hparam, message);
	}
}

int
getNumCUDADevices()
{
	int count = 0;
	checkCudaErrors(cudaGetDeviceCount(&count));
	return count;
}

void
getCudaDeviceName(int dev, char* name)
{
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
	memcpy(name,prop.name,sizeof(char)*256);
}

void
setCudaDevice(int dev)
{
	cudaError_t err = cudaSetDevice(dev);
	checkCudaErrors(err);
}


