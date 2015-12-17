#ifndef DECONVOLVE_H
#define DECONVOLVE_H

#include <cuda_runtime.h>
#include <cufft.h>

#include "DataRetrieverInterface.h"
#include "DataReceiverInterface.h"
#include "IterationType.h"
#include "DeconvolveKernels.cuh" // for fComplex

void *fmvd_malloc(size_t size);
void fmvd_free(void *p);

template<typename T>
class Deconvolve
{
private:
	int dataH_, dataW_;
	int fftH_, fftW_;
	int kernelH_, kernelW_;
	int nViews_;
	int nStreams_;

	fComplex **d_kernelSpectrum_;
	fComplex **d_kernelHatSpectrum_;

	T **h_data_;
	T **d_data_;
	T **d_paddedData_;
	float **d_paddedWeights_;

	float *d_estimate_;
	fComplex *d_estimateSpectrum_;

	float *d_tmp_;

	cudaStream_t *streams_;

	cufftHandle *fftPlanFwd_, *fftPlanInv_;

	DataRetrieverInterface<T> *dataRetriever_;
	DataReceiverInterface<T> *dataReceiver_;

	void padData(int v, int stream);
	void prepareKernels(float **h_kernel, IterationType::Type type);
	void prepareWeights(float const * const * const h_weights);
public:
	Deconvolve(
			int dataH, int dataW,
			float const * const * const h_weights,
			float **h_kernel, int kernelH, int kernelW,
			IterationType::Type type,
			int nViews,
			int nStreams,
			DataRetrieverInterface<T> *dataRetriever,
			DataReceiverInterface<T> *dataReceiver);
	~Deconvolve();

	void deconvolvePlanes(int iterations);
};

#endif // DECONVOLVE_H

