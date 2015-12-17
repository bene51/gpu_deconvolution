#include "Deconvolve.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "DeconvolveKernels.cuh"
#include "CudaUtils.h"

template<typename T>
void
Deconvolve<T>::padData(int v, int stream)
{
	int fftOffset	   = stream * fftW_  * fftH_;
	int dataOffset	   = stream * dataW_ * dataH_;
	padDataClampToBorderAndInitialize(
		d_estimate_ + fftOffset,
		d_paddedData_[v] + fftOffset,
		d_data_[v] + dataOffset,
		fftH_,
		fftW_,
		dataH_,
		dataW_,
		kernelH_,
		kernelW_,
		streams_[stream]);
}

template<typename T>
void
Deconvolve<T>::prepareKernels(
		float **h_kernel,
		IterationType::Type type)
{
	float *d_kernel, *d_paddedKernel, *d_paddedKernelHat;

	int paddedsize = fftH_ * fftW_ * sizeof(float);
	int kernelsize = kernelH_ * kernelW_ * sizeof(float);

	checkCudaErrors(cudaMalloc((void **)&d_kernel,          kernelsize));
	checkCudaErrors(cudaMalloc((void **)&d_paddedKernel,    paddedsize));
	checkCudaErrors(cudaMalloc((void **)&d_paddedKernelHat, paddedsize));

	float **h_kernel2 = IterationType::createKernelHat(h_kernel, kernelW_, kernelH_,
			nViews_, type);

	for(int v = 0; v < nViews_; v++)
		IterationType::normalizeKernel(h_kernel[v], kernelW_ * kernelH_);

	for(int v = 0; v < nViews_; v++) {
		checkCudaErrors(cudaMemcpy(d_kernel, h_kernel[v], kernelsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_paddedKernel, 0, paddedsize));
		padKernel(d_paddedKernel, d_kernel, fftH_, fftW_, kernelH_, kernelW_, 0); // default stream
		checkCudaErrors(cufftExecR2C(fftPlanFwd_[0], (cufftReal *)d_paddedKernel, (cufftComplex *)d_kernelSpectrum_[v]));

		checkCudaErrors(cudaMemcpy(d_kernel, h_kernel2[v], kernelsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_paddedKernelHat, 0, paddedsize));
		padKernel(d_paddedKernelHat, d_kernel, fftH_, fftW_, kernelH_, kernelW_, 0); // default stream
		checkCudaErrors(cufftExecR2C(fftPlanFwd_[0], (cufftReal *)d_paddedKernelHat, (cufftComplex *)d_kernelHatSpectrum_[v]));
	}

	for(int v = 0; v < nViews_; v++)
		delete h_kernel2[v];
	delete h_kernel2;
	checkCudaErrors(cudaFree(d_kernel));
	checkCudaErrors(cudaFree(d_paddedKernel));
	checkCudaErrors(cudaFree(d_paddedKernelHat));
}

template<typename T>
void
Deconvolve<T>::prepareWeights(float const * const * const h_weights)
{
	float *d_weights;
	float *d_paddedWeightSums;

	int paddedsizeFloat = fftH_ * fftW_ * sizeof(float);
	int datasize = dataH_ * dataW_ * sizeof(float);

	checkCudaErrors(cudaMalloc((void **)&d_weights, datasize));
	checkCudaErrors(cudaMalloc((void **)&d_paddedWeightSums, paddedsizeFloat));
	checkCudaErrors(cudaMemset(d_paddedWeightSums, 0, paddedsizeFloat));
	for(int v = 0; v < nViews_; v++) {
		checkCudaErrors(cudaMemcpy(d_weights, h_weights[v], datasize, cudaMemcpyHostToDevice));
		padWeights(
			d_paddedWeights_[v],
			d_paddedWeightSums,
			d_weights,
			fftH_,
			fftW_,
			dataH_,
			dataW_,
			kernelH_,
			kernelW_,
			0 // default stream
		);
	}

	for(int v = 0; v < nViews_; v++) {
		printf("Normalizing weights for view %d\n", v);
		normalizeWeights(d_paddedWeights_[v], d_paddedWeightSums, fftH_, fftW_, 0);
	}

	checkCudaErrors(cudaFree(d_weights));
	checkCudaErrors(cudaFree(d_paddedWeightSums));
}

void *fmvd_malloc(size_t size)
{
	void *p;
	checkCudaErrors(cudaMallocHost(&p, size));
	return p;
}

void fmvd_free(void *p)
{
	cudaFreeHost(p);
}

template<typename T>
Deconvolve<T>::Deconvolve(
		int dataH, int dataW,
		float const * const * const h_weights,
		float **h_kernel, int kernelH, int kernelW,
		IterationType::Type type,
		int nViews,
		int nStreams,
		DataRetrieverInterface<T> *dataRetriever,
		DataReceiverInterface<T> *dataReceiver) :
	dataH_(dataH), dataW_(dataW), kernelH_(kernelH),
	kernelW_(kernelW), nViews_(nViews), nStreams_(nStreams),
	dataRetriever_(dataRetriever), dataReceiver_(dataReceiver)
{
	fftH_ = snapTransformSize(dataH + kernelH - 1);
	fftW_ = snapTransformSize(dataW + kernelW - 1);
	int fftsize = fftH_ * (fftW_ / 2 + 1) * sizeof(fComplex);
	int paddedsizeFloat = fftH_ * fftW_ * sizeof(float);
	int paddedsizeData = fftH_ * fftW_ * sizeof(T);
	int datasize = dataH * dataW * sizeof(T);

	h_data_              = (T **)  malloc(nViews * sizeof(T *));
	d_data_              = (T **)  malloc(nViews * sizeof(T *));
	d_paddedData_        = (T **)  malloc(nViews * sizeof(T *));
	d_paddedWeights_     = (float **)   malloc(nViews * sizeof(float *));
	d_kernelSpectrum_    = (fComplex **)malloc(nViews * sizeof(fComplex *));
	d_kernelHatSpectrum_ = (fComplex **)malloc(nViews * sizeof(fComplex *));

	for(int v = 0; v < nViews; v++) {
		checkCudaErrors(cudaMallocHost(&h_data_[v],                   nStreams * datasize));
		checkCudaErrors(cudaMalloc((void **)&d_data_[v],              nStreams * datasize));
		checkCudaErrors(cudaMalloc((void **)&d_paddedData_[v],        nStreams * paddedsizeData));
		checkCudaErrors(cudaMalloc((void **)&d_paddedWeights_[v],     paddedsizeFloat));
		checkCudaErrors(cudaMalloc((void **)&d_kernelSpectrum_[v],    nStreams * fftsize));
		checkCudaErrors(cudaMalloc((void **)&d_kernelHatSpectrum_[v], nStreams * fftsize));
	}

	checkCudaErrors(cudaMalloc((void **)&d_estimate_,         nStreams * paddedsizeFloat));
	checkCudaErrors(cudaMalloc((void **)&d_tmp_,              nStreams * paddedsizeFloat));
	checkCudaErrors(cudaMalloc((void **)&d_estimateSpectrum_, nStreams * fftsize));


	fftPlanFwd_ = (cufftHandle *)malloc(nStreams * sizeof(cufftHandle));
	fftPlanInv_ = (cufftHandle *)malloc(nStreams * sizeof(cufftHandle));
	streams_    = (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));

	for(int stream = 0; stream < nStreams; stream++) {
		cudaStreamCreate(&streams_[stream]);
		checkCudaErrors(cufftPlan2d(&fftPlanFwd_[stream], fftH_, fftW_, CUFFT_R2C));
		checkCudaErrors(cufftSetStream(fftPlanFwd_[stream], streams_[stream]));
		checkCudaErrors(cufftPlan2d(&fftPlanInv_[stream], fftH_, fftW_, CUFFT_C2R));
		checkCudaErrors(cufftSetStream(fftPlanInv_[stream], streams_[stream]));
	}

	prepareKernels(h_kernel, type);
	prepareWeights(h_weights);
}

template<typename T>
Deconvolve<T>::~Deconvolve()
{
	for(int v = 0; v < nViews_; v++) {
		checkCudaErrors(cudaFreeHost(h_data_[v]));
		checkCudaErrors(cudaFree(d_data_[v]));
		checkCudaErrors(cudaFree(d_paddedData_[v]));
		checkCudaErrors(cudaFree(d_paddedWeights_[v]));
		checkCudaErrors(cudaFree(d_kernelSpectrum_[v]));
		checkCudaErrors(cudaFree(d_kernelHatSpectrum_[v]));
	}
	checkCudaErrors(cudaFree(d_estimate_));
	checkCudaErrors(cudaFree(d_estimateSpectrum_));
	checkCudaErrors(cudaFree(d_tmp_));

	free(h_data_);
	free(d_data_);
	free(d_paddedData_);
	free(d_paddedWeights_);
	free(d_kernelSpectrum_);
	free(d_kernelHatSpectrum_);

	for(int stream = 0; stream < nStreams_; stream++) {
		checkCudaErrors(cufftDestroy(fftPlanInv_[stream]));
		checkCudaErrors(cufftDestroy(fftPlanFwd_[stream]));
		checkCudaErrors(cudaStreamDestroy(streams_[stream]));
	}
	free(streams_);
	free(fftPlanFwd_);
	free(fftPlanInv_);
}

template<typename T>
void
Deconvolve<T>::deconvolvePlanes(int iterations)
{
	int paddedsizeFloat = fftH_ * fftW_ * sizeof(float);
	int paddedsizeData = fftH_ * fftW_ * sizeof(T);
	int datasize = dataH_ * dataW_ * sizeof(T);

	int streamIdx;
	for(streamIdx = 0; streamIdx < nStreams_; streamIdx++) {
		int dataOffset = streamIdx * dataH_ * dataW_;
		dataRetriever_->getNextPlane(h_data_, dataOffset);
	}

#ifdef _WIN32
	int start = GetTickCount();
#endif
	for(int z = 0; ; z++) {
		streamIdx = z % nStreams_;
		cudaStream_t stream = streams_[streamIdx];
		int fftOffset	   = streamIdx * fftW_ * fftH_;
		int spectrumOffset = streamIdx * (fftW_ / 2 + 1) * fftH_;
		int dataOffset	   = streamIdx * dataW_ * dataH_;

		float *d_estimate = d_estimate_ + fftOffset;
		fComplex *d_estimateSpectrum = d_estimateSpectrum_ + spectrumOffset;
		float *d_tmp = d_tmp_ + fftOffset;

		cufftHandle fftFwd = fftPlanFwd_[streamIdx];
		cufftHandle fftInv = fftPlanInv_[streamIdx];

		// load from HDD
		if(z >= nStreams_) {
			checkCudaErrors(cudaStreamSynchronize(stream));
			dataReceiver_->returnNextPlane(h_data_[0] + dataOffset);
			bool hasMore = dataRetriever_->getNextPlane(h_data_, dataOffset);
			if(!hasMore)
				break;
		}

		// H2D
		for(int v = 0; v < nViews_; v++) {
			// load data from file and upload it to GPU
			T *d_data = d_data_[v] + dataOffset;
			T *h_data = h_data_[v] + dataOffset;
			checkCudaErrors(cudaMemcpyAsync(d_data, h_data, datasize, cudaMemcpyHostToDevice, stream));
		}
		checkCudaErrors(cudaMemsetAsync(d_estimate, 0, paddedsizeFloat, streams_[streamIdx]));

		for(int v = 0; v < nViews_; v++) {
			T *d_paddedData = d_paddedData_[v] + fftOffset;
			checkCudaErrors(cudaMemsetAsync(d_paddedData, 0, paddedsizeData, stream));
			padData(v, streamIdx);
		}

		for(int it = 0; it < iterations; it++) {
			for(int v = 0; v < nViews_; v++) {
				checkCudaErrors(cufftExecR2C(fftFwd, (cufftReal *)d_estimate, (cufftComplex *)d_estimateSpectrum));
				modulateAndNormalize(d_estimateSpectrum, d_kernelSpectrum_[v], fftH_, fftW_, 1, stream);
				checkCudaErrors(cufftExecC2R(fftInv, (cufftComplex *)d_estimateSpectrum, (cufftReal *)d_tmp));
				divide(d_paddedData_[v] + fftOffset, d_tmp, d_tmp, fftH_, fftW_, stream);

				checkCudaErrors(cufftExecR2C(fftFwd, (cufftReal *)d_tmp, (cufftComplex *)d_estimateSpectrum));
				modulateAndNormalize(d_estimateSpectrum, d_kernelHatSpectrum_[v], fftH_, fftW_, 1, stream);
				checkCudaErrors(cufftExecC2R(fftInv, (cufftComplex *)d_estimateSpectrum, (cufftReal *)d_tmp));
				multiply32(d_estimate, d_tmp, d_paddedWeights_[v], d_estimate, fftH_, fftW_, stream);
			}
		}

		unpadData(d_data_[0] + dataOffset, d_estimate, fftH_, fftW_, dataH_, dataW_, stream);

		// D2H
		checkCudaErrors(cudaMemcpyAsync(
				h_data_[0] + dataOffset,
				d_data_[0] + dataOffset,
				datasize,
				cudaMemcpyDeviceToHost,
				stream));

	}

	// save the remaining nStreams planes
	for(int z = 0; z < nStreams_ - 1; z++) {
		streamIdx = (streamIdx + 1) % nStreams_;
		int dataOffset = streamIdx * dataW_ * dataH_;
		checkCudaErrors(cudaStreamSynchronize(streams_[streamIdx]));
		dataReceiver_->returnNextPlane(h_data_[0] + dataOffset);
	}

#ifdef _WIN32
	int stop = GetTickCount();
	printf("Overall time: %d ms\n", (stop - start));
#endif
}

// explicit template instantiation
template class Deconvolve<unsigned char>;
template class Deconvolve<unsigned short>;

