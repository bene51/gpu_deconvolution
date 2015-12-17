#include "Transform.h"

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "TransformKernels.cuh"
#include "CudaUtils.h"

template<typename T>
Transform<T>::Transform(
		const T * const * const data,
		int dataWidth, int dataHeight, int dataDepth,
		int targetWidth, int targetHeight, int targetDepth,
		const float * const inverseTransform) :
	dataWidth_(dataWidth), dataHeight_(dataHeight), dataDepth_(dataDepth),
	targetWidth_(targetWidth),
	targetHeight_(targetHeight),
	targetDepth_(targetDepth),
	planeSize_(targetWidth * targetHeight * sizeof(T))
{
	const cudaExtent volumeSize = make_cudaExtent(
			dataWidth, dataHeight, dataDepth);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
	checkCudaErrors(cudaMalloc3DArray(&d_data_, &desc, volumeSize));
	checkCudaErrors(cudaMallocHost((void **)&h_transformed_, planeSize_));

	// copy data to 3D array
	for(int z = 0; z < dataDepth; z++) {
		cudaMemcpy3DParms copyParams = {0};
		copyParams.dstArray = d_data_;
		copyParams.extent   = make_cudaExtent(dataWidth, dataHeight, 1);
		copyParams.kind     = cudaMemcpyHostToDevice;
		copyParams.srcPtr   = make_cudaPitchedPtr((void *)data[z],
			       	dataWidth * sizeof(T), dataWidth, dataHeight);
		copyParams.dstPos = make_cudaPos(0, 0, z);
		checkCudaErrors(cudaMemcpy3D(&copyParams));
	}

	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_data_;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeBorder;
	texDescr.addressMode[1] = cudaAddressModeBorder;
	texDescr.addressMode[2] = cudaAddressModeBorder;
	texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors(cudaCreateTextureObject(
				&texture_, &texRes, &texDescr, NULL));

	checkCudaErrors(cudaMalloc((void **)&d_transformed_,
				nStreams * planeSize_));
	checkCudaErrors(cudaMalloc((void **)&d_inverseTransform_,
				12 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_inverseTransform_, inverseTransform,
			       	12 * sizeof(float), cudaMemcpyHostToDevice));

	streams_ = new cudaStream_t[nStreams];
		// (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));

	for(int streamIdx = 0; streamIdx < nStreams; streamIdx++)
		cudaStreamCreate(&streams_[streamIdx]);
}

template<typename T>
void
Transform<T>::createWeights(int border, float zspacing, const char *maskfile)
{
	int streamIdx = 0;
	cudaStream_t stream = streams_[streamIdx];
	int planesizeFloat = targetWidth_ * targetHeight_ * sizeof(float);
	float *d_weights;
	checkCudaErrors(cudaMalloc((void **)&d_weights, planesizeFloat));
	float *h_weights;
	checkCudaErrors(cudaMallocHost((void **)&h_weights, planesizeFloat));
	createWeightsCuda(
			d_weights,
			targetDepth_ / 2,
			dataWidth_, dataHeight_, dataDepth_,
			zspacing,
			targetWidth_, targetHeight_, (float)border,
			d_inverseTransform_,
			stream);
	checkCudaErrors(cudaMemcpyAsync(h_weights, d_weights,
			planesizeFloat, cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	FILE *maskout = fopen(maskfile, "wb");
	fwrite(h_weights, sizeof(float), targetWidth_ * targetHeight_, maskout);
	fclose(maskout);
	cudaFree(d_weights);
	cudaFreeHost(h_weights);
}

template<typename T>
void
Transform<T>::transform(const char *outfile)
{
	FILE *out = fopen(outfile, "wb");
	int targetWH = targetWidth_ * targetHeight_;

#ifdef _WIN32
	int start = GetTickCount();
#endif
	int streamIdx;
	for(int z = 0; z < targetDepth_; z++) {
		streamIdx = z % nStreams;
		cudaStream_t stream = streams_[streamIdx];
		T *d_transformed = d_transformed_ + streamIdx * targetWH;

		// save the data before overwriting
		if(z >= nStreams) {
			checkCudaErrors(cudaMemcpyAsync(
						h_transformed_,
					       	d_transformed,
					       	planeSize_,
					       	cudaMemcpyDeviceToHost,
					       	stream));
			checkCudaErrors(cudaStreamSynchronize(stream));
			fwrite(h_transformed_, sizeof(T), targetWH, out);
		}

		// launch the kernel
		transformPlaneCuda(texture_, d_transformed, z,
				targetWidth_, targetHeight_,
				d_inverseTransform_, stream);
	}

	for(int z = 0; z < nStreams; z++) {
		streamIdx = (streamIdx + 1) % nStreams;
		cudaStream_t stream = streams_[streamIdx];
		T *d_transformed = d_transformed_ + streamIdx * targetWH;

		// save the remaining planes
		checkCudaErrors(cudaMemcpyAsync(h_transformed_, d_transformed,
				planeSize_, cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaStreamSynchronize(stream));
		fwrite(h_transformed_, sizeof(T), targetWH, out);
	}
#ifdef _WIN32
	int end = GetTickCount();
	printf("needed %d ms\n", (end - start));
#endif
	fclose(out);
}

template<typename T>
Transform<T>::~Transform()
{
	cudaDestroyTextureObject(texture_);
	cudaFreeArray(d_data_);
	cudaFree(d_transformed_);
	cudaFree(d_inverseTransform_);
	cudaFreeHost(h_transformed_);
	for(int streamIdx = 0; streamIdx < nStreams; streamIdx++)
		checkCudaErrors(cudaStreamDestroy(streams_[streamIdx]));
	delete[] streams_;
}

// excplicit template instantiation
template class Transform<unsigned char>;
template class Transform<unsigned short>;

