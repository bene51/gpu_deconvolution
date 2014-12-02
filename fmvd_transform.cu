#include "fmvd_transform.h"

#include <windows.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <math_constants.h>

#include "fmvd_utils.h"

static texture<unsigned short, 3, cudaReadModeNormalizedFloat> tex;

__global__ void
transform_plane_kernel(
		unsigned short *dTransformed,
		int z,
		int w,
		int h,
		int d,
		int wTransformed,
		int hTransformed,
		const float *inv_matrix)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	const float *m = inv_matrix;

	if(x < wTransformed && y < hTransformed) {
		// apply inverse transform
		float rx = m[0] * x + m[1] * y + m[2]  * z + m[3];
		float ry = m[4] * x + m[5] * y + m[6]  * z + m[7];
		float rz = m[8] * x + m[9] * y + m[10] * z + m[11];

		/*
		// mirror
		if(rx < 0) rx = -rx; if(rx > w - 1) rx = 2 * w - rx - 2;
		if(ry < 0) ry = -ry; if(ry > h - 1) ry = 2 * h - ry - 2;
		if(rz < 0) rz = -rz; if(rz > d - 1) rz = 2 * d - rz - 2;
		*/

		float v = tex3D(tex, rx + 0.5, ry + 0.5, rz + 0.5);
		unsigned short iv = (unsigned short)(v * 65535 + 0.5);
		dTransformed[y * wTransformed + x] = iv;
	}
}

__global__ void
create_transformed_mask_kernel(
		unsigned short *dTransformed,
		int z,
		int w,
		int h,
		int d,
		float zspacing,
		int wTransformed,
		int hTransformed,
		float border,
		const float *inv_matrix)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	const float *m = inv_matrix;

	if(x < wTransformed && y < hTransformed) {
		int idx = y * wTransformed + x;
		float rx = m[0] * x + m[1] * y + m[2]  * z + m[3];
		float ry = m[4] * x + m[5] * y + m[6]  * z + m[7];
		float rz = m[8] * x + m[9] * y + m[10] * z + m[11];

		if(rx < 0 || rx >= w ||
				ry < 0 || ry >= h ||
				rz < 0 || rz >= d) {
			dTransformed[idx] = 0;
		} else {
			float pi = CUDART_PI_F;
			float v = 1;
			float dx = rx < w / 2 ? rx : w - rx;
			float dy = ry < h / 2 ? ry : h - ry;
			float dz = rz < d / 2 ? rz : d - rz;

			dx += 1;
			dy += 1;
			dz += 1;
			dz *= zspacing;

			if(dx < border)
				v = v * (0.5f * (1 - cos(dx / border * pi)));
			if(dy < border)
				v = v * (0.5f * (1 - cos(dy / border * pi)));
			if(dz < border)
				v = v * (0.5f * (1 - cos(dz / border * pi)));
			dTransformed[idx] = (unsigned short)(65535 * v + 0.5);
		}
	}
}

void
transform_plane(
		unsigned short *d_trans,
		int z,
		int w,
		int h,
		int d,
		int tw,
		int th,
		const float *d_inverse,
		cudaStream_t stream)
{
	dim3 threads(32, 32);
	dim3 grid(iDivUp(tw, threads.x), iDivUp(th, threads.y));
	transform_plane_kernel<<<grid, threads, 0, stream>>>(
			d_trans, z, w, h, d, tw, th, d_inverse);
	getLastCudaError("transform_data_kernel<<<>>> execution failed\n");
}

void
create_transformed_mask(
		unsigned short *d_transformed,
		int tz,
		int w,
		int h,
		int d,
		float zspacing,
		int tw,
		int th,
		float border,
		const float *d_inverse,
		cudaStream_t stream)
{
	dim3 threads(32, 32);
	dim3 grid(iDivUp(tw, threads.x), iDivUp(th, threads.y));
	create_transformed_mask_kernel<<<grid, threads, 0, stream>>>(
			d_transformed, tz, w, h, d, zspacing,
			tw, th, border, d_inverse);
	getLastCudaError("transform_mask_kernel<<<>>> execution failed\n");
}

void transform_cuda(
		unsigned short **h_data,
		int w,
		int h,
		int d,
		int tw,
		int th,
		int td,
		float *h_inverse,
		const char *outfile,
		int createTransformedMasks,
		int border,
		float zspacing,
		const char *maskfile)
{
	cudaArray *d_data = 0;
	unsigned short *d_transformed;
	unsigned short *h_transformed;
	float *d_inverse;

	int plane_size = tw * th * sizeof(unsigned short);

	const cudaExtent volumeSize = make_cudaExtent(w, h, d);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned short>();
	checkCudaErrors(cudaMalloc3DArray(&d_data, &desc, volumeSize));
	checkCudaErrors(cudaMallocHost((void **)&h_transformed, plane_size));

	printf("td = %d\n", td);

	// copy data to 3D array
	for(int z = 0; z < d; z++) {
		cudaMemcpy3DParms copyParams = {0};
		copyParams.dstArray = d_data;
		copyParams.extent   = make_cudaExtent(w, h, 1);
		copyParams.kind     = cudaMemcpyHostToDevice;
		copyParams.srcPtr = make_cudaPitchedPtr((void *)h_data[z],
			       	w * sizeof(unsigned short), w, h);
		copyParams.dstPos = make_cudaPos(0, 0, z);
		checkCudaErrors(cudaMemcpy3D(&copyParams));
	}

	// set texture parameters
	tex.normalized = false;                     // access with unnormalized
	                                            // texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeBorder; // pad with 0
	tex.addressMode[1] = cudaAddressModeBorder;
	tex.addressMode[2] = cudaAddressModeBorder;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex, d_data, desc));

	int nStreams = 2;

	checkCudaErrors(cudaMalloc((void **)&d_transformed,
				nStreams * plane_size));
	checkCudaErrors(cudaMalloc((void **)&d_inverse,
				12 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_inverse, h_inverse,
			       	12 * sizeof(float), cudaMemcpyHostToDevice));

	cudaStream_t *streams = (cudaStream_t *)malloc(
				nStreams * sizeof(cudaStream_t));

	int streamIdx;
	for(streamIdx = 0; streamIdx < nStreams; streamIdx++)
		cudaStreamCreate(&streams[streamIdx]);

	// transform the mask
	if(createTransformedMasks) {
		streamIdx = 0;
		cudaStream_t stream = streams[streamIdx];
		create_transformed_mask(
				d_transformed,
				td / 2,
				w, h, d,
				zspacing,
				tw, th, (float)border,
				d_inverse,
				stream);
		checkCudaErrors(cudaMemcpyAsync(h_transformed, d_transformed,
				plane_size, cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaStreamSynchronize(stream));
		FILE *maskout = fopen(maskfile, "wb");
		fwrite(h_transformed, sizeof(unsigned short), tw * th, maskout);
		fclose(maskout);
	}

	// transform the data
	FILE *out = fopen(outfile, "wb");
	long start = GetTickCount();
	for(int z = 0; z < td; z++) {
		streamIdx = z % nStreams;
		cudaStream_t stream = streams[streamIdx];
		unsigned short* d_trans = d_transformed + streamIdx * tw * th;

		// save the data before overwriting
		if(z >= nStreams) {
			checkCudaErrors(cudaMemcpyAsync(
						h_transformed,
					       	d_trans,
					       	plane_size,
					       	cudaMemcpyDeviceToHost,
					       	stream));
			checkCudaErrors(cudaStreamSynchronize(stream));
			fwrite(h_transformed, sizeof(unsigned short),
					tw * th, out);
		}

		// launch the kernel
		transform_plane(d_trans, z, w, h, d, tw, th, d_inverse, stream);
	}

	for(int z = 0; z < nStreams; z++) {
		streamIdx = (streamIdx + 1) % nStreams;
		cudaStream_t stream = streams[streamIdx];
		unsigned short* d_trans = d_transformed + streamIdx * tw * th;

		// save the remaining planes
		checkCudaErrors(cudaMemcpyAsync(h_transformed, d_trans,
				plane_size, cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaStreamSynchronize(stream));
		fwrite(h_transformed, sizeof(unsigned short), tw * th, out);
	}


	long end = GetTickCount();
	printf("needed %d ms\n", (end - start));
	fclose(out);

	cudaUnbindTexture(tex);
	cudaFreeArray(d_data);
	cudaFree(d_transformed);
	cudaFree(d_inverse);
	cudaFreeHost(h_transformed);
	for(streamIdx = 0; streamIdx < nStreams; streamIdx++)
		checkCudaErrors(cudaStreamDestroy(streams[streamIdx]));
}

//void invert3x3(float *mat)
//{
//	double sub00 = mat[5] * mat[10] - mat[6] * mat[9];
//	double sub01 = mat[4] * mat[10] - mat[6] * mat[8];
//	double sub02 = mat[4] * mat[9] - mat[5] * mat[8];
//	double sub10 = mat[1] * mat[10] - mat[2] * mat[9];
//	double sub11 = mat[0] * mat[10] - mat[2] * mat[8];
//	double sub12 = mat[0] * mat[9] - mat[1] * mat[8];
//	double sub20 = mat[1] * mat[6] - mat[2] * mat[5];
//	double sub21 = mat[0] * mat[6] - mat[2] * mat[4];
//	double sub22 = mat[0] * mat[5] - mat[1] * mat[4];
//	double det = mat[0] * sub00 - mat[1] * sub01 + mat[2] * sub02;
//	
//	mat[0]  =  (float)(sub00 / det);
//	mat[1]  = -(float)(sub10 / det);
//	mat[2]  =  (float)(sub20 / det);
//	mat[4]  = -(float)(sub01 / det);
//	mat[5]  =  (float)(sub11 / det);
//	mat[6]  = -(float)(sub21 / det);
//	mat[8]  =  (float)(sub02 / det);
//	mat[9]  = -(float)(sub12 / det);
//	mat[10] =  (float)(sub22 / det);
//}
//
//void invert(float *mat)
//{
//	float dx = mat[3];
//	float dy = mat[7];
//	float dz = mat[11];
//	invert3x3(mat);
//
//	mat[3]  = mat[0] * dx + mat[1] * dy + mat[2]  * dz;
//	mat[7]  = mat[4] * dx + mat[5] * dy + mat[6]  * dz;
//	mat[11] = mat[8] * dx + mat[9] * dy + mat[10] * dz;
//}
//
//static void read_dimensions(const char *dimfile, int *dims)
//{
//	char buffer[256];
//	char *p;
//	FILE *f = fopen(dimfile, "r");
//	fgets(buffer, 256, f);
//	p = strchr(buffer, ':') + 1;
//	dims[0] = atoi(p);
//	fgets(buffer, 256, f);
//	p = strchr(buffer, ':') + 1;
//	dims[1] = atoi(p);
//	fgets(buffer, 256, f);
//	p = strchr(buffer, ':') + 1;
//	dims[2] = atoi(p);
//
//	fclose(f);
//}
//
//static void read_transformation(const char *regfile, float *matrix)
//{
//	FILE *f = fopen(regfile, "r");
//	char buffer[256];
//	char *p;
//	int i;
//	float zscaling;
//	for(i = 0; i < 12; i++) {
//		fgets(buffer, 256, f);
//		p = strchr(buffer, ':') + 1;
//		matrix[i] = atof(p);
//	}
//	while(fgets(buffer, 256, f) != NULL) {
//		if(!strncmp(buffer, "z-scaling", 9)) {
//			p = strchr(buffer, ':') + 1;
//			zscaling = atof(p);
//		}
//	}
//	fclose(f);
//	
//	matrix[2]  *= zscaling;
//	matrix[6]  *= zscaling;
//	matrix[10] *= zscaling;
//}
//
//void
//apply(const float *mat, float x, float y, float z, float *result)
//{
//	result[0] = mat[0] * x + mat[1] * y + mat[2]  * z + mat[3];
//	result[1] = mat[4] * x + mat[5] * y + mat[6]  * z + mat[7];
//	result[2] = mat[8] * x + mat[9] * y + mat[10] * z + mat[11];
//}
//
//static
//void test_transform(int argc, char **argv)
//{
//	printf("%s\n", argv[0]);
//	const char *infile = "v0.raw";
//	const int w = 1698;
//	const int h = 1410;
//	const int d = 210;
//	const int tw = w;
//	const int th = h;
//	const int td = d;
//	unsigned short **data = (unsigned short **)malloc(d * sizeof(unsigned short *));
//	for(int z = 0; z < d; z++)
//		checkCudaErrors(cudaMallocHost((void**)&data[z], w * h * sizeof(unsigned short)));
//
//	FILE *f = fopen(infile, "rb");
//	fread(data, sizeof(unsigned short), w * h * d, f);
//	fclose(f);
//	float *mat = (float *)malloc(12 * sizeof(float));
//	mat[0] = 1; mat[1] = 0; mat[2]  = 0; mat[3]  = 20;
//	mat[4] = 0; mat[5] = 1; mat[6]  = 0; mat[7]  = 0;
//	mat[8] = 0; mat[9] = 0; mat[10] = 1; mat[11] = 0;
//	invert(mat);
//	transform_cuda(data, w, h, d, tw, th, td, mat, "v0.out.raw", 0, 0, 0, NULL);
//
//	free(mat);
//	for(int z = 0; z < d; z++)
//		cudaFreeHost(data[z]);
//	free(data);
//
//	cudaDeviceReset();
//}
//
