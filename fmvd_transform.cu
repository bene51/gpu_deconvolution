#include "fmvd_transform.h"

__global__ void
create_transformed_mask_kernel(float *, int, int, int, int, float, int, int, float, float *);

void
create_transformed_mask(float *, int, int, int, int, float, int, int, float, const float *, cudaStream_t);



#define SAMPLE              unsigned short
#define BITS_PER_SAMPLE     16 
#include "fmvd_transform.impl.cu"

#define SAMPLE              unsigned char
#define BITS_PER_SAMPLE     8
#include "fmvd_transform.impl.cu"

__global__ void
create_transformed_mask_kernel(
		float *dTransformed,
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

//			dx += 1;
//			dy += 1;
//			dz += 1;
			dz *= zspacing;

			if(dx < border)
				v = v * (0.5f * (1 - cos(dx / border * pi)));
			if(dy < border)
				v = v * (0.5f * (1 - cos(dy / border * pi)));
			if(dz < border)
				v = v * (0.5f * (1 - cos(dz / border * pi)));
			dTransformed[idx] = v;
		}
	}
}

void
create_transformed_mask(
		float *d_transformed,
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
