#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fftw-3.3.4-dll64/fftw3.h"
#include "fmvd_deconvolve_fftw.h"
#include "fmvd_deconvolve_common.h"


static void
padKernelCPU(const struct fmvd_plan *plan, const float *h_Kernel, float *d_PaddedKernel, float *d_PaddedKernelHat)
{
	int x, y;
	int kernelH = plan->kernelH;
	int kernelW = plan->kernelW;
	int kernelY = kernelH / 2;
	int kernelX = kernelW / 2;
	int fftH = plan->fftH;
	int fftW = plan->fftW;
	for(y = 0; y < kernelH; y++) {
		for(x = 0; x < kernelW; x++) {
			int ky = y - kernelY;
			if (ky < 0)
				ky += fftH;

			int kx = x - kernelX;
			if (kx < 0)
				kx += fftW;

			int idx = ky * plan->fftW + kx;

			d_PaddedKernel   [idx] = h_Kernel[y * kernelW + x];
			d_PaddedKernelHat[idx] = h_Kernel[(kernelH - y - 1) * kernelW + (kernelW - x - 1)];
		}
	}
}

static void
padDataClampToBorderCPU(const struct fmvd_plan *plan, const float *h_data, int v)
{
	int x, y;
	const int borderH = plan->dataH + plan->kernelH / 2;
	const int borderW = plan->dataW + plan->kernelW / 2;

	for(y = 0; y < plan->fftH; y++) {
		for(x = 0; x < plan->fftW; x++) {
			int dy, dx, idx;
			float value;

			if (y < plan->dataH)
				dy = y;
			else if (y >= plan->dataH && y < borderH)
				dy = plan->dataH - 1;
			else if (y >= borderH)
				dy = 0;

			if (x < plan->dataW)
				dx = x;
			else if (x >= plan->dataW && x < borderW)
				dx = plan->dataW - 1;
			else if (x >= borderW)
				dx = 0;

			value = h_data[dy * plan->dataW + dx];
			idx = y * plan->fftW + x;
			plan->h_PaddedData[v][idx] = value;
			plan->h_estimate[idx] += value / plan->nViews;
		}
	}
}

static void
unpadDataCPU(const struct fmvd_plan *plan, float *h_Data)
{
	int x, y;
	for(y = 0; y < plan->dataH; y++)
		for(x = 0; x < plan->dataW; x++)
			h_Data[y * plan->dataW + x] = plan->h_estimate[y * plan->fftW + x];

}

static void
modulateAndNormalizeCPU(fftwf_complex *d_Dst, fftwf_complex *d_Src, int fftH, int fftW)
{
	int i;
	const int dataSize = fftH * (fftW / 2 + 1);
	float c = 1.0f / (float)(fftW * fftH);

	for(i = 0; i < dataSize; i++) {
		float a_re = d_Src[i][0];
		float a_im = d_Src[i][1];
		float b_re = d_Dst[i][0];
		float b_im = d_Dst[i][1];

		d_Dst[i][0] = c * (a_re * b_re - a_im * b_im);
		d_Dst[i][1] = c * (a_im * b_re + a_re * b_im);
	}
}

static void
divideCPU(float *d_a, float *d_b, float *d_dest, int fftH, int fftW)
{
	const int dataSize = fftH * fftW;
	int i;


	for(i = 0; i < dataSize; i++)
		d_dest[i] = d_a[i] / d_b[i];
}

static void
mulCPU(float *d_a, float *d_b, float *d_dest, int fftH, int fftW)
{
	const int dataSize = fftH * fftW;
	int i;

	for(i = 0; i < dataSize; i++) {
		// d_dest[i] = d_a[i] * d_b[i];
		float target = d_a[i] * d_b[i];
		float change = target - d_dest[i];
		change *= 0.5;
		d_dest[i] += change;
	}
}

static void
prepareKernelsCPU(const struct fmvd_plan *plan, float const* const* h_Kernel)
{
	int v, paddedsize;

	paddedsize = plan->fftH * plan->fftW * sizeof(float);

	float *d_PaddedKernel    = (float *)fftwf_malloc(paddedsize);
	float *d_PaddedKernelHat = (float *)fftwf_malloc(paddedsize);
	for(v = 0; v < plan->nViews; v++) {
		memset(d_PaddedKernel,    0, paddedsize);
		memset(d_PaddedKernelHat, 0, paddedsize);

		padKernelCPU(plan, h_Kernel[v], d_PaddedKernel, d_PaddedKernelHat);

		fftwf_execute_dft_r2c(plan->fftPlanFwd, d_PaddedKernel,    plan->h_KernelSpectrum[v]);
		fftwf_execute_dft_r2c(plan->fftPlanFwd, d_PaddedKernelHat, plan->h_KernelHatSpectrum[v]);
	}
	fftwf_free(d_PaddedKernel);
	fftwf_free(d_PaddedKernelHat);
}

struct fmvd_plan *
fmvd_initializeCPU(int dataH, int dataW, float const* const* h_Kernel, int kernelH, int kernelW, int nViews, int nthreads)
{
	int v, fftH, fftW, fftsize, paddedsize;
	struct fmvd_plan *plan;
	
	fftwf_init_threads();
	fftwf_plan_with_nthreads(nthreads);

	fftH = snapTransformSize(dataH + kernelH - 1);
	fftW = snapTransformSize(dataW + kernelW - 1);
	fftsize = fftH * (fftW / 2 + 1) * sizeof(fftwf_complex);
	paddedsize = fftH * fftW * sizeof(float);

	plan = (struct fmvd_plan *)malloc(sizeof(struct fmvd_plan));
	plan->dataH   = dataH;
	plan->dataW   = dataW;
	plan->fftH    = fftH;
	plan->fftW    = fftW;
	plan->kernelH = kernelH;
	plan->kernelW = kernelW;
	plan->nViews  = nViews;

	plan->h_KernelSpectrum    = (fftwf_complex **)malloc(nViews * sizeof(fftwf_complex *));
	plan->h_KernelHatSpectrum = (fftwf_complex **)malloc(nViews * sizeof(fftwf_complex *));
	plan->h_PaddedData        =         (float **)malloc(nViews * sizeof(float *));

	for(v = 0; v < nViews; v++) {
		plan->h_PaddedData[v]        =         (float *)fftwf_malloc(paddedsize);
		plan->h_KernelSpectrum[v]    = (fftwf_complex *)fftwf_malloc(fftsize);
		plan->h_KernelHatSpectrum[v] = (fftwf_complex *)fftwf_malloc(fftsize);
	}

	plan->h_estimate          =         (float *)fftwf_malloc(paddedsize);
	plan->h_estimateSpectrum  = (fftwf_complex *)fftwf_malloc(fftsize);
	plan->h_tmp               =         (float *)fftwf_malloc(paddedsize);

	plan->fftPlanFwd = fftwf_plan_dft_r2c_2d(fftH, fftW, plan->h_estimate, plan->h_estimateSpectrum, FFTW_ESTIMATE);
	plan->fftPlanInv = fftwf_plan_dft_c2r_2d(fftH, fftW, plan->h_estimateSpectrum, plan->h_estimate, FFTW_ESTIMATE);

	prepareKernelsCPU(plan, h_Kernel);
	return plan;
}

void
fmvd_deconvolvePlaneCPU(const struct fmvd_plan *plan, float **h_Data, int iterations)
{
	int v, it, fftH, fftW;

	fftH = plan->fftH;
	fftW = plan->fftW;

	memset(plan->h_estimate, 0, fftH * fftW * sizeof(float));
	for(v = 0; v < plan->nViews; v++) {
		memset(plan->h_PaddedData[v], 0, fftH * fftW * sizeof(float));
		padDataClampToBorderCPU(plan, h_Data[v], v);
	}

	for(it = 0; it < iterations; it++) {
		for(v = 0; v < plan->nViews; v++) {
			fftwf_execute_dft_r2c(plan->fftPlanFwd, plan->h_estimate, plan->h_estimateSpectrum);
			modulateAndNormalizeCPU(plan->h_estimateSpectrum, plan->h_KernelSpectrum[v], fftH, fftW);
			fftwf_execute_dft_c2r(plan->fftPlanInv, plan->h_estimateSpectrum, plan->h_tmp);
			divideCPU(plan->h_PaddedData[v], plan->h_tmp, plan->h_tmp, fftH, fftW);
	
			fftwf_execute_dft_r2c(plan->fftPlanFwd, plan->h_tmp, plan->h_estimateSpectrum);
			modulateAndNormalizeCPU(plan->h_estimateSpectrum, plan->h_KernelHatSpectrum[v], fftH, fftW);
			fftwf_execute_dft_c2r(plan->fftPlanInv, plan->h_estimateSpectrum, plan->h_tmp);
			mulCPU(plan->h_estimate, plan->h_tmp, plan->h_estimate, fftW, fftH);
		}
	}
	unpadDataCPU(plan, h_Data[0]);
}

void
fmvd_destroy(struct fmvd_plan *plan)
{
	int v;

	fftwf_destroy_plan(plan->fftPlanFwd);
	fftwf_destroy_plan(plan->fftPlanInv);

	fftwf_free(plan->h_estimate);
	fftwf_free(plan->h_estimateSpectrum);
	fftwf_free(plan->h_tmp);

	for(v = 0; v < plan->nViews; v++) {
		fftwf_free(plan->h_PaddedData[v]);
		fftwf_free(plan->h_KernelSpectrum[v]);
		fftwf_free(plan->h_KernelHatSpectrum[v]);
	}

	free(plan->h_KernelSpectrum);
	free(plan->h_KernelHatSpectrum);
	free(plan->h_PaddedData);

	free(plan);
}

void
fmvd_deconvolveCPU(
		FILE **dataFiles,
		FILE *resultFile,
		int dataW,
		int dataH,
		int dataD,
		float const* const* h_Kernel,
		int kernelH,
		int kernelW,
		int nViews,
		int iterations,
		int nthreads)
{
	long start, stop;
	int v, z;

	struct fmvd_plan *plan = fmvd_initializeCPU(dataH, dataW, h_Kernel, kernelH, kernelW, nViews, nthreads);
	
	float **h_Data = (float **)malloc(nViews * sizeof(float *));
	for(v = 0; v < nViews; v++) {
		h_Data[v] = (float *)fftwf_malloc(dataW * dataH * dataD * sizeof(float));
		fread(h_Data[v], sizeof(float), dataW * dataH * dataD, dataFiles[v]);
		// normalizeMinMax(h_Data[v], dataW * dataH * dataD, 40, 62000);
	}

	start = GetTickCount();
	for(z = 0; z < dataD; z++) {
		fmvd_deconvolvePlaneCPU(plan, h_Data, iterations);
		for(v = 0; v < nViews; v++)
			h_Data[v] += (dataW * dataH);
	}
	stop = GetTickCount();
	printf("Overall time: %d ms\n", (stop - start));

	for(v = 0; v < nViews; v++)
		h_Data[v] -= (dataW * dataH * dataD);
	fwrite(h_Data[0] , sizeof(float), dataW * dataH * dataD, resultFile);


	fmvd_destroy(plan);
	for(v = 0; v < nViews; v++)
		fftwf_free(h_Data[v]);
	free(h_Data);

	printf("...shutting down\n");
}

static void
load(int w, int h, float *data, const char *path)
{
	FILE *f = fopen(path, "rb");
	fread(data, sizeof(float), w * h, f);
	fclose(f);
}

int
main(int argc, char **argv)
{
	int it;
	printf("[%s] - Starting...\n", argv[0]);

	const int W = 600;
	const int H = 600;
	const int D = 100;

	const int KW = 83;
	const int KH = 83;

	const int nthreads = 16;

	float **kernel = (float **)malloc(2 * sizeof(float *));
	kernel[0] = (float *)malloc(KW * KH * sizeof(float));
	kernel[1] = (float *)malloc(KW * KH * sizeof(float));

	load(KW, KH, kernel[0], "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\psf3.raw");
	load(KW, KH, kernel[1], "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\psf5.raw");

	normalize(kernel[0], KW * KH);
	normalize(kernel[1], KW * KH);

	FILE **dataFiles = (FILE**)malloc(2 * sizeof(FILE *));
	char path[256];
	for(it = 3; it < 4; it++) {
		dataFiles[0] = fopen("E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v3.raw", "rb");
		dataFiles[1] = fopen("E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v5.raw", "rb");
		fseek(dataFiles[0], 300 * W * H * sizeof(float), SEEK_SET);
		fseek(dataFiles[1], 300 * W * H * sizeof(float), SEEK_SET);

		sprintf(path, "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v35.deconvolved.gpu.raw");
		FILE *resultFile = fopen(path, "wb");

		fmvd_deconvolveCPU(dataFiles, resultFile, W, H, D, kernel, KH, KW, 2, it, nthreads);

		fclose(dataFiles[0]);
		fclose(dataFiles[1]);
		fclose(resultFile);
	}

	free(kernel[0]);
	free(kernel[1]);
	free(kernel);
	free(dataFiles);


	exit(EXIT_SUCCESS);
}

