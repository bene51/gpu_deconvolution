#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convolutionFFT2D_common.h"

#include "fftw-3.3.4-dll64/fftw3.h"


static int snapTransformSize(int dataSize)
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

static void padKernelCPU(
		float *d_Dst,
		float *d_DstHat,
		float *d_Src,
		int fftH,
		int fftW,
		int kernelH,
		int kernelW)
{
	int x, y;
	int kernelY = kernelH / 2;
	int kernelX = kernelW / 2;
	for(y = 0; y < kernelH; y++) {
		for(x = 0; x < kernelW; x++) {
			int ky = y - kernelY;
			if (ky < 0)
				ky += fftH;

			int kx = x - kernelX;
			if (kx < 0)
				kx += fftW;

			int idx = ky * fftW + kx;

			d_Dst   [idx] = d_Src[y * kernelW + x];
			d_DstHat[idx] = d_Src[(kernelH - y - 1) * kernelW + (kernelW - x - 1)];
		}
	}
}

static void padDataClampToBorderCPU(
		float *d_estimate,
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		int nViews
		)
{
	int x, y;
	const int borderH = dataH + kernelH / 2;
	const int borderW = dataW + kernelW / 2;

	for(y = 0; y < fftH; y++) {
		for(x = 0; x < fftW; x++) {
			int dy, dx, idx;
			float v;

			if (y < dataH)
				dy = y;
			else if (y >= dataH && y < borderH)
				dy = dataH - 1;
			else if (y >= borderH)
				dy = 0;

			if (x < dataW)
				dx = x;
			else if (x >= dataW && x < borderW)
				dx = dataW - 1;
			else if (x >= borderW)
				dx = 0;

			v = d_Src[dy * dataW + dx];
			idx = y * fftW + x;
			d_Dst[idx] = v;
			d_estimate[idx] += v / nViews;
		}
	}
}

static void unpadDataCPU(
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW
		)
{
	int x, y;
	for(y = 0; y < dataH; y++)
		for(x = 0; x < dataW; x++)
			d_Dst[y * dataW + x] = d_Src[y * fftW + x];
}


static void modulateAndNormalizeCPU(
		fftwf_complex *d_Dst,
		fftwf_complex *d_Src,
		int fftH,
		int fftW,
		int padding
		)
{
	int i;
	const int dataSize = fftH * (fftW / 2 + padding);
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

static void divideCPU(
		float *d_a,
		float *d_b,
		float *d_dest,
		int fftH,
		int fftW
		)
{
	const int dataSize = fftH * fftW;
	int i;


	for(i = 0; i < dataSize; i++)
		d_dest[i] = d_a[i] / d_b[i];
}

static void mulCPU(
		float *d_a,
		float *d_b,
		float *d_dest,
		int fftH,
		int fftW
		)
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

static void prepareKernelsCPU(
	float **h_Kernel,
	int kernelH,
	int kernelW,
	int fftH,
	int fftW,
	int nViews,
	fftwf_complex **h_KernelSpectrum,
	fftwf_complex **h_KernelHatSpectrum,
	fftwf_plan fftPlanFwd
)
{
	int v;

	float *d_PaddedKernel    = (float *)fftwf_malloc(fftH    * fftW    * sizeof(float));
	float *d_PaddedKernelHat = (float *)fftwf_malloc(fftH    * fftW    * sizeof(float));
	for(v = 0; v < nViews; v++) {
		memset(d_PaddedKernel,    0, fftH * fftW * sizeof(float));
		memset(d_PaddedKernelHat, 0, fftH * fftW * sizeof(float));

		padKernelCPU(
			d_PaddedKernel,
			d_PaddedKernelHat,
			h_Kernel[v],
			fftH,
			fftW,
			kernelH,
			kernelW
		);

		fftwf_execute_dft_r2c(fftPlanFwd, d_PaddedKernel,    h_KernelSpectrum[v]);
		fftwf_execute_dft_r2c(fftPlanFwd, d_PaddedKernelHat, h_KernelHatSpectrum[v]);
	}
	fftwf_free(d_PaddedKernel);
	fftwf_free(d_PaddedKernelHat);
}

static void normalizeMinMax(float *data, int len, float min, float max)
{
	int i;
	float *k = data;
	for(i = 0; i < len; i++) {
		float v = *k;
		v = (v - min) / (max - min);

		if(v < 0) v = 0;
		if(v > 1) v = 1;
		(*k) = v;
		k++;
	}
}

void deconvolveCPU(FILE **dataFiles, FILE *resultFile, int dataW, int dataH, int dataD, float **h_Kernel, int kernelH, int kernelW, int nViews, int iterations)
{
	long start, stop;
	int v, z, it;
	float *h_ResultGPU;
	
	fftwf_init_threads();
	fftwf_plan_with_nthreads(16);

	fftwf_plan fftPlanFwd, fftPlanInv;

	const int fftH = snapTransformSize(dataH + kernelH - 1);
	const int fftW = snapTransformSize(dataW + kernelW - 1);

	float	**h_Data                    =         (float **)malloc(nViews * sizeof(float *));
	float	**h_PaddedData              =         (float **)malloc(nViews * sizeof(float *));
	fftwf_complex **h_KernelSpectrum    = (fftwf_complex **)malloc(nViews * sizeof(fftwf_complex *));
	fftwf_complex **h_KernelHatSpectrum = (fftwf_complex **)malloc(nViews * sizeof(fftwf_complex *));

	float *h_estimate, *h_tmp;
	fftwf_complex *h_estimateSpectrum;

	h_estimate         =         (float *)fftwf_malloc(fftH *           fftW * sizeof(float));
	h_tmp              =         (float *)fftwf_malloc(fftH *           fftW * sizeof(float));
	h_estimateSpectrum = (fftwf_complex *)fftwf_malloc(fftH * (fftW / 2 + 1) * sizeof(fftwf_complex));

	for(v = 0; v < nViews; v++) {
		h_Data[v]              =         (float *)fftwf_malloc(dataW *          dataH * dataD * sizeof(float));
		h_PaddedData[v]        =         (float *)fftwf_malloc(fftH  *           fftW * sizeof(float));
		h_KernelSpectrum[v]    = (fftwf_complex *)fftwf_malloc(fftH  * (fftW / 2 + 1) * sizeof(fftwf_complex));
		h_KernelHatSpectrum[v] = (fftwf_complex *)fftwf_malloc(fftH  * (fftW / 2 + 1) * sizeof(fftwf_complex));
	}


	fftPlanFwd = fftwf_plan_dft_r2c_2d(fftH, fftW, h_estimate, h_estimateSpectrum, FFTW_ESTIMATE);
	fftPlanInv = fftwf_plan_dft_c2r_2d(fftH, fftW, h_estimateSpectrum, h_estimate, FFTW_ESTIMATE);


	prepareKernelsCPU(h_Kernel, kernelH, kernelW, fftH, fftW, nViews, h_KernelSpectrum, h_KernelHatSpectrum, fftPlanFwd);

	// load from HDD
	for(v = 0; v < nViews; v++) {
		fread(h_Data[v], sizeof(float), dataW * dataH * dataD, dataFiles[v]);
		normalizeMinMax(h_Data[v], dataW * dataH * dataD, 40, 62000);
	}
	start = GetTickCount();

	for(z = 0; z < dataD; z++) {

		// H2D
		memset(h_estimate, 0, fftH * fftW * sizeof(float));
		for(v = 0; v < nViews; v++) {
			memset(h_PaddedData[v], 0, fftH * fftW * sizeof(float));
			padDataClampToBorderCPU(h_estimate, h_PaddedData[v], h_Data[v], fftH, fftW, dataH, dataW, kernelH, kernelW, nViews);
		}

		for(it = 0; it < iterations; it++) {
			for(v = 0; v < nViews; v++) {
				fftwf_execute_dft_r2c(fftPlanFwd, h_estimate, h_estimateSpectrum);
				modulateAndNormalizeCPU(h_estimateSpectrum, h_KernelSpectrum[v], fftH, fftW, 1);
				fftwf_execute_dft_c2r(fftPlanInv, h_estimateSpectrum, h_tmp);
				divideCPU(h_PaddedData[v], h_tmp, h_tmp, fftH, fftW);
		
				fftwf_execute_dft_r2c(fftPlanFwd, h_tmp, h_estimateSpectrum);
				modulateAndNormalizeCPU(h_estimateSpectrum, h_KernelHatSpectrum[v], fftH, fftW, 1);
				fftwf_execute_dft_c2r(fftPlanInv, h_estimateSpectrum, h_tmp);
				mulCPU(h_estimate, h_tmp, h_estimate, fftW, fftH);
			}
		}
		unpadDataCPU(h_Data[0], h_estimate, fftH, fftW, dataH, dataW);

		for(v = 0; v < nViews; v++)
			h_Data[v] += (dataW * dataH);

	}

	stop = GetTickCount();
	printf("Overall time: %d ms\n", (stop - start));

	for(v = 0; v < nViews; v++)
		h_Data[v] -= (dataW * dataH * dataD);
	fwrite(h_Data[0] , sizeof(float), dataW * dataH * dataD, resultFile);


	fftwf_destroy_plan(fftPlanFwd);
	fftwf_destroy_plan(fftPlanInv);

	for(v = 0; v < nViews; v++) {
		fftwf_free(h_Data[v]);
		fftwf_free(h_PaddedData[v]);
		fftwf_free(h_KernelSpectrum[v]);
		fftwf_free(h_KernelHatSpectrum[v]);
	}
	fftwf_free(h_estimate);
	fftwf_free(h_estimateSpectrum);
	fftwf_free(h_tmp);

	free(h_Data);
	free(h_PaddedData);
	free(h_KernelSpectrum);
	free(h_KernelHatSpectrum);

	printf("...shutting down\n");

}

void load(int w, int h, float *data, const char *path)
{
	FILE *f = fopen(path, "rb");
	fread(data, sizeof(float), w * h, f);
	fclose(f);
}

void normalize(float *kernel, int len)
{
	int i;
	double sum = 0;
	float *k = kernel;
	for(i = 0; i < len; i++) {
		sum += *k;
		k++;
	}
	k = kernel;
	for(i = 0; i < len; i++) {
		*k /= sum;
		k++;
	}
}

void normalizeRange(float *kernel, int len)
{
	int i;
	float *k = kernel;
	float min = kernel[0];
	float max = kernel[0];
	k++;
	for(i = 1; i < len; i++) {
		float v = *k;
		if(v < min) min = v;
		if(v > max) max = v;
		k++;
	}
	k = kernel;
	for(i = 0; i < len; i++) {
		float v = *k;
		v = (v - min) / (max - min);
		(*k) = v;
		k++;
	}
}

int main(int argc, char **argv)
{
	int it;
	printf("[%s] - Starting...\n", argv[0]);

	const int W = 600;
	const int H = 600;
	const int D = 100;

	const int KW = 83;
	const int KH = 83;

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

		sprintf(path, "E:\\SPIM5_Deconvolution\\m6\\cropped\\forplanewisedeconv\\v35.deconvolved.gpu.raw");
		FILE *resultFile = fopen(path, "wb");

		deconvolveCPU(dataFiles, resultFile, W, H, D, kernel, KH, KW, 2, it);

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

