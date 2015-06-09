#define PASTE(name, type) name ## _ ## type
#define EVAL(name, type) PASTE(name, type)
#define MAKE_NAME(name) EVAL(name, BITS_PER_SAMPLE)

// returns 0 if no more data is available
typedef int (*MAKE_NAME(datasource_t))(SAMPLE **buffer, int offset);

typedef void (*MAKE_NAME(datasink_t))(SAMPLE *buffer);

struct MAKE_NAME(fmvd_plan_cuda) {
	int dataH, dataW;
	int fftH, fftW;
	int kernelH, kernelW;
	int nViews;
	int nStreams;

	fComplex **d_KernelSpectrum;
	fComplex **d_KernelHatSpectrum;

	SAMPLE **h_Data;
	SAMPLE **d_Data;
	SAMPLE **d_PaddedData;
	float **d_PaddedWeights;

	float *d_estimate;
	fComplex *d_estimateSpectrum;

	float *d_tmp;

	cudaStream_t *streams;

	cufftHandle *fftPlanFwd, *fftPlanInv;

	MAKE_NAME(datasource_t) get_next_plane;
	MAKE_NAME(datasink_t) return_next_plane;

};

struct MAKE_NAME(fmvd_plan_cuda) *
MAKE_NAME(fmvd_initialize_cuda)(
	int dataH, int dataW,
	float const* const* h_Weights,
	float **h_Kernel, int kernelH, int kernelW,
	fmvd_psf_type iteration_type,
	int nViews,
	int nstreams,
	MAKE_NAME(datasource_t) get_next_plane,
	MAKE_NAME(datasink_t) return_next_plane);


void
MAKE_NAME(fmvd_deconvolve_planes_cuda)(const struct MAKE_NAME(fmvd_plan_cuda) *plan, int iterations);

void
MAKE_NAME(fmvd_destroy_cuda)(struct MAKE_NAME(fmvd_plan_cuda) *plan);

void
MAKE_NAME(fmvd_deconvolve_files_cuda)(
	FILE **dataFiles,
	FILE *resultFile,
	int dataW, int dataH, int dataD,
	float **h_Weights,
	float **h_Kernel, int kernelH, int kernelW,
	fmvd_psf_type iteration_type,
	int nViews,
	int iterations);

