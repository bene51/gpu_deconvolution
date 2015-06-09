#define PASTE(name, type) name ## _ ## type
#define EVAL(name, type) PASTE(name, type)
#define MAKE_NAME(name) EVAL(name, BITS_PER_SAMPLE)


extern "C" void MAKE_NAME(padDataClampToBorderAndInitialize)(
		float *d_estimate,
		SAMPLE *d_PaddedData,
		SAMPLE *d_Data,
		float *d_Weights,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		cudaStream_t stream
		);

extern "C" void MAKE_NAME(unpadData)(
		SAMPLE *d_Data,
		float *d_PaddedData,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		cudaStream_t stream
		);

extern "C" void MAKE_NAME(divide)(
		SAMPLE *d_a,
		float *d_b,
		float *d_dest,
		int fftH,
		int fftW,
		cudaStream_t stream
		);


