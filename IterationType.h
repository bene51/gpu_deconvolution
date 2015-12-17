#ifndef ITERATION_TYPE_H
#define ITERATION_TYPE_H

class IterationType
{
private:
	static void computeInvertedKernel(const float *kernel,
			int kw, int kh, float *out);
	static void computeExponentialKernel(const float *kernel,
			int kw, int kh, int exponent, float *out);
	static void convolveSinglePlane(float *data, int dataW, int tdataH,
			const float *kernel, int kernelW, int kernelH);

public:
	enum Type
	{
		INDEPENDENT,
		EFFICIENT_BAYESIAN,
		OPTIMIZATION_1,
		OPTIMIZATION_2
	};
	static void normalizeKernel(float *kernel, int len);
	static float **createKernelHat(
			const float * const * kernel, int kernelW, int kernelH,
			int nViews, IterationType::Type type);

};

#endif // ITERATION_TYPE_H
