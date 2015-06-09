#ifndef FMVD_DECONVOLVE
#define FMVD_DECONVOLVE

#include <cufft.h>
#include <stdio.h>
#include "fmvd_deconvolve_cuda.cuh"


enum fmvd_psf_type { independent, efficient_bayesian, optimization_1, optimization_2 };

void *
fmvd_malloc(size_t size);

void
fmvd_free(void *p);

#define SAMPLE              unsigned short
#define BITS_PER_SAMPLE     16 
#include "fmvd_deconvolve.impl.h"

#define SAMPLE              unsigned char
#define BITS_PER_SAMPLE     8
#include "fmvd_deconvolve.impl.h"



#endif // FMVD_DECONVOLVE
