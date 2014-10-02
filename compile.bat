nvcc -O2 --cl-version 2010 --use-local-env -Xcompiler /EHsc,/Zi,/MT -L fftw-3.3.4-dll64 -lcufft -llibfftw3f-3 main.cpp convolutionFFT2D.cu
