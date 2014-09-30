nvcc -O2 --cl-version 2010 --use-local-env -lcufft --include-path ..\..\common\inc main.cpp convolutionFFT2D.cu
