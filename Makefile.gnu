NVCC = nvcc
INC = --include-path "c:/Program Files/Java/jdk1.7.0_51/include" --include-path "c:/Program Files/Java/jdk1.7.0_51/include/win32"
NVCC_FLAGS = -c -O2 -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT" --cl-version 2012

all: NativeSPIMReconstructionCuda.dll

CPP_SOURCES = fastspim_NativeSPIMReconstructionCuda.cpp fmvd_deconvolve_cuda.cpp fmvd_deconvolve_cuda_hdd.cpp

CU_SOURCES = fmvd_transform_cuda.cu convolutionFFT2D.cu fmvd_deconvolve_common.cu

CPP_OBJS = fastspim_NativeSPIMReconstructionCuda.obj fmvd_deconvolve_cuda.obj fmvd_deconvolve_cuda_hdd.obj
CU_SOURCES = fmvd_transform_cuda.cu_obj convolutionFFT2D.cu_obj fmvd_deconvolve_common.cu_obj

REBUILDABLES = $(CPP_OBJS) $(CU_OBJS) NativeSPIMReconstructionCuda.dll

clean :
	rm -f $(REBUILDABLES)

NativeSPIMReconstructionCuda.dll : $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) --shared $** -lcufft -o $@

.cpp.obj :
	@echo Building $@ from $<
	$(NVCC) $(NVCC_FLAGS) $(INC) -o $@ $<

%.cu_obj : %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<


