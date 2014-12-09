Requirements
============

* Cuda toolkit version 6.5, maybe 5.5 is sufficient, but we did not test it.
  Please make sure you have set it up correctly, following the instructions
  from NVidia.

* A compatible compiler. On Windows, that's the Microsoft compiler, on Mac OS X
  and Linux, this is gcc. More details on versions can be found at
  https://nvidia.com.

* For the Java wrappers, the Java Development Kit (JDK) and the environment
  variable JAVA_HOME pointing the folder it's installed.

Build instructions
==================

To simplify building, we provide Makefiles for all major operating systems.
More details below.

Windows
-------

* It should be possible to use any version of Visual Studio or just the
  Windows SDK alone, but you'll need to edit the Makefile (Makefile.win) and
  adjust CL_VERSION appropriately.
  Windows 7 SDK:    CL_VERSION = 2010
  Visual Studio 10: CL_VERSION = 2010
  Visual Studio 11: CL_VERSION = 2012
  Visual Studio 12: CL_VERSION = 2013

  We were using the Windows 7 SDK.

* If you want to compile the DLL needed for the Fiji plugin, just type:
  nmake -f Makefile.win

* If you want to use the C library from within your own project, just
  adjust the Makefile accordingly, it's very basic.

* Also check
  http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows


Linux
-----

* If you want to compile the shared library needed for the Fiji plugin,
  just type:
  make -f Makefile.linux

* If you want to use the C library from within your own project, just
  adjust the Makefile accordingly, it's very basic.

* Also check
  http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux


Mac OS X
--------

* If you want to compile the shared library needed for the Fiji plugin,
  just type:
  make -f Makefile.macosx

* If you want to use the C library from within your own project, just
  adjust the Makefile accordingly, it's very basic.

* Also check
  http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x


Using the library
=================

Basic usage of the library is demonstrated in the file that wraps the call
from Java, namely fastspim_NativeSPIMReconstructionCuda.cpp

