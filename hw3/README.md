# Installation

From the root directory (the same directory as this file),
run the command `cmake .` followed by `cmake --build .`.

This will create the /build folder and populate it with the
binaries for `heat_diffusion_cuda`, `heat_diffusion_cpu`, and
`heat_diffusion_opencl`. As the names imply, each implements
the heat_diffusion 3D alorithm on a different processing node.

## CUDA implementation
CUDA will only work on systems with an NVIDIA graphics card and
the NVIDIA CUDA SDK installed.

If CUDA is not available on the system, a warning will appear
and no binary will be generated for `heat_diffusion_cuda`.

## CPU implementation
Threading is used in `heat_diffusion_cpu` to attempt to maximize
the efficiency of the algorithm. `heat_diffusion_cpu` will a
number of threads equal to the number of hardware threads available
on the system.

If std::threading is not available on the system (C++0x or C++11+),
the CPU implementation will be single threaded instead.

## OpenCL implementation
OpenCL will work on systems for which an appropriate OpenCL SDK
has been installed. The OpenCL SDK used must correspond with the
graphics card (i.e. Intel, NVIDIA, or AMD) installed in the system.

If OpenCL is not available on the system, a warning will appear
and no binary will be generated for `heat_diffusion_opencl`.

# Command-Line Arguments
These are a list of command line arguments that can be used when
running any of the exectuables.

| Argument | Purpose |
|----------|---------|
|--cpu         | (OpenCL Only) Attempt to run on the CPU instead of GPU |
|--sim_size    | x y z size of the simulation |
|--window_size | x y size of the window |
