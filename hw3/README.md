# Installation

From the root directory (the same directory as this file),
run the command `cmake .` followed by `cmake --build .`.

This will create the /build folder and populate it with the
binaries for `heat_diffusion_cuda`, `heat_diffusion_cpu`, and
`heat_diffusion_opencl`. As the names imply, each implements
the heat_diffusion 3D alorithm on a different processing node.

If CUDA is not available on the system, a warning will appear
and no binary will be generated for `heat_diffusion_cuda`.
