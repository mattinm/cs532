# Installation

From the root directory (the same directory as this file),
run the command `cmake .` followed by `cmake --build .`.

This will create the /build folder and populate it with the
binaries for both `kmeans` and `mpi_kmeans`. As the names
imply, `kmeans` is a single-node version of the k-means algorithm,
while `mpi_kmeans` is the distributed version.

By default, no output files are created and some debug prints are
omitted. See the [Optional Arguments] section for more information
about changing this behavior.

## Optional Arguments

The arguments can be implemented at the command line with `-DFLAG=1`.
The purpose of each argument is listed below.

Since CMake caches arguments, delete the `CMakeCache.txt` file
between compilations if arguments are being passed.

| Flag         | Purpose |
| ------------ | ------- |
|-DOUTPUTFILES | Outputs the respective cluster files |
|-DNOPRINT     | Omits some debug prints |
