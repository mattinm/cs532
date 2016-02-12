cmake_minimum_required (VERSION 2.6)
project (mpi_kmeans C)

# must find MPI
find_package (MPI REQUIRED)

SET (CMAKE_C_COMPILER             ${MPI_C_COMPILER})
SET (CMAKE_C_FLAGS                "-Wall -std=c99")
SET (CMAKE_C_FLAGS_DEBUG          "-g")
SET (CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")
SET (CMAKE_C_COMPILE_FLAGS ${CMAKE_C_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS})
SET (CMAKE_C_LINK_FLAGS ${CMAKE_C_LINK_FLAGS} ${MPI_C_LINK_FLAGS})

include_directories(${MPI_INCLUDE_C_PATH})
add_executable(mpi_kmeans mpi_kmeans.c)
target_link_libraries(mpi_kmeans m ${MPI_LIBRARIES})
