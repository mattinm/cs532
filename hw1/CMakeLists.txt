cmake_minimum_required (VERSION 2.6)
project (kmeanshw1)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(kmeans)
add_subdirectory(mpi_kmeans)
