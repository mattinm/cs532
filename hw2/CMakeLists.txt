cmake_minimum_required (VERSION 2.6)
project (genome)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# must find MPI
find_package (MPI REQUIRED)

include (CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-std=c++11 COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG(-std=c++0x COMPILER_SUPPORTS_CXX0x)

#SET (CMAKE_CXX_COMPILER             ${MPI_CXX_COMPILER})
SET (CMAKE_CXX_FLAGS                "-Wall")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
SET (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

# add C++11 support
if (COMPILER_SUPPORTS_CXX11)
    set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (COMPILER_SUPPORTS_CXX0x)
    set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message (STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.
    Please update the compiler and try again.")
endif()

include_directories(${MPI_INCLUDE_PATH})
add_executable(genome genome.cxx)
target_link_libraries(genome ${MPI_LIBRARIES})
