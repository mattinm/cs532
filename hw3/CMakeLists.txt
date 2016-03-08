cmake_minimum_required (VERSION 2.6)
project (DISPLAY_IMAGE)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)

include (CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-std=c++11 COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG(-std=c++0x COMPILER_SUPPORTS_CXX0x)

# Win32 needs to have the Win32\GLUT folder specified
if (WIN32)
	set(GLUT_ROOT_PATH ${CMAKE_SOURCE_DIR}/windows/GLUT)
	message(STATUS "WIN32 detected. Using ${GLUT_ROOT_PATH} for GLUT.")
endif(WIN32)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
set(CUSTOM_LIBRARIES ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

add_subdirectory(heat_diffusion_cuda)
add_subdirectory(heat_diffusion_cpu)
