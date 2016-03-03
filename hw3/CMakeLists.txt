cmake_minimum_required (VERSION 2.6)
project (DISPLAY_IMAGE)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
set(CUSTOM_LIBRARIES ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(heat_diffusion_cuda)
add_subdirectory(heat_diffusion_cpu)
