cmake_minimum_required (VERSION 2.6)
project (DISPLAY_IMAGE)

#set(CUDA_HOST_COMPILER /usr/bin/clang++)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

find_package(CUDA)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

#find_package(OpenCV COMPONENTS features2d nonfree flann imgproc highgui legacy core)

#include_directories(${OpenCV_INCLUDE_DIR})
#include_directories(${OpenCV_INCLUDE_DIR}/../)

#message(STATUS "opencv libraries: ${OpenCV_LIBRARIES}")

cuda_add_executable(seamcarving seamcarving.cu lodepng)
target_link_libraries(seamcarving ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(col_removal col_removal.cu lodepng)
target_link_libraries(col_removal ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

