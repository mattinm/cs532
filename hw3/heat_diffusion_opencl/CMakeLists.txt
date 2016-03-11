cmake_minimum_required (VERSION 2.8)
project (DISPLAY_IMAGE_OPENCL)

find_package(OpenCL)

if (OPENCL_FOUND)
    include_directories(${OPENCL_INCLUDE_DIRS})
    add_executable(heat_diffusion_opencl heat_diffusion_opencl.cxx)
    target_link_libraries(heat_diffusion_opencl ${CUSTOM_LIBRARIES} ${OPENCL_LIBRARIES})
else (OPENCL_FOUND)
    message(WARNING "OpenCL not found on the system. Skipping OpenCL build.")
endif (OPENCL_FOUND)
