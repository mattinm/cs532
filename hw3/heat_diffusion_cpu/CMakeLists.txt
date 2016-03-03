cmake_minimum_required (VERSION 2.6)
project (DISPLAY_IMAGE_CPU)

add_executable(heat_diffusion_cpu heat_diffusion_cpu.cxx)
target_link_libraries(heat_diffusion_cpu ${CUSTOM_LIBRARIES})
