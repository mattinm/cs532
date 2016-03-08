cmake_minimum_required (VERSION 2.6)
project (DISPLAY_IMAGE_CPU)

CHECK_CXX_COMPILER_FLAG("-pthread" COMPILER_PTHREAD)
if (COMPILER_PTHREAD)
    #set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pthread")
endif()

if (COMPILER_SUPPORTS_CXX11 OR COMPILER_SUPPORTS_CXX0x)
    find_package(Threads REQUIRED)
    set(CUSTOM_LIBRARIES ${CUSTOM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()

add_executable(heat_diffusion_cpu heat_diffusion_cpu.cxx)
target_link_libraries(heat_diffusion_cpu ${CUSTOM_LIBRARIES})
