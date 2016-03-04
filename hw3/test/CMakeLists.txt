cmake_minimum_required (VERSION 2.6)
project (DISPLAY_IMAGE_TEST)

add_executable(gluttest test.cxx)
target_link_libraries(gluttest ${CUSTOM_LIBRARIES})
