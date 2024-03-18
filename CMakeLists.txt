cmake_minimum_required(VERSION 3.24)
project(playing_with_h5cpp)

set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(HDF5 REQUIRED)
find_package(h5cpp REQUIRED)

add_executable(playing_with_h5cpp main.cpp)
target_link_libraries(playing_with_h5cpp hdf5::hdf5 h5cpp::h5cpp)