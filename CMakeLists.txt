cmake_minimum_required(VERSION 3.24)
project(playing_with_h5cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(${OUTPUT_DIRS} KEEP_RPATHS)

find_package(HDF5 REQUIRED)
find_package(h5cpp REQUIRED)

add_executable(playing_with_h5cpp main.cpp)
target_link_libraries(playing_with_h5cpp PRIVATE hdf5::hdf5 h5cpp::h5cpp)