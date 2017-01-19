cmake_minimum_required(VERSION 3.6)
project(CompareAdiosKernels)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(CompareAdiosKernels ${SOURCE_FILES})