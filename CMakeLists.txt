cmake_minimum_required(VERSION 3.6)
project(CompareAdiosKernels)

###############################################################################

set(CMAKE_CXX_STANDARD 11)
find_package(Boost REQUIRED COMPONENTS mpi serialization system filesystem)
find_package(MPI REQUIRED)

###############################################################################
# ADIOS
###############################################################################
find_program(ADIOS_CONFIG_EXECUTABLE NAMES adios_config)
execute_process(COMMAND ${ADIOS_CONFIG_EXECUTABLE} -d OUTPUT_VARIABLE
        adios_root OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND ${ADIOS_CONFIG_EXECUTABLE} -c OUTPUT_VARIABLE
#        adios_inc OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${ADIOS_CONFIG_EXECUTABLE} -l OUTPUT_VARIABLE
        adios_lib OUTPUT_STRIP_TRAILING_WHITESPACE)

###############################################################################

set(SOURCE_FILES main.cpp)
set(CMAKE_INCLUDE_CURRENT_DIR on)

include_directories(${BOOST_INCLUDE_DIRS}
                    ${MPI_INCLUDE_PATH}
                    ${adios_root}/include)

add_executable(CompareAdiosKernels ${SOURCE_FILES})

target_link_libraries(CompareAdiosKernels
           ${MPI_CXX_LIBRARIES}
           ${Boost_LIBRARIES}
           ${adios_lib})