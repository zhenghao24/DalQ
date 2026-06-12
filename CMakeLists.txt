cmake_minimum_required(VERSION 3.10)
project(IVF_DALQ)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(MARCH_FLAG "-march=native")
set(AVX512_FLAGS "-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -ffast-math")
set(EXTRA_OPT_FLAGS "-flto -funroll-all-loops -fprefetch-loop-arrays -fno-math-errno -ftree-vectorize")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast ${MARCH_FLAG} ${AVX512_FLAGS} ${EXTRA_OPT_FLAGS}")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP support: ENABLED (${OpenMP_CXX_VERSION})")
else()
    message(WARNING "OpenMP not found - multithreading disabled!")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

set(UTILS_SOURCES
    src/vector_utils.cpp
    src/fvecs_reader.cpp
)

set(DALQ_SOURCES
    src/dalq.cpp
    src/ivf_dalq_index.cpp
)

set(COMMON_SOURCES
    ${UTILS_SOURCES}
    ${DALQ_SOURCES}
)

set(ALL_LIBRARIES
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    list(APPEND ALL_LIBRARIES OpenMP::OpenMP_CXX)
endif()

list(APPEND ALL_LIBRARIES pthread dl)

add_executable(create_index
    src/create_index.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(create_index ${ALL_LIBRARIES})

add_executable(test_qps
    src/test_qps.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(test_qps ${ALL_LIBRARIES})

add_executable(test_mse
    src/test_mse.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(test_mse ${ALL_LIBRARIES})



