CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
project(cudaAugmentation LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found.")
endif()

add_definitions(-DHAVE_CUDA)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc/)

message("warning: helper headers locate at ")
message(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc/)

list(APPEND cudaAugmentation_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math;-Xptxas -O3;-Xcompiler -O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -fPIC)
set(CUDA_SEPARABLE_COMPILATION ON)

set(SOURCE_FILES
        kernel/test.cu
        kernel/utils.cu
        kernel/scale.cu
        kernel/ops_copy.cu)

cuda_add_library(cudaAugmentation SHARED ${SOURCE_FILES})
target_link_libraries(cudaAugmentation ${cudaBERT_EXTERNAL_LIBRARIES})

