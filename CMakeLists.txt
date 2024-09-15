# CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(snn_cuda_project LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/src)

# Collect source files
set(SOURCE_FILES
    src/main.cpp
    src/kernels.cu
)

# Add executable target
add_executable(snn_cuda ${SOURCE_FILES})

# Enable CUDA separable compilation
set_target_properties(snn_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify CUDA architectures
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set_property(TARGET snn_cuda PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 86)
else()
    # For older CMake versions
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_52")
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Link necessary CUDA libraries
target_link_libraries(snn_cuda
    PRIVATE
    curand
)
# Optionally, add tests
# enable_testing()
# add_executable(test_tensor tests/test_tensor.cpp src/Tensor.cpp)
# target_link_libraries(test_tensor PRIVATE CUDA::cudart)
# add_test(NAME TensorTest COMMAND test_tensor)

