cmake_minimum_required(VERSION 3.24)
project(nucleobyte LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "61;61-virtual") 

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/venv/Scripts/python.exe" -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(pybind11_DIR "${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/pybind11")

list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
find_package(pybind11 REQUIRED)

set(SRC_FILES
    src/main.cpp
    src/tokenizer/tokenizer_kernel.cu
    src/attention/sparse_attn.cu
)

pybind11_add_module(_core MODULE ${SRC_FILES})

target_compile_features(_core PRIVATE cxx_std_20)
set_target_properties(_core PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_LINK_WITH_NVCC ON
    PREFIX ""
    SUFFIX ".pyd"
)

target_include_directories(_core PRIVATE
    src/
    src/tokenizer/
    src/attention/
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(_core PRIVATE pybind11::module)