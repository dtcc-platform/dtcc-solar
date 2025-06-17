# Project configuration
cmake_minimum_required(VERSION 3.12)
project(dtcc-solar)

# Note: To building in Debug mode, use cmake .. -DCMAKE_BUILD_TYPE=Debug

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified, using Release by default")
endif()

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Set build flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress specific warning that breaks Eigen's Tensor module
add_compile_options(
    -Wall
    -Werror
    -Wno-unused-but-set-variable
    -Wno-reorder
)

# OpenMP workaround for macOS using libomp from Homebrew
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")

# Try to find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
  message(WARNING "OpenMP NOT found. Try installing with 'brew install libomp'")
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -DLLVM_ENABLE_ASSERTIONS=ON")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Add path to local CMake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Add clang tools
include(cmake/clang-cxx-dev-tools.cmake)

# Find pybind11
execute_process(COMMAND python -c "import site; print([s for s in site.getsitepackages() if 'site-packages' in s][0])" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
set(pybind11_DIR ${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/pybind11)

message("Pybind11 DIR " ${pybind11_DIR})
find_package(pybind11 REQUIRED)

# Find Eigen
find_path(EIGEN3_INCLUDE_DIR Eigen/Dense PATHS /opt/homebrew/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen not found. Make sure it's installed (e.g., via brew) and accessible.")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen found at: ${EIGEN3_INCLUDE_DIR}")

# Set include directories
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)

# Set installation directory
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)
# Needs to be appended to local installations, this is hardcoded for CI
list(APPEND CMAKE_PREFIX_PATH /home/runner/work/dtcc-solar/dtcc-solar)

# Add subdirectories
add_subdirectory(src)

if(NOT WIN32)
  add_subdirectory(tests/cpp)
endif()

# Pre-commit hook
message("Running pre-commit hook")
execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/utils/git/git-pre-commit-format install"
  OUTPUT_VARIABLE outVar)
message("${outVar}")
