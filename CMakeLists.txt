# Project configuration
cmake_minimum_required(VERSION 3.12)
project(dtcc-solar)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified, using Release by default")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set build flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific settings
if(MSVC)
  add_compile_options(/W3)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
  add_compile_options(
    -Wall
    -Werror
    -Wno-unused-but-set-variable
    -Wno-reorder
  )
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -DLLVM_ENABLE_ASSERTIONS=ON")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Add path to local CMake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Add clang tools (not for Windows)
if(NOT WIN32)
  include(cmake/clang-cxx-dev-tools.cmake)
endif()

# Resolve the Python interpreter that drives the build (scikit-build passes this in)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11 using the active Python environment to avoid version mismatches
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
  message(FATAL_ERROR "pybind11 is required but could not be imported with ${Python3_EXECUTABLE}")
endif()

message(STATUS "pybind11 directory: ${pybind11_DIR}")
set(pybind11_DIR "${pybind11_DIR}" CACHE PATH "Path to pybind11 CMake files")
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen
#find_path(EIGEN3_INCLUDE_DIR Eigen/Dense PATHS /opt/homebrew/include/eigen3)

#if(NOT EIGEN3_INCLUDE_DIR)
#  message(FATAL_ERROR "Eigen not found. Ensure it's installed (e.g., via brew or vcpkg) and accessible.")
#endif()
#include_directories(${EIGEN3_INCLUDE_DIR})
#message(STATUS "Eigen found at: ${EIGEN3_INCLUDE_DIR}")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

get_target_property(EIGEN_INCLUDES Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Eigen include dirs: ${EIGEN_INCLUDES}")


# Configure OpenMP discovery (platform specific hints first)
#if(WIN32)
#  find_package(OpenMP)
if(APPLE)
  # macOS-specific OpenMP setup (assumes libomp via Homebrew)
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_C_LIB_NAMES "omp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
  set(OpenMP_INCLUDE_DIR /opt/homebrew/opt/libomp/include)
endif()

find_package(OpenMP)

set(EXTRA_LIBS "")

if(OpenMP_FOUND)
  message(STATUS "OpenMP found. Enabling parallel support.")
  list(APPEND EXTRA_LIBS OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found. Consider installing with 'brew install libomp'")
endif()

# Include project headers
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)

# Installation setup
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)
list(APPEND CMAKE_PREFIX_PATH /home/runner/work/dtcc-solar/dtcc-solar)

# Add subdirectories
add_subdirectory(src)

if(NOT WIN32)
  add_subdirectory(tests/cpp)
endif()

# Pre-commit hook (non-Windows only)
if(NOT WIN32)
  message("Running pre-commit hook")
  execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/utils/git/git-pre-commit-format install" OUTPUT_VARIABLE outVar)
  message("${outVar}")
endif()
