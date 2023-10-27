cmake_minimum_required(VERSION 3.4)

set(BINDINGS py_embree_solar)

project(embree_solar)

FIND_PACKAGE(embree REQUIRED)

execute_process(COMMAND python -c "import site; print([s for s in site.getsitepackages() if 'site-packages' in s][0])" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
set(pybind11_DIR ${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/pybind11)

message("Pybind11 DIR " ${pybind11_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${pybind11_DIR})
message("CMAKE_MODULE_PATH " ${CMAKE_MODULE_PATH})
find_package(pybind11 REQUIRED)

pybind11_add_module(${BINDINGS} embree_solar.cpp)
target_link_libraries(${BINDINGS} PRIVATE embree)

install(TARGETS ${BINDINGS} DESTINATION embree-solar)