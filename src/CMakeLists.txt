#cmake_minimum_required(VERSION 3.4)

set(BINDINGS py_embree_solar)

#project(dtcc-solar)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

FIND_PACKAGE(embree REQUIRED)

#execute_process(COMMAND python -c "import site; print([s for s in site.getsitepackages() if 'site-packages' in s][0])" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
#set(pybind11_DIR ${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/pybind11)

#message("Pybind11 DIR " ${pybind11_DIR})
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${pybind11_DIR})
#message("CMAKE_MODULE_PATH " ${CMAKE_MODULE_PATH})
#find_package(pybind11 REQUIRED)

pybind11_add_module(${BINDINGS} ${SRC_DIR}/embree_solar.cpp ${SRC_DIR}/skydome.cpp ${SRC_DIR}/sunrays.cpp)
target_link_libraries(${BINDINGS} PRIVATE embree)

target_compile_definitions(${BINDINGS} PRIVATE -DTRILIBRARY -DANSI_DECLARATORS)
if(WIN32)
    target_compile_definitions(${BINDINGS} PRIVATE -DNO_TIMER)
endif()

install(TARGETS ${BINDINGS} DESTINATION dtcc_solar)
