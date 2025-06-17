
set(BINDINGS py_embree_solar)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

FIND_PACKAGE(embree 4.0 REQUIRED)

pybind11_add_module(${BINDINGS} ${SRC_DIR}/embree_solar.cpp ${SRC_DIR}/skydome.cpp ${SRC_DIR}/sunrays.cpp ${SRC_DIR}/pydome.cpp)
target_link_libraries(${BINDINGS} PRIVATE embree)

target_link_libraries(py_embree_solar
    PRIVATE
    pybind11::module
    ${EXTRA_LIBS}  # OpenMP will be added here if found
)

target_include_directories(${BINDINGS} PRIVATE ${EIGEN3_INCLUDE_DIR})

target_compile_definitions(${BINDINGS} PRIVATE -DTRILIBRARY -DANSI_DECLARATORS)
if(WIN32)
    target_compile_definitions(${BINDINGS} PRIVATE -DNO_TIMER)
endif()

install(TARGETS ${BINDINGS} DESTINATION dtcc_solar)
