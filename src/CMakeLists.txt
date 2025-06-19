set(BINDINGS py_embree_solar)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

# Find Embree (required)
find_package(embree 4.0 REQUIRED)

# Add pybind11 module
pybind11_add_module(${BINDINGS}
    ${SRC_DIR}/embree_solar.cpp
    ${SRC_DIR}/skydome.cpp
    ${SRC_DIR}/sunrays.cpp
    ${SRC_DIR}/rays.cpp
)

# Link Embree
target_link_libraries(${BINDINGS} PRIVATE embree)

# Link pybind11 and OpenMP if available
target_link_libraries(${BINDINGS}
    PRIVATE
    pybind11::module
    ${EXTRA_LIBS}  # This should include OpenMP if set in root CMakeLists
)

# Add include path for Eigen
target_include_directories(${BINDINGS} PRIVATE ${EIGEN3_INCLUDE_DIR})

# Platform-specific definitions
target_compile_definitions(${BINDINGS} PRIVATE -DTRILIBRARY -DANSI_DECLARATORS)
if(WIN32)
    target_compile_definitions(${BINDINGS} PRIVATE -DNO_TIMER)
endif()

# Ensure embree and others are linked properly on Windows
if(MSVC)
    # Force use of multi-threaded runtime
    set_target_properties(${BINDINGS} PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Install target
install(TARGETS ${BINDINGS} DESTINATION dtcc_solar)