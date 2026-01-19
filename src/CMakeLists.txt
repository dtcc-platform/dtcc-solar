set(BINDINGS py_solar)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

# Build Python extension with pybind11
pybind11_add_module(${BINDINGS}
    ${SRC_DIR}/dtcc_solar.cpp
    ${SRC_DIR}/skydome.cpp
    ${SRC_DIR}/sunrays.cpp
    ${SRC_DIR}/rays.cpp
)


# Link libraries
target_link_libraries(${BINDINGS}
    PRIVATE
      pybind11::module
      Eigen3::Eigen
)

# Add OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(${BINDINGS} PRIVATE OpenMP::OpenMP_CXX)
endif()


# Compile definitions
target_compile_definitions(${BINDINGS} PRIVATE -DTRILIBRARY -DANSI_DECLARATORS)
if(WIN32)
  target_compile_definitions(${BINDINGS} PRIVATE -DNO_TIMER)
endif()

# MSVC runtime
if(MSVC)
  set_target_properties(${BINDINGS} PROPERTIES
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Install
install(TARGETS ${BINDINGS} DESTINATION dtcc_solar)
