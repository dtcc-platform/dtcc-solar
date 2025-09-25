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

# Propagate embree's runtime library directory so users do not have to export
# LD_LIBRARY_PATH manually when loading the extension.
get_target_property(_embree_target_type embree TYPE)
set(_embree_rpaths "")
if(_embree_target_type STREQUAL "IMPORTED")
  foreach(_cfg IN ITEMS RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
    get_target_property(_embree_location embree IMPORTED_LOCATION_${_cfg})
    if(_embree_location)
      get_filename_component(_embree_dir "${_embree_location}" DIRECTORY)
      list(APPEND _embree_rpaths "${_embree_dir}")
    endif()
  endforeach()
  if(NOT _embree_rpaths)
    get_target_property(_embree_location embree IMPORTED_LOCATION)
    if(_embree_location)
      get_filename_component(_embree_dir "${_embree_location}" DIRECTORY)
      list(APPEND _embree_rpaths "${_embree_dir}")
    endif()
  endif()
endif()

if(_embree_rpaths)
  list(REMOVE_DUPLICATES _embree_rpaths)
  set_target_properties(${BINDINGS}
    PROPERTIES
      BUILD_RPATH "${_embree_rpaths}"
      INSTALL_RPATH "${_embree_rpaths}"
  )
endif()

set_property(TARGET ${BINDINGS} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add include path for Eigen
#target_include_directories(${BINDINGS} PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${BINDINGS} PRIVATE Eigen3::Eigen)

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
