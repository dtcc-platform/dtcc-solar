set(BINDINGS py_embree_solar)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

# Embree
find_package(embree 4.0 REQUIRED)

# Build Python extension with pybind11
pybind11_add_module(${BINDINGS}
    ${SRC_DIR}/embree_solar.cpp
    ${SRC_DIR}/skydome.cpp
    ${SRC_DIR}/sunrays.cpp
    ${SRC_DIR}/rays.cpp
)

# Link libraries
target_link_libraries(${BINDINGS}
    PRIVATE
      embree
      pybind11::module
      Eigen3::Eigen
)

# Add OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(${BINDINGS} PRIVATE OpenMP::OpenMP_CXX)
endif()

# RPATH for embree
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