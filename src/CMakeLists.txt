set(BINDINGS py_embree_solar)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

# Embree is expected to be available on the system (e.g. installed under /usr/local)
find_package(embree 4.0 REQUIRED CONFIG)

# Build Python extension with pybind11
pybind11_add_module(${BINDINGS}
    ${SRC_DIR}/embree_solar.cpp
    ${SRC_DIR}/skydome.cpp
    ${SRC_DIR}/sunrays.cpp
    ${SRC_DIR}/rays.cpp
)

# Resolve Embree imported target
set(_embree_target "")
if(TARGET embree)
  set(_embree_target embree)
elseif(TARGET Embree::Embree)
  set(_embree_target Embree::Embree)
endif()

if(NOT _embree_target)
  message(FATAL_ERROR "Embree imported target not found after find_package")
endif()

# Link libraries
target_link_libraries(${BINDINGS}
    PRIVATE
      ${_embree_target}
      pybind11::module
      Eigen3::Eigen
)

# Add OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(${BINDINGS} PRIVATE OpenMP::OpenMP_CXX)
endif()

# RPATH for Embree
get_target_property(_embree_target_type ${_embree_target} TYPE)
set(_embree_rpaths "")
set(_embree_runtime_dirs "")
if(_embree_target_type STREQUAL "IMPORTED")
  foreach(_cfg IN ITEMS RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
    get_target_property(_embree_location ${_embree_target} IMPORTED_LOCATION_${_cfg})
    if(_embree_location)
      get_filename_component(_embree_dir "${_embree_location}" DIRECTORY)
      list(APPEND _embree_rpaths "${_embree_dir}")
      list(APPEND _embree_runtime_dirs "${_embree_dir}")
    endif()
  endforeach()
  if(NOT _embree_rpaths)
    get_target_property(_embree_location ${_embree_target} IMPORTED_LOCATION)
    if(_embree_location)
      get_filename_component(_embree_dir "${_embree_location}" DIRECTORY)
      list(APPEND _embree_rpaths "${_embree_dir}")
      list(APPEND _embree_runtime_dirs "${_embree_dir}")
    endif()
  endif()
endif()

set(_embree_root "")
if(DEFINED embree_DIR)
  get_filename_component(_embree_root "${embree_DIR}" DIRECTORY)
  get_filename_component(_embree_root "${_embree_root}" DIRECTORY)
  get_filename_component(_embree_root "${_embree_root}" DIRECTORY)
endif()

if(_embree_rpaths)
  list(REMOVE_DUPLICATES _embree_rpaths)
  set_target_properties(${BINDINGS}
    PROPERTIES
      BUILD_RPATH "${_embree_rpaths}"
      INSTALL_RPATH "${_embree_rpaths}"
  )
endif()

if(WIN32)
  if(_embree_root)
    list(APPEND _embree_runtime_dirs "${_embree_root}/bin" "${_embree_root}")
  endif()
  if(_embree_runtime_dirs)
    list(REMOVE_DUPLICATES _embree_runtime_dirs)
    foreach(_runtime_dir IN LISTS _embree_runtime_dirs)
      if(EXISTS "${_runtime_dir}")
        file(GLOB _embree_runtime_dlls "${_runtime_dir}/*.dll")
        foreach(_dll_path IN LISTS _embree_runtime_dlls)
          add_custom_command(TARGET ${BINDINGS} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll_path}" $<TARGET_FILE_DIR:${BINDINGS}>
          )
          install(FILES "${_dll_path}" DESTINATION dtcc_solar)
        endforeach()
      endif()
    endforeach()
  endif()
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
