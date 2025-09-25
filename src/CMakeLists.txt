set(BINDINGS py_embree_solar)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

# Embree discovery (supports CI-provided paths without breaking local installs)
cmake_policy(PUSH)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

if(NOT embree_DIR AND DEFINED ENV{embree_DIR})
  set(embree_DIR "$ENV{embree_DIR}" CACHE PATH "Embree config dir (lowercase)" FORCE)
endif()

if(NOT Embree_DIR AND DEFINED ENV{Embree_DIR})
  set(Embree_DIR "$ENV{Embree_DIR}" CACHE PATH "Embree config dir" FORCE)
endif()

if(NOT embree_DIR AND Embree_DIR)
  set(embree_DIR "${Embree_DIR}" CACHE PATH "Embree config dir (lowercase)" FORCE)
elseif(NOT Embree_DIR AND embree_DIR)
  set(Embree_DIR "${embree_DIR}" CACHE PATH "Embree config dir" FORCE)
endif()

if(NOT Embree_DIR AND DEFINED ENV{EMBREE_ROOT})
  set(_embree_root "$ENV{EMBREE_ROOT}")
  find_file(_embree_config
    NAMES embree-config.cmake EmbreeConfig.cmake
    PATHS
      "${_embree_root}"
      "${_embree_root}/lib"
      "${_embree_root}/lib/cmake"
    NO_DEFAULT_PATH
  )
  if(_embree_config)
    get_filename_component(_embree_dir "${_embree_config}" DIRECTORY)
    set(Embree_DIR "${_embree_dir}" CACHE PATH "Embree config dir" FORCE)
    set(embree_DIR "${_embree_dir}" CACHE PATH "Embree config dir (lowercase)" FORCE)
  else()
    list(APPEND CMAKE_PREFIX_PATH "${_embree_root}")
    list(APPEND CMAKE_PREFIX_PATH "${_embree_root}/lib/cmake")
  endif()
endif()

if(Embree_DIR)
  message(STATUS "Using Embree_DIR: ${Embree_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${Embree_DIR}")
endif()

find_package(embree 4.0 CONFIG REQUIRED)

cmake_policy(POP)

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
if(_embree_target_type STREQUAL "IMPORTED")
  foreach(_cfg IN ITEMS RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
    get_target_property(_embree_location ${_embree_target} IMPORTED_LOCATION_${_cfg})
    if(_embree_location)
      get_filename_component(_embree_dir "${_embree_location}" DIRECTORY)
      list(APPEND _embree_rpaths "${_embree_dir}")
    endif()
  endforeach()
  if(NOT _embree_rpaths)
    get_target_property(_embree_location ${_embree_target} IMPORTED_LOCATION)
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
