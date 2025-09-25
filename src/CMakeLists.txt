set(BINDINGS py_embree_solar)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

# Embree discovery (support CI-provided paths)
cmake_policy(PUSH)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

if(NOT embree_DIR AND DEFINED ENV{embree_DIR})
  set(embree_DIR "$ENV{embree_DIR}")
endif()

if(NOT Embree_DIR AND DEFINED ENV{Embree_DIR})
  set(Embree_DIR "$ENV{Embree_DIR}")
elseif(NOT Embree_DIR AND embree_DIR)
  set(Embree_DIR "${embree_DIR}")
endif()

if(NOT Embree_DIR AND DEFINED ENV{EMBREE_ROOT})
  file(GLOB _embree_configs
    "$ENV{EMBREE_ROOT}/lib/cmake/embree*/embree-config.cmake"
    "$ENV{EMBREE_ROOT}/lib/cmake/embree*/EmbreeConfig.cmake"
  )
  if(_embree_configs)
    list(GET _embree_configs 0 _embree_config_path)
    get_filename_component(_embree_dir "${_embree_config_path}" DIRECTORY)
    set(Embree_DIR "${_embree_dir}" CACHE PATH "Path to Embree config" FORCE)
    set(embree_DIR "${_embree_dir}" CACHE PATH "Path to Embree config (lowercase)" FORCE)
  else()
    list(APPEND CMAKE_PREFIX_PATH "$ENV{EMBREE_ROOT}")
  endif()
endif()

if(Embree_DIR)
  message(STATUS "Using Embree_DIR: ${Embree_DIR}")
endif()

find_package(Embree 4.0 REQUIRED CONFIG)

cmake_policy(POP)

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
