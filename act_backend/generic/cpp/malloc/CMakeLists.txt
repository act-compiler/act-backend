cmake_minimum_required(VERSION 3.20)
project(malloc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIB_SOURCE_FILES
    src/act_malloc.cc
    src/constraint.cc
    src/emitter.cc
    src/globals.cc
    src/parser.cc
    src/scheduler.cc
)

# OR-Tools, in the directory ORTOOLS_DIR names
if(DEFINED ENV{ORTOOLS_DIR})
  set(ORTOOLS_ROOT "$ENV{ORTOOLS_DIR}")
else()
  set(ORTOOLS_ROOT "/opt/ortools/")
endif()
set(CMAKE_PREFIX_PATH "${ORTOOLS_ROOT}" ${CMAKE_PREFIX_PATH})
find_package(ortools REQUIRED PATHS "${ORTOOLS_ROOT}" NO_DEFAULT_PATH)

# The JSON headers, in the directory NLOHMANN_JSON_DIR names
if(DEFINED ENV{NLOHMANN_JSON_DIR})
  set(JSON_INCLUDE_PATH "$ENV{NLOHMANN_JSON_DIR}/include")
else()
  set(JSON_INCLUDE_PATH "/opt/nlohmann_json/include")
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${JSON_INCLUDE_PATH})

# Create a static library to be linked with Rust
add_library(malloc_static STATIC ${LIB_SOURCE_FILES})
target_link_libraries(malloc_static PUBLIC ortools::ortools)
target_link_libraries(malloc_static PUBLIC nlohmann_json)
target_include_directories(malloc_static PRIVATE include)

# The default cost model, in a backend that names no other
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../cost/count.cc")
  add_library(cost_static STATIC ../cost/count.cc)
endif()
