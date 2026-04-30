cmake_minimum_required(VERSION 3.1)
include(CheckCXXCompilerFlag)

# project settings
project(HDGA)

# ========== GPU 配置 ==========
OPTION(USE_GPU_PHASE1 "Use GPU implementation for Phase1 alignment" OFF)
OPTION(USE_GPU_PHASE2 "Use GPU implementation for Phase2 alignment" OFF)

IF(USE_GPU_PHASE1 OR USE_GPU_PHASE2)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA enabled, version: ${CUDAToolkit_VERSION}")
ENDIF()

##### General Compilation Settings
set(CMAKE_CXX_FLAGS "-w ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)

# ========== 添加 CUDA 编译设置 ==========
IF(USE_GPU_PHASE1 OR USE_GPU_PHASE2)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_ARCHITECTURES 86) # 适配显卡架构，可根据实际修改
  # 移除 -v 选项以隐藏 ptxas 详细信息
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -O3")
  # 添加 CUDA 头文件路径，使普通 C++ 文件也能找到 cuda_runtime.h
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
ENDIF()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

#Set default cmake build type to RelWithDebInfo during prototyping
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo"
    FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )

# Check if the user want to build google test applications
OPTION(BUILD_TESTS "Inform whether test applications should be built" ON)
# Set vtune support
OPTION(ENABLE_VTUNE "Turn on/off the vtune support with intel compiler" OFF)
# Check if user wants to print verbose debug messages
OPTION(DEFINE_DEBUG_MACRO "Print verbose debug messages" OFF) 

IF(DEFINE_DEBUG_MACRO)
  ADD_DEFINITIONS(-DDEBUG)
ENDIF(DEFINE_DEBUG_MACRO)

# Assume avx2 availability by default
SET(SIMD_SUPPORT avx2 CACHE STRING "Choose SIMD instrinsics, options are: avx2 none")
SET_PROPERTY(CACHE SIMD_SUPPORT PROPERTY STRINGS avx2 none)

#User expected to provide the SIMD option
if (SIMD_SUPPORT STREQUAL avx2)
  # Check if compiler supports AVX2
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORT_AVX2)
    if(COMPILER_SUPPORT_AVX2)
      message(STATUS "GNU C++ compiler supports AVX2")
    else()
      message(FATAL_ERROR "Compiler must support AVX2 (GCC >= 4.9 or Intel >= 15)")
    endif()
  endif()
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    check_cxx_compiler_flag("-xCORE-AVX2" COMPILER_SUPPORT_AVX2)
    if(COMPILER_SUPPORT_AVX2)
      message(STATUS "Intel C++ compiler supports AVX2")
    else()
      message(FATAL_ERROR "Compiler must support AVX2 (GCC >= 4.9 or Intel >= 15)")
    endif()
  endif()

  # compiler flag to enable AVX2 simd instructions
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -DHDGA_ENABLE_AVX2")
  message(STATUS "Enabling AVX2...")

else()

  message(STATUS "AVX option not provided, disabling SIMD support...")

endif()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)

  #for recognizing numbers and boolean constants in if()
  if(CMAKE_VERSION VERSION_GREATER 2.6.4)
    cmake_policy(SET CMP0012 NEW)
  endif()
endif()

#### OpenMP
find_package(OpenMP)
# FindOpenMP defines the OpenMP_C_FLAGS and OpenMP_CXX_FLAGS.
if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "Found OpenMP")
endif()

#compile testing code
if (BUILD_TESTS)
  #Add googleTest support
  add_subdirectory("${PROJECT_SOURCE_DIR}/ext/gtest/googletest")
endif(BUILD_TESTS)

### Include protobuf
# Allow PROTOBUF_DIR to be set by cache or environment, otherwise use system discovery.
if(NOT PROTOBUF_DIR AND DEFINED ENV{PROTOBUF_DIR})
  set(PROTOBUF_DIR "$ENV{PROTOBUF_DIR}" CACHE STRING "absolute path to protobuf installation directory")
endif()

if(PROTOBUF_DIR)
  set(CMAKE_PREFIX_PATH "${PROTOBUF_DIR};${CMAKE_PREFIX_PATH}")
endif()

find_package(Protobuf REQUIRED)

if(PROTOBUF_DIR)
  # The variables below are set based on PROTOBUF_DIR
  set(PROTOBUF_INCLUDE_DIR "${PROTOBUF_DIR}/include")
  set(PROTOBUF_LIBRARY "${PROTOBUF_DIR}/lib/libprotobuf.so")
  set(PROTOBUF_PROTOC_EXECUTABLE "${PROTOBUF_DIR}/bin/protoc")

  if(EXISTS "${PROTOBUF_INCLUDE_DIR}" AND EXISTS "${PROTOBUF_LIBRARY}" AND EXISTS "${PROTOBUF_PROTOC_EXECUTABLE}")
    message(STATUS "Protobuf libraries, header files and executables located")
  else()
    message(FATAL_ERROR "Protobuf path is invalid; set PROTOBUF_DIR or install Protobuf in system paths.")
  endif()
else()
  set(PROTOBUF_INCLUDE_DIR ${Protobuf_INCLUDE_DIRS})
  set(PROTOBUF_LIBRARY ${Protobuf_LIBRARIES})
  set(PROTOBUF_PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})
endif()

include_directories(${PROTOBUF_INCLUDE_DIR})

### Include and compile htslib
include_directories("${PROJECT_SOURCE_DIR}/ext/htslib")
# Advice: in case htslib is missing some core libraries, install them
add_custom_target(LIBHTS COMMAND make WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/ext/htslib" COMMENT "Compiling htslib")
set(HTS_LIBRARY "${PROJECT_SOURCE_DIR}/ext/htslib/libhts.so")

### Include libvgio
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS "${PROJECT_SOURCE_DIR}/ext/libvgio/deps/vg.proto")
add_custom_target(SYMLNK COMMAND ${CMAKE_COMMAND} -E create_symlink . vg  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}" COMMENT "Adding symbolic link vg") 
include_directories("${PROJECT_SOURCE_DIR}/ext/libvgio/include")
file(GLOB VGIO_SRC "${PROJECT_SOURCE_DIR}/ext/libvgio/src/*.cpp")
add_library(LIBVGIO ${PROTO_SRCS} ${VGIO_SRC})
add_dependencies(LIBVGIO SYMLNK)

#include external header files
include_directories("${PROJECT_SOURCE_DIR}/ext")

#include our own directories
include_directories("${PROJECT_SOURCE_DIR}/src/include")
include_directories(${CMAKE_CURRENT_BINARY_DIR})
# add_subdirectory("${PROJECT_SOURCE_DIR}/tests")

#### To compile with Intel vtune library if requested
if(ENABLE_VTUNE)
  #Configure path to vtune_amplifier library 
  include_directories("/opt/intel/vtune_amplifier_2018/include")
  set(VTUNE_LIBRARY "/opt/intel/vtune_amplifier_2018/lib64/libittnotify.a")
  add_definitions(-DVTUNE_SUPPORT)
endif(ENABLE_VTUNE)

# ========== 根据选项添加宏定义 ==========
IF(USE_GPU_PHASE1)
  ADD_DEFINITIONS(-DUSE_GPU_PHASE1)
  message(STATUS "GPU Phase1 alignment enabled")
ENDIF(USE_GPU_PHASE1)

IF(USE_GPU_PHASE2)
  ADD_DEFINITIONS(-DUSE_GPU_PHASE2)
  message(STATUS "GPU Phase2 alignment enabled")
ENDIF(USE_GPU_PHASE2)

# ========== 设置源文件 ==========
set(HDGA_SRCS
  src/main.cpp
)

# 如果启用了 GPU，添加 CUDA 源文件
IF(USE_GPU_PHASE1 OR USE_GPU_PHASE2)
  list(APPEND HDGA_SRCS src/align_gpu_kernel.cu)
ENDIF()

add_executable(HDGA ${HDGA_SRCS})
add_dependencies(HDGA LIBHTS SYMLNK)
target_link_libraries(HDGA ${PROTOBUF_LIBRARY} LIBVGIO ${HTS_LIBRARY} ${VTUNE_LIBRARY} -lz -lpthread)

IF(USE_GPU_PHASE1 OR USE_GPU_PHASE2)
  target_link_libraries(HDGA ${CUDAToolkit_LIBRARIES})
ENDIF()
