#    Copyright (c) 2021 Vivante Corporation
#
#    Permission is hereby granted, free of charge, to any person obtaining a
#    copy of this software and associated documentation files (the "Software"),
#    to deal in the Software without restriction, including without limitation
#    the rights to use, copy, modify, merge, publish, distribute, sublicense,
#    and/or sell copies of the Software, and to permit persons to whom the
#    Software is furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#    DEALINGS IN THE SOFTWARE.
#
cmake_minimum_required(VERSION 3.16)

option(TFLITE_ENABLE_MULTI_DEVICE       "Enable multi devices support"          OFF)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

project(tflite_vx_delegate)

OPTION(ENABLE_NBG_SUPPORT "enable customized nbg op in tflite" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ANDROID_TOOLCHAIN)
# bypass warning as error since tensorflow lite can not pass with android ndk r22b
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-c++11-narrowing")
endif()

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH}
)

find_package(tensorflow REQUIRED)
find_package(tim-vx REQUIRED)

list(APPEND VX_DELEGATES_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/delegate_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/op_map.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vx_delegate_adaptor.cc
)

if(TFLITE_ENABLE_MULTI_DEVICE)
  ADD_DEFINITIONS(-DMULTI_DEVICE_FEATURE_MODE)
endif()

add_library(vx_delegate SHARED ${VX_DELEGATES_SRCS})

list(APPEND VX_CUSTOM_OP_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/vsi_npu_custom_op.cc
)
if(ANDROID_TOOLCHAIN)
  list(APPEND VX_DELEGATE_DEPENDENCIES log)
endif()

target_link_libraries(vx_delegate ${VX_DELEGATE_DEPENDENCIES})
if((NOT DEFINED TIM_VX_INSTALL))
target_link_libraries(vx_delegate -Wl,--whole-archive tim-vx)
endif()
add_library(vx_custom_op STATIC ${VX_CUSTOM_OP_SRCS})
target_include_directories(vx_custom_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(vx_custom_op TensorFlow::tensorflow-lite)
add_dependencies(vx_custom_op vx_delegate)

set_target_properties(benchmark_model PROPERTIES INTERFACE_LINK_LIBRARIES vx_custom_op)
set_target_properties(label_image PROPERTIES INTERFACE_LINK_LIBRARIES vx_custom_op)

add_subdirectory(examples/minimal)
if(TFLITE_ENABLE_MULTI_DEVICE)
  add_subdirectory(examples/multi_device)
endif()
