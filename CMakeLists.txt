#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)
project(tflite_vx_delegate)

OPTION(ENABLE_NBG_SUPPORT "enable customized nbg op in tflite" ON)

if(ANDROID_TOOLCHAIN)
# bypass warning as error since tensorflow lite can not pass with android ndk r22b
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-c++11-narrowing")
endif()

include(FetchContent)
FetchContent_Declare(
  tensorflow
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG v2.6.0
)
FetchContent_GetProperties(tensorflow)
if(NOT tensorflow_POPULATED)
  FetchContent_Populate(tensorflow)
endif()

# TODO(sven) this make build slow/stuck if modify cmake files after build
if (EXISTS ${tensorflow_SOURCE_DIR}/tensorflow/lite/kernels/vsi_npu_precompiled.cc)
  message("Patch already applied to " ${tensorflow_SOURCE_DIR})
else()
  message("Apply tensorflow lite patches ...")
  execute_process(
    COMMAND git apply ${PROJECT_SOURCE_DIR}/patches/tflite/0001-Add-Customized-NBG-Tflite-Model-Support.patch
    TIMEOUT 2
    WORKING_DIRECTORY  ${tensorflow_SOURCE_DIR}
  )
endif()

add_subdirectory("${tensorflow_SOURCE_DIR}/tensorflow/lite"
  "${tensorflow_BINARY_DIR}")

list(APPEND VX_DELEGATE_DEPENDENCIES tensorflow-lite)
list(APPEND VX_DELEGATES_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/delegate_main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/op_map.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vx_delegate_adaptor.cc
)

if((NOT DEFINED EXTERNAL_VIV_SDK) AND (NOT DEFINED TIM_VX_INSTALL))
  FetchContent_Declare(
  tim-vx
  GIT_REPOSITORY https://github.com/VeriSilicon/TIM-VX.git
  GIT_TAG main
  )
  FetchContent_GetProperties(tim-vx)
  if(NOT tim-vx_POPULATED)
    FetchContent_Populate(tim-vx)
  endif()
  set(EXTERNAL_VIV_SDK ${tim-vx_SOURCE_DIR}/prebuilt-sdk/x86_64_linux)
  execute_process(COMMAND cmake . WORKING_DIRECTORY ${tim-vx_SOURCE_DIR})
  execute_process(COMMAND make install -j8 WORKING_DIRECTORY ${tim-vx_SOURCE_DIR})
  set(TIM_VX_INSTALL ${tim-vx_SOURCE_DIR}/install)
endif()

if(NOT TIM_VX_INSTALL OR NOT EXISTS ${TIM_VX_INSTALL})
  message(FATAL_ERROR "Set TIM_VX_INSTALL to tim-vx install folder(by make install) instead of ${TIM_VX_INSTALL}")
endif()

message("=== Building with TIM_VX_LIBRIRIES from ${TIM_VX_INSTALL} ===")
include_directories(${TIM_VX_INSTALL}/include)
list(APPEND VX_DELEGATE_DEPENDENCIES ${TIM_VX_INSTALL}/lib/libtim-vx.so)

add_library(vx_delegate SHARED ${VX_DELEGATES_SRCS})
target_link_libraries(vx_delegate ${VX_DELEGATE_DEPENDENCIES})

add_subdirectory(examples/minimal)
