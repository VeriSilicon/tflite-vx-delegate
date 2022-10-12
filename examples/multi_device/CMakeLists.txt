#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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


# Builds the multi_device Tensorflow Lite example.

#cmake_minimum_required(VERSION 3.16)
#project(multi_device C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    ${tensorflow_SOURCE_DIR}
    ABSOLUTE
  )
endif()

include_directories(${TFLITE_SOURCE_DIR}/delegates/external)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

set(CMAKE_CXX_STANDARD 14)
add_executable(multi_device
  multi_device.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../util.cc
)
target_link_libraries(multi_device
  tensorflow-lite
  vx_custom_op
  ${CMAKE_DL_LIBS}
)

if(ANDROID_TOOLCHAIN)
  target_link_libraries(multi_device
    log
  )
endif()
