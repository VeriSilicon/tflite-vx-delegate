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
#    DEALINGS IN THE SOFTWARE
#
set(TIM_VX_ENABLE_40BIT      ON)
set(TIM_VX_ENABLE_PLATFORM   ON)
if((NOT DEFINED TIM_VX_INSTALL))
  include(FetchContent)
  FetchContent_Declare(
    tim-vx
    GIT_REPOSITORY https://github.com/VeriSilicon/TIM-VX.git
    GIT_TAG main
  )
  FetchContent_GetProperties(tim-vx)
  if(NOT tim-vx_POPULATED)
    FetchContent_Populate(tim-vx)
  endif()
  include_directories(${tim-vx_SOURCE_DIR}/include)
  add_subdirectory("${tim-vx_SOURCE_DIR}"
                   "${tim-vx_BINARY_DIR}")
  list(APPEND VX_DELEGATE_DEPENDENCIES tim-vx)
else()
  message("=== Building with TIM_VX_LIBRIRIES from ${TIM_VX_INSTALL} ===")
  include_directories(${TIM_VX_INSTALL}/include)
  set(LIBDIR lib)
  list(APPEND VX_DELEGATE_DEPENDENCIES ${TIM_VX_INSTALL}/${LIBDIR}/libtim-vx.so)
endif()
