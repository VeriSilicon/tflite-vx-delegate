/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#ifndef VX_DELEGATE_EXAMPLE_UTIL_H_
#define VX_DELEGATE_EXAMPLE_UTIL_H_

#include <vector>
#include <sstream>
#include <cstring>
#include <map>
#include <fstream>
#include <iostream>
#include <cassert>
#include <math.h>
#include <cstdint>

#define TFLITE_EXAMPLE_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

template< typename T>
float cosine(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  auto calc_m = [](const std::vector<T>& lhs) {
    float lhs_m = 0.0f;

    for(auto iter = lhs.begin(); iter != lhs.end(); ++iter) {
      lhs_m += *iter * (*iter);
    }
    lhs_m = std::sqrt(lhs_m);

    return lhs_m;
  };

  if (lhs.size() == 1 ){ // Two values are both scalar, just compare similarity instead of cosinesimilarity
    float ans = 0.f;
    ans = (float)lhs[0]/(float)rhs[0] > 1? (float)rhs[0]/(float)lhs[0]  :(float)lhs[0]/(float)rhs[0] ;
    return ans;
  }

  auto lhs_m = calc_m(lhs);
  auto rhs_m = calc_m(rhs);

  float element_sum = 0.f;
  for(auto i = 0U; i < lhs.size(); ++i) {
    element_sum += lhs[i]*rhs[i];
  }

  return element_sum/(lhs_m*rhs_m);
}

std::vector<uint8_t> ReadData(const char* model_location,
                              const char* filename,
                              size_t input_id,
                              size_t required);

std::vector<uint32_t> StringToInt(std::string string);

void UnpackConfig(const char* filename,
                   std::vector<std::string>& model_locations,
                   std::vector<uint32_t>& model_num,
                   std::vector<uint32_t>& devs_id,
                   std::vector<std::vector<std::string>>& inputs_datas);

template <typename T>
void CompareTensorResult(size_t idx,
                         T* npu_out_buf,
                         T* cpu_out_buf,
                         uint32_t bytes) {
  int count = 0;
  if (typeid(T) == typeid(int8_t)) {
    for (auto j = 0; j < bytes; ++j) {
      if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 2 && count < 100) {
        std::cout << "[Result mismatch]: Output[" << idx <<","<<j <<"], CPU vs NPU("
                  << static_cast<int32_t>(cpu_out_buf[j]) << ","
                  << static_cast<int32_t>(npu_out_buf[j]) << ")" << std::endl;

        count++;
      }
      else if(count == 100) break;
    }
  } else if (typeid(T) == typeid(uint8_t)) {
    for (auto j = 0; j < bytes; ++j) {
      if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 2 && count < 100) {
        std::cout << "[Result mismatch]: Output[" << idx <<","<<j <<"], CPU vs NPU("
                  << static_cast<int32_t>(cpu_out_buf[j]) << ","
                  << static_cast<int32_t>(npu_out_buf[j]) << ")" << std::endl;

        count++;
      }
      else if(count == 100) break;
    }
  } else if (typeid(T) == typeid(float_t)) {
      for (auto j = 0; j < bytes / sizeof(float_t); ++j) {
        if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 0.001f && count < 100) {  // TODO{sven}: not accurate
          std::cout << "[Result mismatch]: Output[" << idx <<","<<j <<"], CPU vs NPU("
                    << cpu_out_buf[j] << "," << npu_out_buf[j] << ")"
                    << std::endl;
          count++;
        }
        else if(count == 100) break;
      }
  } else if (typeid(T) == typeid(int32_t)) {
      for (auto j = 0; j < bytes / sizeof(int32_t); ++j) {
        if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 2 && count < 100) {
          std::cout << "[Result mismatch]: Output[" << idx <<","<<j <<"], CPU vs NPU("
                    << cpu_out_buf[j] << "," << npu_out_buf[j] << ")"
                    << std::endl;
          count++;
        }
        else if(count == 100) break;
      }
    }
  else {
    TFLITE_EXAMPLE_CHECK(false);
  }

  std::vector<T> lhs(bytes / sizeof(T));
  std::vector<T> rhs(bytes / sizeof(T));

  memcpy(lhs.data(), cpu_out_buf, bytes);
  memcpy(rhs.data(), npu_out_buf, bytes);

  std::cout << "The "<<idx<<" output CosineCosineSimilarity = " << cosine(lhs, rhs) << std::endl;
};

#endif /* VX_DELEGATE_EXAMPLE_UTIL_H_ */