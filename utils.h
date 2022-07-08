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
#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_

#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>
#include <sstream>
#include <cstring>
#include <cstdio>
#include "delegate_main.h"

namespace vx {
namespace delegate {
namespace utils {

// transpose channel_dim while doing transpose operation.
int32_t TransposeChannelDim(const std::vector<uint32_t>& perm,
                            int32_t channel_dim);

// Convert the perm in TfLite to the perm in vx-delegate when transpose.
std::vector<uint32_t> GetOvxTransposePerm(const std::vector<uint32_t>& perm);

// Convert TfLite axis to OpenVX kind.
inline int32_t ConvertAxis(int32_t axisIn, uint32_t dimNum) {
  return dimNum - (axisIn < 0 ? dimNum + axisIn : axisIn) - 1;
}

template <typename T>
std::vector<T> TransposeVec(const std::vector<T>& input,
                            const std::vector<int>& perm) {
  if (input.size() != perm.size()) {
    return std::vector<T>();
  };

  std::vector<T> output(input.size());
  for (int i = 0; i < perm.size(); i++) {
    output[i] = input[perm[i]];
  }

  return output;
}

inline int32_t CalcWeightSizeForBilinear(int32_t scale) {
  return 2 * scale - scale % 2;
}

inline int32_t CalcPadSizeForBilinear(int32_t scale) { return scale / 2; }

void GenerateWeightsDataForBilinear(float* data,
                                    const std::vector<uint32_t>& weight_shape,
                                    uint32_t scale_w,
                                    uint32_t scale_h);

void GenerateWeightDataForNearest(float* data,
                                  const std::vector<uint32_t>& weight_shape);

template <typename T>
inline void Quantize(const std::vector<float>& data, float scale,
                               int32_t zero_point, std::vector<T>& quant_data) {
  for (const auto& f : data) {
    quant_data.push_back(static_cast<T>(std::max<float>(
        std::numeric_limits<T>::min(),
        std::min<float>(std::numeric_limits<T>::max(),
                        std::round(zero_point + (f / scale))))));
  }
}

#define TFLITE_EXAMPLE_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

std::vector<uint8_t> read_data(const char * filename, size_t required);

std::vector<uint32_t> string_to_int(std::string string);

void UnpackConfig(const char* filename,
                   std::vector<std::string>& model_locations,
                   std::vector<uint32_t>& model_num,
                   std::vector<uint32_t>& devs_id,
                   std::vector<std::vector<std::string>>& inputs_datas);

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

  auto lhs_m = calc_m(lhs);
  auto rhs_m = calc_m(rhs);

  float element_sum = 0.f;
  for(auto i = 0U; i < lhs.size(); ++i) {
    element_sum += lhs[i]*rhs[i];
  }

  return element_sum/(lhs_m*rhs_m);
}

}  // namespace utils
}  // namespace delegate
}  // namespace vx

#endif /* TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_ */