/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_

#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>
#include "delegate_main.h"
#include "tensorflow/lite/tools/logging.h"

namespace vx {
namespace delegate {
namespace utils {

// transpose channel_dim while doing transpose operation.
inline int32_t TransposeChannelDim(const std::vector<uint32_t>& perm,
                                   int32_t channel_dim) {
  if (channel_dim < 0) {
    TFLITE_LOG(ERROR) << "invalid channel_dim";
    return -1;
  }
  for (uint32_t i = 0; i < perm.size(); i++) {
    if (channel_dim == perm.at(i)) {
      return i;
    }
  }
  TFLITE_LOG(ERROR) << "Can't find channle_dim";
  return -1;
}

// Convert the perm in TfLite to the perm in vx-delegate when transpose.
inline std::vector<uint32_t> GetOvxTransposePerm(const std::vector<uint32_t>& perm) {
  std::vector<uint32_t> perm_out(perm.rbegin(), perm.rend());
  std::vector<uint32_t> perm_in, ovx_perm;
  for (int i = perm.size() - 1; i >= 0; i--) {
    perm_in.push_back(i);
  }
  for (auto o : perm_out) {
    for (int i = 0; i < perm_in.size(); i++) {
      if (o == perm_in[i]) {
        ovx_perm.push_back(i);
        break;
      }
    }
  }

  return ovx_perm;
}

// Convert TfLite axis to OpenVX kind.
inline int32_t ConvertAxis(int32_t axisIn, uint32_t dimNum) {
  return dimNum - (axisIn < 0 ? dimNum + axisIn : axisIn) - 1;
}

template <typename T>
std::vector<T> TransposeVec(const std::vector<T>& input,
                            const std::vector<int>& perm) {
  assert(input.size() == perm.size());

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

}  // namespace utils
}  // namespace delegate
}  // namespace vx

#endif /* TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_ */