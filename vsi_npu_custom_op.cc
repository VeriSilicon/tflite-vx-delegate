/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/spectrogram.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include "vsi_npu_custom_op.h"

namespace tflite {
namespace ops {
namespace custom {
namespace vsi_npu {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TfLiteVsiNpuParams* data = reinterpret_cast<TfLiteVsiNpuParams*>(
      malloc(sizeof(TfLiteVsiNpuParams) + sizeof(char) * length));
  data->length = length;
  data->binary = reinterpret_cast<char*>(data) + sizeof(TfLiteVsiNpuParams);
  memcpy(reinterpret_cast<char*>(data->binary), buffer, length);
  return reinterpret_cast<void*>(data);
}

void Free(TfLiteContext* context, void* buffer) {
  auto* data = reinterpret_cast<TfLiteVsiNpuParams*>(buffer);
  delete data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* data =
      reinterpret_cast<TfLiteVsiNpuParams*>(node->user_data);
  data->input_count = NumInputs(node);
  data->output_cout = NumOutputs(node);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

}  // namespace vsi_npu

TfLiteRegistration* Register_VSI_NPU_PRECOMPILED() {
  static TfLiteRegistration r = {
      vsi_npu::Init, vsi_npu::Free,
      vsi_npu::Prepare,vsi_npu::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
