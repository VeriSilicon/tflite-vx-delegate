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
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include "op_map.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/lstm_shared.h"
#include "tim/vx/ops.h"
#include "utils.h"
#include "vsi_npu_custom_op.h"
#include "delegate_main.h"

using namespace tflite;
using namespace tflite::ops::builtin;

namespace {

inline tim::vx::PadType TflitePadTypeToVsiPadType(TfLitePadding pad) {
  switch (pad) {
    case kTfLitePaddingUnknown:
      return tim::vx::PadType::AUTO;
    case kTfLitePaddingValid:
      return tim::vx::PadType::VALID;
    case kTfLitePaddingSame:
      return tim::vx::PadType::SAME;
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unsuppoted pad type: %d", pad);
      break;
  }

  return tim::vx::PadType::AUTO;
}

/// Insert activation layer before the `original_tensor`
/// Return the input tensor of new activation layer
std::shared_ptr<tim::vx::Tensor> ProcessFusedActivation(
    vx::delegate::Delegate* delegate,
    TfLiteFusedActivation fused_activation,
    const std::shared_ptr<tim::vx::Tensor>& original_tensor) {
  std::shared_ptr<tim::vx::Operation> op = nullptr;
  switch (fused_activation) {
    case kTfLiteActNone:
      return original_tensor;
    case kTfLiteActRelu:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Relu>();
      break;
    case kTfLiteActReluN1To1:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Relu1>();
      break;
    case kTfLiteActRelu6:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Relu6>();
      break;
    case kTfLiteActTanh:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Tanh>();
      break;
    case kTfLiteActSigmoid:
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Sigmoid>();
      break;
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "Unsupported fused activation: %d",
                      fused_activation);
  }

  auto processed_tensor = delegate->GetGraph()->CreateTensor(
      original_tensor->GetSpec().AsTransientSpec());

  (*op).BindInput(processed_tensor);
  (*op).BindOutput(original_tensor);

  delegate->GetOps().push_back(op);
  // delegate->GetTensors().push_back(processed_tensor);
  // To prevent the id conflict between processed_tensor and model tensor,
  // add an offset to the processed_tensor tensor id
  delegate->GetTensors().insert(
      std::make_pair(delegate->GetTensors().size() + 0x40000000, processed_tensor));

  return processed_tensor;
}

std::shared_ptr<tim::vx::Tensor> ReverseInputTensor(
    vx::delegate::Delegate* delegate,
    const std::shared_ptr<tim::vx::Tensor>& original_tensor,
    std::vector<int32_t> axis) {
  auto reversed_tensor =
      delegate->GetGraph()->CreateTensor(original_tensor->GetSpec());
  std::shared_ptr<tim::vx::Operation> op =
      delegate->GetGraph()->CreateOperation<tim::vx::ops::Reverse>(axis);
  (*op).BindInput(original_tensor);
  (*op).BindOutput(reversed_tensor);

  delegate->GetOps().push_back(op);
  // delegate->GetTensors().push_back(reversed_tensor);
  // To prevent the id conflict between processed_tensor and model tensor,
  // add an offset to the processed_tensor tensor id
  delegate->GetTensors().insert(
      std::make_pair(delegate->GetTensors().size() + 0x40000000, reversed_tensor));

  return reversed_tensor;
}

bool ResizeToTransposeConv(
    vx::delegate::Delegate* delegate,
    std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
    std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
    tim::vx::ResizeType resizeType,
    uint32_t channel,
    uint32_t scale_w,
    uint32_t scale_h) {
  uint32_t kernel_w = 0;
  uint32_t kernel_h = 0;
  uint32_t pad_w = 0;
  uint32_t pad_h = 0;
  std::vector<float> weight_data;

  if (resizeType == tim::vx::ResizeType::BILINEAR) {
    kernel_w = vx::delegate::utils::CalcWeightSizeForBilinear(scale_w);
    kernel_h = vx::delegate::utils::CalcWeightSizeForBilinear(scale_h);

    pad_w = vx::delegate::utils::CalcPadSizeForBilinear(scale_w);
    pad_h = vx::delegate::utils::CalcPadSizeForBilinear(scale_h);

    weight_data.resize(kernel_h * kernel_w * channel * channel);
    vx::delegate::utils::GenerateWeightsDataForBilinear(
        weight_data.data(),
        {kernel_w, kernel_h, channel, channel},
        scale_w,
        scale_h);
  } else if (resizeType == tim::vx::ResizeType::NEAREST_NEIGHBOR) {
    kernel_w = scale_w;
    kernel_h = scale_h;

    pad_w = 0;
    pad_h = 0;

    weight_data.resize(kernel_h * kernel_w * channel * channel);
    vx::delegate::utils::GenerateWeightDataForNearest(
        weight_data.data(), {kernel_w, kernel_h, channel, channel});
  }

  auto weight_spec = tim::vx::TensorSpec(tim::vx::DataType::FLOAT32,
                                         {kernel_w, kernel_h, channel, channel},
                                         tim::vx::TensorAttribute::CONSTANT);
  std::shared_ptr<tim::vx::Tensor> weight_tensor;

  auto input_type = inputs[0]->GetDataType();
  auto input_quant = inputs[0]->GetQuantization();
  uint32_t kernel_size = kernel_h * kernel_w * channel * channel;
  std::vector<uint8_t> weight_quant_data(kernel_size);

  if (input_quant.Type() == tim::vx::QuantType::ASYMMETRIC) {
    float scale = input_quant.Scales()[0];
    int32_t zp = input_quant.ZeroPoints()[0];
    if (input_type == tim::vx::DataType::INT8) {
      std::vector<int8_t> quant_i8;
      vx::delegate::utils::Quantize<int8_t>(weight_data, scale, zp, quant_i8);
      weight_spec.SetDataType(tim::vx::DataType::INT8);
      memcpy(weight_quant_data.data(), quant_i8.data(), kernel_size);
    } else if (input_type == tim::vx::DataType::UINT8) {
      std::vector<uint8_t> quant_u8;
      vx::delegate::utils::Quantize<uint8_t>(weight_data, scale, zp, quant_u8);
      weight_spec.SetDataType(tim::vx::DataType::UINT8);
      memcpy(weight_quant_data.data(), quant_u8.data(), kernel_size);
    }

    weight_spec.SetQuantization(input_quant);
    weight_tensor = delegate->GetGraph()->CreateTensor(
        weight_spec, weight_quant_data.data());
  } else {
    weight_tensor =
        delegate->GetGraph()->CreateTensor(weight_spec, weight_data.data());
  }

  std::array<uint32_t, 2> ksize{kernel_w, kernel_h};
  std::array<uint32_t, 2> stride{scale_w, scale_h};
  std::array<uint32_t, 2> output_padding{0, 0};
  std::array<uint32_t, 4> pad{pad_w, pad_w, pad_h, pad_h};

  auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::DeConv2d>(
      channel,
      tim::vx::PadType::SAME,
      ksize,
      stride,
      output_padding,
      pad,
      1,
      tim::vx::DataLayout::CWHN);

  std::vector<std::shared_ptr<tim::vx::Tensor>> final_inputs;
  final_inputs.push_back(inputs[0]);
  final_inputs.push_back(weight_tensor);

  (*op).BindInputs(final_inputs);
  (*op).BindOutput(outputs[0]);

  delegate->GetOps().push_back(std::move(op));
  return true;
}

enum class ActionTargetType { INPUT, OUTPUT, STATE };

struct IAction {
  virtual ActionTargetType GetActionTargetType() const = 0;
  virtual bool process(vx::delegate::Delegate* delegate,
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
                       const void* params) const = 0;
  virtual ~IAction(){};
};

template <ActionTargetType type, int Port>
struct ActionBase : public IAction {
  ActionTargetType type_{type};
  int port_{Port};
  bool process(vx::delegate::Delegate* delegate,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
               const void* params) const override {
    return true;
  }
  ActionTargetType GetActionTargetType() const final { return type_; }
};

template <int Port, typename T_Param>
struct FusedActivationAction
    : public ActionBase<ActionTargetType::OUTPUT, Port> {
  bool process(vx::delegate::Delegate* delegate,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
               const void* params) const final {
    const auto builtin = reinterpret_cast<const T_Param*>(params);
    outputs[this->port_] = ProcessFusedActivation(
        delegate, builtin->activation, outputs[this->port_]);
    return true;
  }
};

template <typename T_Param, typename... Actions>
struct OpMapperBase : public vx::op_map::IOpMapper {
  std::vector<std::unique_ptr<IAction>> actions_;

  OpMapperBase() {
    (void)std::initializer_list<int>{
        0, (actions_.emplace_back(std::make_unique<Actions>()), 0)...};
  }

  size_t GetParamSize() const override { return sizeof(T_Param); }

  bool IsSupported(TfLiteContext* context,
                   TfLiteNode* node,
                   const TfLiteRegistration* registration) const override {
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (input_index < 0) {
        continue;
      }
      if (context->tensors[input_index].type == kTfLiteInt64) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Int64 input is not supported");
        return false;
      }
      if (context->tensors[input_index].type == kTfLiteString) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "String input is not supported");
        return false;
      }
      if (context->tensors[input_index].dims->size > 6) {
        TFLITE_LOG_PROD(
            TFLITE_LOG_ERROR,
            "vx-delegate doesn't support the tensor whose dimension "
            "is greater than 6.");
        return false;
      }
      for (int j = 0; j < context->tensors[input_index].dims->size; j++) {
        if (context->tensors[input_index].dims->data[j] == 0) {
          TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                          "vx-delegate doesn't support the tensor of which one "
                          "of dims is 0");
          return false;
        }
        if ((context->tensors[input_index].dims->data[j] > 65535) && (j > 0)) {
          TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                          "vx-delegate doesn't support tensor height/width > "
                          "65535");
          return false;
        }
      }
    }
    for (int i = 0; i < node->outputs->size; i++) {
      int output_index = node->outputs->data[i];
      if (context->tensors[output_index].type == kTfLiteInt16) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Int16 output is not supported");
        return false;
      }
      if (context->tensors[output_index].type == kTfLiteInt64) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Int64 output is not supported");
        return false;
      }
      for (int j = 0; j < context->tensors[output_index].dims->size; j++) {
        if (context->tensors[output_index].dims->data[j] == 0) {
          TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                          "vx-delegate doesn't support the tensor of which one "
                          "of dims is 0");
          return false;
        }
      }
    }

    return IsOpSupported(context, node, registration);
  }

  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    return true;
  }

  bool MapOp(vx::delegate::Delegate* delegate,
             std::vector<std::shared_ptr<tim::vx::Tensor>> inputs,
             std::vector<std::shared_ptr<tim::vx::Tensor>> outputs,
             std::vector<std::shared_ptr<tim::vx::Tensor>> states,
             const void* params) {
    bool status = true;

    for (auto& a : actions_) {
      if (a->GetActionTargetType() == ActionTargetType::INPUT) {
        a->process(delegate, inputs, outputs, states, params);
      }
    }

    for (auto it = actions_.rbegin(); it != actions_.rend(); it++) {
      if ((*it)->GetActionTargetType() == ActionTargetType::OUTPUT) {
        (*it)->process(delegate, inputs, outputs, states, params);
      }
    }

    status = HandleMapOp(delegate, inputs, outputs, states, params);

    return status;
  }

  virtual bool HandleMapOp(
      vx::delegate::Delegate* delegate,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& states,
      const void* params) {
    return HandleMapOp(delegate, inputs, outputs, params);
  }

  virtual bool HandleMapOp(
      vx::delegate::Delegate* delegate,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
      const void* params) {
    return false;
  }
};

}  // namespace
namespace vx {
namespace op_map {

template <typename T_OperationType>
struct SimpleOpMapper : public OpMapperBase<EmptyStructPlaceholder> {
  std::string name_;

  SimpleOpMapper(std::string name) : name_(name) {}

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating %s op", name_.c_str());

    auto op = delegate->GetGraph()->CreateOperation<T_OperationType>();
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct MinimumMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Minimum op");

    bool boradcast_required = (inputs[0]->GetShape() != inputs[1]->GetShape());
    std::vector<std::shared_ptr<tim::vx::Tensor>> elementwise_inputs;
    if (boradcast_required) {
      tim::vx::TensorSpec spec = inputs[0]->GetSpec();
      spec = spec.AsTransientSpec();
      auto broadcast_out = delegate->GetGraph()->CreateTensor(spec);
      if (inputs[0]->GetShape().size() > inputs[1]->GetShape().size()) {
        std::vector<uint32_t> shape = inputs[0]->GetShape();
        std::vector<int32> broadcast_param;
        for (auto iter = shape.begin(); iter != shape.end(); iter++) {
          broadcast_param.push_back(*iter);
        }
        auto op_broadcast =
            delegate->GetGraph()->CreateOperation<tim::vx::ops::Broadcast>(
                broadcast_param);
        (*op_broadcast).BindInput(inputs[1]).BindOutput(broadcast_out);
        elementwise_inputs.push_back(inputs[0]);
        elementwise_inputs.push_back(broadcast_out);
      } else {
        std::vector<uint32_t> shape = inputs[1]->GetShape();
        std::vector<int32> broadcast_param;
        for (auto iter = shape.begin(); iter != shape.end(); iter++) {
          broadcast_param.push_back(*iter);
        }
        auto op_broadcast =
            delegate->GetGraph()->CreateOperation<tim::vx::ops::Broadcast>(
                broadcast_param);
        (*op_broadcast).BindInput(inputs[0]).BindOutput(broadcast_out);
        elementwise_inputs.push_back(broadcast_out);
        elementwise_inputs.push_back(inputs[1]);
      }
    }
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Minimum>();
    boradcast_required ? (*op).BindInputs(elementwise_inputs)
                       : (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);
    delegate->GetOps().push_back(std::move(op));
    return true;
  }
};

struct MaximumMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Maximum op");

    bool boradcast_required = (inputs[0]->GetShape() != inputs[1]->GetShape());
    std::vector<std::shared_ptr<tim::vx::Tensor>> elementwise_inputs;
    if (boradcast_required) {
      tim::vx::TensorSpec spec = inputs[0]->GetSpec();
      spec = spec.AsTransientSpec();
      auto broadcast_out = delegate->GetGraph()->CreateTensor(spec);
      if (inputs[0]->GetShape().size() > inputs[1]->GetShape().size()) {
        std::vector<uint32_t> shape = inputs[0]->GetShape();
        std::vector<int32> broadcast_param;
        for (auto iter = shape.begin(); iter != shape.end(); iter++) {
          broadcast_param.push_back(*iter);
        }
        auto op_broadcast =
            delegate->GetGraph()->CreateOperation<tim::vx::ops::Broadcast>(
                broadcast_param);
        (*op_broadcast).BindInput(inputs[1]).BindOutput(broadcast_out);
        elementwise_inputs.push_back(inputs[0]);
        elementwise_inputs.push_back(broadcast_out);
      } else {
        std::vector<uint32_t> shape = inputs[1]->GetShape();
        std::vector<int32> broadcast_param;
        for (auto iter = shape.begin(); iter != shape.end(); iter++) {
          broadcast_param.push_back(*iter);
        }
        auto op_broadcast =
            delegate->GetGraph()->CreateOperation<tim::vx::ops::Broadcast>(
                broadcast_param);
        (*op_broadcast).BindInput(inputs[0]).BindOutput(broadcast_out);
        elementwise_inputs.push_back(broadcast_out);
        elementwise_inputs.push_back(inputs[1]);
      }
    }
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Maximum>();
    boradcast_required ? (*op).BindInputs(elementwise_inputs)
                       : (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);
    delegate->GetOps().push_back(std::move(op));
    return true;
  }
};

template <typename T_OperationType>
struct PowMapper : public SimpleOpMapper<T_OperationType> {

  PowMapper(std::string name) : SimpleOpMapper<T_OperationType>(name) {}

  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    auto input_tensor0 = context->tensors[node->inputs->data[0]];
    auto input_tensor1 = context->tensors[node->inputs->data[1]];
    if (input_tensor0.type == kTfLiteInt32 &&
        input_tensor1.type == kTfLiteInt32) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "I32 input/I32 output is not supported in pow.");
      return false;
    }
    return true;
  }
};

template <typename T_OperationType>
struct DequantizeMapper : public SimpleOpMapper<T_OperationType> {

  DequantizeMapper(std::string name) : SimpleOpMapper<T_OperationType>(name) {}

  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto output_tensor = context->tensors[node->outputs->data[0]];
    const TfLiteAffineQuantization* params =
        reinterpret_cast<const TfLiteAffineQuantization*>(
            input_tensor.quantization.params);
    if ((input_tensor.type == kTfLiteInt16 ||
         input_tensor.type == kTfLiteFloat16) &&
        output_tensor.type == kTfLiteFloat32 &&
        input_tensor.quantization.type == kTfLiteAffineQuantization) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "ASYM I16/F16 input / F32 output is not supported");
      return false;
    }
    if ((input_tensor.type == kTfLiteInt8 ||
         input_tensor.type == kTfLiteUInt8) &&
        output_tensor.type == kTfLiteFloat32 &&
        input_tensor.quantization.type == kTfLiteAffineQuantization &&
        params->scale->size>1) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "SYMM PerChannel I8/U8 input / F32 output is not supported");
      return false;
    }
    return true;
  }
};

template <typename T_OperationType>
struct QuantizeMapper : public SimpleOpMapper<T_OperationType> {

  QuantizeMapper(std::string name) : SimpleOpMapper<T_OperationType>(name) {}

  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto output_tensor = context->tensors[node->outputs->data[0]];
    const TfLiteAffineQuantization* params =
        reinterpret_cast<const TfLiteAffineQuantization*>(
            output_tensor.quantization.params);
    if (input_tensor.type == kTfLiteInt32 &&
        (output_tensor.type == kTfLiteUInt8||output_tensor.type == kTfLiteInt8) &&
        input_tensor.quantization.type == kTfLiteAffineQuantization) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "ASYM I16 input / ASYM U8/ASYM I8 output is not supported");
      return false;
    }
    if (input_tensor.type == kTfLiteInt16  &&
        (output_tensor.type == kTfLiteUInt8||output_tensor.type == kTfLiteInt8) &&
        input_tensor.quantization.type == kTfLiteAffineQuantization) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "ASYM I32 input / ASYM U8/ASYM I8 output is not supported");
      return false;
    }
    if (input_tensor.type == kTfLiteInt16  &&
        output_tensor.type == kTfLiteInt32 &&
        input_tensor.quantization.type == kTfLiteAffineQuantization) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "ASYM I16 input / ASYM I32 output is not supported");
      return false;
    }
    if (input_tensor.type == kTfLiteFloat32  &&
        (output_tensor.type == kTfLiteUInt8||output_tensor.type == kTfLiteInt8) &&
        output_tensor.quantization.type == kTfLiteAffineQuantization &&
        params->scale->size>1) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "F32 input / SYMM PerChannel I8/U8 output is not supported");
      return false;
    }
    return true;
  }
};

template <typename T_OperationType, typename T_Param>
struct SimpleOpWithFusedActiovationMapper
    : public OpMapperBase<T_Param, FusedActivationAction<0, T_Param>> {
  std::string name_;

  SimpleOpWithFusedActiovationMapper(std::string name) : name_(name) {}

  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    const auto builtin = reinterpret_cast<const T_Param*>(node->builtin_data);
    auto in_tensor0 = context->tensors[node->inputs->data[0]];
    auto out_tensor0 = context->tensors[node->outputs->data[0]];
    if (in_tensor0.type == kTfLiteInt16 &&
        (out_tensor0.type == kTfLiteUInt8 || out_tensor0.type == kTfLiteInt8) &&
        in_tensor0.quantization.type == kTfLiteAffineQuantization &&
        out_tensor0.quantization.type == kTfLiteAffineQuantization) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "ASYM I16 input0 / ASYM U8/I8 output is not supported");
      return false;
    }
    if (builtin->activation == kTfLiteActReluN1To1 &&
        context->tensors[node->inputs->data[0]].type == kTfLiteInt32 &&
        context->tensors[node->outputs->data[0]].type == kTfLiteInt32) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "I32 input/I32 output is not supported in Relu1.");
      return false;
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating %s op", name_.c_str());

    bool boradcast_required = (inputs[0]->GetShape() != inputs[1]->GetShape());
    std::vector<std::shared_ptr<tim::vx::Tensor>> elementwise_inputs;
    if (boradcast_required) {
      tim::vx::TensorSpec spec = inputs[0]->GetSpec();
      spec = spec.AsTransientSpec();
      auto broadcast_out = delegate->GetGraph()->CreateTensor(spec);
      if (inputs[0]->GetShape().size() > inputs[1]->GetShape().size()) {
        std::vector<uint32_t> shape = inputs[0]->GetShape();
        std::vector<int32> broadcast_param;
        for (auto iter = shape.begin(); iter != shape.end(); iter++) {
          broadcast_param.push_back(*iter);
        }
        auto op_broadcast =
            delegate->GetGraph()->CreateOperation<tim::vx::ops::Broadcast>(
                broadcast_param);
        (*op_broadcast).BindInput(inputs[1]).BindOutput(broadcast_out);
        elementwise_inputs.push_back(inputs[0]);
        elementwise_inputs.push_back(broadcast_out);
      } else {
        std::vector<uint32_t> shape = inputs[1]->GetShape();
        std::vector<int32> broadcast_param;
        for (auto iter = shape.begin(); iter != shape.end(); iter++) {
          broadcast_param.push_back(*iter);
        }
        auto op_broadcast =
            delegate->GetGraph()->CreateOperation<tim::vx::ops::Broadcast>(
                broadcast_param);
        (*op_broadcast).BindInput(inputs[0]).BindOutput(broadcast_out);
        elementwise_inputs.push_back(broadcast_out);
        elementwise_inputs.push_back(inputs[1]);
      }
    }

    auto op = delegate->GetGraph()->CreateOperation<T_OperationType>();
    boradcast_required ? (*op).BindInputs(elementwise_inputs)
                   : (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);
    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

template <typename T_Param>
struct Conv2dKind
    : public OpMapperBase<T_Param, FusedActivationAction<0, T_Param>> {};

struct FullyConnectedMapper
    : public OpMapperBase<
          TfLiteFullyConnectedParams,
          FusedActivationAction<0, TfLiteFullyConnectedParams>> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    const auto builtin =
        reinterpret_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto weight_tensor = context->tensors[node->inputs->data[1]];

    if (input_tensor.type != weight_tensor.type) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "hybrid data type is not supported in fullyconnected.");
      return false;
    }
    if (builtin->weights_format ==
        kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Shuffled weight is not supported");
      return false;
    }
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (input_index >= 0 && input_index < context->tensors_size &&
          context->tensors[input_index].type == kTfLiteInt16) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Int16 input is not supported");
        return false;
      }
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating fully connected op");
    const auto builtin =
        reinterpret_cast<const TfLiteFullyConnectedParams*>(params);
    auto input_tensor = inputs[0];
    auto weight_tensor = inputs[1];

    if (input_tensor->GetShape().size() > 2 ||
        (input_tensor->GetShape().size() == 2 &&
         input_tensor->GetShape()[0] != weight_tensor->GetShape()[0])) {
      uint32_t input_size = weight_tensor->GetShape()[0];
      uint32_t total_input_size = 1;
      for (int i = 0; i < input_tensor->GetShape().size(); i++) {
        total_input_size *= input_tensor->GetShape()[i];
      }
      uint32_t input_batch = total_input_size / input_size;
      auto reshape_output = delegate->GetGraph()->CreateTensor(
          input_tensor->GetSpec().AsTransientSpec());
      std::vector<uint32_t> new_shape{input_size, input_batch};
      auto reshape_op =
          delegate->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(
              new_shape);
      (*reshape_op).BindInput(inputs[0]);
      (*reshape_op).BindOutput(reshape_output);
      delegate->GetOps().push_back(reshape_op);
      inputs[0] = reshape_output;
    }

    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::FullyConnected>(
            0, weight_tensor->GetShape()[1]);
    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct SoftmaxMapper : public OpMapperBase<TfLiteSoftmaxParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating softmax op");
    auto builtin = reinterpret_cast<const TfLiteSoftmaxParams*>(params);
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Softmax>(
        builtin->beta, 0);
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Conv2dMapper : public Conv2dKind<TfLiteConvParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto weight_tensor = context->tensors[node->inputs->data[1]];

    if (input_tensor.type != weight_tensor.type) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "hybrid data type is not supported in conv2d.");
      return false;
    }
    bool is_grouped = (input_tensor.dims->data[3] != weight_tensor.dims->data[3]);
    bool is_batched = (input_tensor.dims->data[0] != 1);
    if (is_grouped && is_batched) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "batch is not supported in grouped conv2d.");
      return false;
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    // input layout CWHN, weight layout IWHO
    uint32_t groups = inputs[0]->GetShape()[0] / inputs[1]->GetShape()[0];
    uint32_t weights = inputs[1]->GetShape()[3];
    uint32_t kernel_h = inputs[1]->GetShape()[2];
    uint32_t kernel_w = inputs[1]->GetShape()[1];
    const auto builtin = reinterpret_cast<const TfLiteConvParams*>(params);
    std::shared_ptr<tim::vx::Operation> op;
    if (inputs[0]->GetShape()[0] == inputs[1]->GetShape()[0]) {
      TFLITE_LOG(TFLITE_LOG_INFO, "Creating Conv2d op");
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
          static_cast<int32_t>(weights),
          TflitePadTypeToVsiPadType(builtin->padding),
          std::array<uint32_t, 2>({kernel_w, kernel_h}),
          std::array<uint32_t, 2>(
              {builtin->stride_width, builtin->stride_height}),
          std::array<uint32_t, 2>({builtin->dilation_width_factor,
                                   builtin->dilation_height_factor}),
          0,
          tim::vx::DataLayout::CWHN);
    } else {
      TFLITE_LOG(TFLITE_LOG_INFO, "Creating Grouped Conv2d op");
      op = delegate->GetGraph()->CreateOperation<tim::vx::ops::GroupedConv2d>(
          TflitePadTypeToVsiPadType(builtin->padding),
          std::array<uint32_t, 2>(
              {builtin->stride_width, builtin->stride_height}),
          std::array<uint32_t, 2>({builtin->dilation_width_factor,
                                   builtin->dilation_height_factor}),
          groups,
          tim::vx::DataLayout::CWHN);
    }

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct TransposeConvMapper : public OpMapperBase<TfLiteTransposeConvParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    auto output_tensor = context->tensors[node->outputs->data[0]];

    if (0 == output_tensor.dims->size) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "transpose conv cannot support dynamic shape");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create TransposeConv op");
    const auto builtin =
        reinterpret_cast<const TfLiteTransposeConvParams*>(params);
    auto padding = TflitePadTypeToVsiPadType(builtin->padding);
    auto stride_width = builtin->stride_width;
    auto stride_height = builtin->stride_height;

    std::vector<int32_t> output_shape(inputs[0]->GetShape()[0]);
    inputs[0]->CopyDataFromTensor(output_shape.data());

    uint32_t input_width = inputs[2]->GetShape()[1];
    uint32_t input_height = inputs[2]->GetShape()[2];
    uint32_t ksize_width = inputs[1]->GetShape()[1];
    uint32_t ksize_height = inputs[1]->GetShape()[2];
    uint32_t weights = inputs[1]->GetShape()[3];
    int32_t pad_left_inter = static_cast<int32_t>(
        ksize_width + stride_width * (input_width - 1) - output_shape[2]);
    uint32_t pad_left = pad_left_inter / 2;
    uint32_t pad_right = pad_left_inter - pad_left;
    int32_t pad_top_inter = static_cast<int32_t>(
        ksize_height + stride_height * (input_height - 1) - output_shape[1]);
    uint32_t pad_top = pad_top_inter / 2;
    uint32_t pad_bottom = pad_top_inter - pad_top;
    std::array<uint32_t, 2> ksize{ksize_width, ksize_height};
    std::array<uint32_t, 2> stride{stride_width, stride_height};
    std::array<uint32_t, 2> output_padding{0, 0};
    std::array<uint32_t, 4> pad{pad_left, pad_right, pad_top, pad_bottom};

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::DeConv2d>(
        weights,
        padding,
        ksize,
        stride,
        output_padding,
        pad,
        1,
        tim::vx::DataLayout::CWHN,
        tim::vx::DataLayout::IcWHOc);

    std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensor;
    input_tensor.push_back(inputs[2]);
    input_tensor.push_back(inputs[1]);
    if (inputs.size() == 4) {
      input_tensor.push_back(inputs[3]);
    }
    (*op).BindInputs(input_tensor);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));
    return true;
  }
};

template <tim::vx::PoolType poolType>
struct Pool2dMapper : public Conv2dKind<TfLitePoolParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(
        TFLITE_LOG_INFO, "Creating Pool2d(%d) op", static_cast<int>(poolType));
    const auto builtin = reinterpret_cast<const TfLitePoolParams*>(params);

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Pool2d>(
        poolType,
        TflitePadTypeToVsiPadType(builtin->padding),
        std::array<uint32_t, 2>(
            {builtin->filter_width, builtin->filter_height}),
        std::array<uint32_t, 2>(
            {builtin->stride_width, builtin->stride_height}),
        tim::vx::RoundType::FLOOR,
        tim::vx::DataLayout::CWHN);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct DepthwiseConv2dMapper : public Conv2dKind<TfLiteDepthwiseConvParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto weight_tensor = context->tensors[node->inputs->data[1]];

    if (input_tensor.type != weight_tensor.type) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "hybrid data type is not supported in DepthwiseConv2d.");
      return false;
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating DepthwiseConv2d op");
    const auto builtin =
        reinterpret_cast<const TfLiteDepthwiseConvParams*>(params);

    uint32_t weights = inputs[1]->GetShape()[0];
    uint32_t kernel_h = inputs[1]->GetShape()[2];
    uint32_t kernel_w = inputs[1]->GetShape()[1];

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
        static_cast<int32_t>(weights),
        TflitePadTypeToVsiPadType(builtin->padding),
        std::array<uint32_t, 2>({kernel_w, kernel_h}),
        std::array<uint32_t, 2>(
            {builtin->stride_width, builtin->stride_height}),
        std::array<uint32_t, 2>(
            {builtin->dilation_width_factor, builtin->dilation_height_factor}),
        builtin->depth_multiplier,
        tim::vx::DataLayout::CWHN);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct ConcatenationMapper
    : public OpMapperBase<TfLiteConcatenationParams,
                          FusedActivationAction<0, TfLiteConcatenationParams>> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Concatenation op");
    const auto builtin =
        reinterpret_cast<const TfLiteConcatenationParams*>(params);
    auto output_tensor = outputs[0];

    auto axis = vx::delegate::utils::ConvertAxis(builtin->axis,
                                                 inputs[0]->GetShape().size());

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Concat>(
        axis, inputs.size());

    // If input from dynamic graph, tensor may have shape with 0
    std::vector<std::shared_ptr<tim::vx::Tensor>> none_zero_inputs;

    for(const auto& in : inputs) {
      if (in->GetSpec().GetElementNum() != 0) {
        none_zero_inputs.push_back(in);
      } else {
        TFLITE_LOG(TFLITE_LOG_INFO, "Remove zero sized tensor from concat's input list");
      }
    }

    (*op).BindInputs(none_zero_inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct LocalResponseNormalizationMapper
    : public OpMapperBase<TfLiteLocalResponseNormParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating LRN op");
    const auto builtin =
        reinterpret_cast<const TfLiteLocalResponseNormParams*>(params);
    int size = builtin->radius * 2 + 1; // radius is the half of normalization window
    auto op = delegate->GetGraph()
                  ->CreateOperation<tim::vx::ops::LocalResponseNormalization>(
                      size, builtin->alpha, builtin->beta, builtin->bias, 0);
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct L2NormalizationMapper
    : public OpMapperBase<TfLiteL2NormParams,
                          FusedActivationAction<0, TfLiteL2NormParams>> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating L2Normaliztion op");
    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::L2Normalization>(0);

    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct ReshapeMapper : public OpMapperBase<TfLiteReshapeParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto output_index = node->outputs->data[0];

    if (context->tensors[output_index].dims->size == 0) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "dynamic shape in not support in reshape.");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Reshape op");
    const auto builtin = reinterpret_cast<const TfLiteReshapeParams*>(params);
    std::vector<int32_t> new_shape;
    uint32_t total_size = 1, negative_index = 0;
    std::vector<uint32_t> no_nagetive_shape;
    bool do_shape_inference = false;

    // The new shape may be passed to reshape op by
    // builtin prarameters or inputs[1], the two formats should be handled.
    if (inputs.size() == 2 &&
        inputs[1]->GetDataType() == tim::vx::DataType::INT32 &&
        inputs[1]->GetShape().size() == 1 &&
        inputs[1]->GetSpec().attr_ == tim::vx::TensorAttribute::CONSTANT) {
      std::vector<int32_t> shape_from_input1(inputs[1]->GetShape()[0]);
      inputs[1]->CopyDataFromTensor(shape_from_input1.data());
      new_shape.assign(shape_from_input1.rbegin(), shape_from_input1.rend());
    } else {
      for (int i = builtin->num_dimensions - 1; i >= 0; i--) {
        new_shape.push_back(static_cast<int32_t>(builtin->shape[i]));
      }
    }

    for (uint32_t i = 0; i < inputs[0]->GetShape().size(); ++i) {
      total_size *= inputs[0]->GetShape().at(i);
    }
    for (uint32_t i = 0; i < new_shape.size(); ++i) {
      if (new_shape.at(i) != -1) {
        total_size /= new_shape.at(i);
        no_nagetive_shape.push_back(new_shape.at(i));
      } else {
        do_shape_inference = true;
        negative_index = i;
        no_nagetive_shape.push_back(0);  // hold a place for changes to the
                                         // value
      }
    }
    if (do_shape_inference) {
      no_nagetive_shape.at(negative_index) = total_size;
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(
        no_nagetive_shape);
    (*op).BindInput(inputs[0]);
    (*op).BindOutput(outputs[0]);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct StridedSliceMapper : public OpMapperBase<TfLiteStridedSliceParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    TFLITE_LOG(TFLITE_LOG_INFO, "Check  StridedSlice");
    const auto builtin =
        reinterpret_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
    auto begin_tensor = context->tensors[node->inputs->data[1]];
    auto end_tensor = context->tensors[node->inputs->data[2]];
    auto strides_tensor = context->tensors[node->inputs->data[3]];
    if (begin_tensor.allocation_type != kTfLiteMmapRo ||
        end_tensor.allocation_type != kTfLiteMmapRo ||
        strides_tensor.allocation_type != kTfLiteMmapRo){
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "begin_tensor, end_tensor and strides_tensor must be constant.");
      return false;
    }
      if (builtin->new_axis_mask) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "new_axis_mask > 0 is not supported");
        return false;
      }
    if (builtin->ellipsis_mask) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "ellipsis_mask > 0 is not supported");
      return false;
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating StridedSlice op");
    const auto builtin =
        reinterpret_cast<const TfLiteStridedSliceParams*>(params);
    auto input_tensor = inputs[0];
    auto begin_tensor = inputs[1];
    auto end_tensor = inputs[2];
    auto strides_tensor = inputs[3];
    auto output_tensor = outputs[0];
    auto const& input_shape = input_tensor->GetShape();
    int begin_mask = builtin->begin_mask;
    int end_mask = builtin->end_mask;
    int ellipsis_mask = builtin->ellipsis_mask;
    int shrink_axis_mask = builtin->shrink_axis_mask;

    std::vector<int32_t> begin_dims(begin_tensor->GetShape()[0]);
    begin_tensor->CopyDataFromTensor(begin_dims.data());
    for (size_t i = 0; i < begin_dims.size(); i++) {
      if (begin_mask & (1 << i)) {
        begin_dims[i] = -1;
      }
    }
    std::reverse(begin_dims.begin(), begin_dims.end());

    std::vector<int32_t> end_dims(end_tensor->GetShape()[0]);
    int32_t end_pos = 1 + std::accumulate(input_shape.begin(),
                                          input_shape.end(),
                                          0,
                                          [](int32_t lhs, int32_t rhs) {
                                            return std::max(lhs, rhs);
                                          });
    end_tensor->CopyDataFromTensor(end_dims.data());
    for (size_t i = 0; i < end_dims.size(); i++) {
      if (end_mask & (1 << i)) {
        end_dims[i] = end_pos;
      }
    }
    std::reverse(end_dims.begin(), end_dims.end());

    std::vector<int32_t> strides_dims(strides_tensor->GetShape()[0]);
    strides_tensor->CopyDataFromTensor(strides_dims.data());
    std::reverse(strides_dims.begin(), strides_dims.end());

    if (ellipsis_mask) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "ellipsis_mask > 0 is not supported");
    } else {
      size_t i = begin_dims.size();
      for (; i < input_shape.size(); i++) {
        begin_dims.insert(begin_dims.begin(), -1);
      }
      i = end_dims.size();
      for (; i < input_shape.size(); i++) {
        end_dims.insert(end_dims.begin(), end_pos);
      }
      i = strides_dims.size();
      for (; i < input_shape.size(); i++) {
        strides_dims.insert(strides_dims.begin(), -1);
      }
    }

    for (size_t i = 0; i < begin_dims.size(); i++) {
      begin_dims[i] = begin_dims[i] == -1 ? 0 : begin_dims[i];
    }

    for (size_t i = 0; i < end_dims.size(); i++) {
      end_dims[i] = end_dims[i] == end_pos ? input_shape[i] : end_dims[i];
      end_dims[i] = end_dims[i] > static_cast<int32_t>(input_shape[i])
                        ? input_shape[i]
                        : end_dims[i];
    }

    for (size_t i = 0; i < strides_dims.size(); i++) {
      strides_dims[i] = strides_dims[i] == -1 ? 1 : strides_dims[i];
    }

    if (begin_mask) {
      int32_t t = 0;
      int32_t input_dim = input_shape.size();
      for (size_t i = 0; i < input_dim; i++) {
        if (begin_mask & (1 << i)) {
          t = t | (1 << (input_dim - i - 1));
        }
      }
      begin_mask = t;
    }
    if (shrink_axis_mask) {
      int32_t t = 0;
      int32_t input_dim = input_shape.size();
      for (size_t i = 0; i < input_dim; i++) {
        if (shrink_axis_mask & (1 << i)) {
          t = t | (1 << (input_dim - i - 1));
        }
      }
      shrink_axis_mask = t;
    }
    if (end_mask) {
      int32_t t = 0;
      int32_t input_dim = input_shape.size();
      for (size_t i = 0; i < input_dim; i++) {
        if (end_mask & (1 << i)) {
          t = t | (1 << (input_dim - i - 1));
        }
      }
      end_mask = t;
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::StridedSlice>(
        begin_dims,
        end_dims,
        strides_dims,
        begin_mask,
        end_mask,
        shrink_axis_mask);
    (*op).BindInput(input_tensor);
    (*op).BindOutput(output_tensor);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct PadMapper : public OpMapperBase<EmptyStructPlaceholder> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {

    if(0 == context->tensors[node->outputs->data[0]].dims->size){
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "Pad cannot support dynamic shape");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Pad op");
    auto padding = inputs[1];
    std::vector<uint32_t> padding_shape = padding->GetShape();
    uint32_t pad = 1;
    for (auto s : padding_shape) {
      pad *= s;
    }
    std::vector<uint32_t> pad_size(pad);
    padding->CopyDataFromTensor(pad_size.data());
    std::vector<uint32_t> front_size;
    std::vector<uint32_t> back_size;
    for (int i = pad_size.size() - 1; i >= 0; i -= 2) {
      back_size.push_back(pad_size[i]);
      front_size.push_back(pad_size[i - 1]);
    }
    int32_t val = 0;
    if (inputs.size() > 2) {
      auto pad_value = inputs[2];
      if (!pad_value->IsPlaceHolder()) {
        pad_value->CopyDataFromTensor(&val);
      }
    }
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Pad>(
        front_size, back_size, val);
    (*op).BindInput(inputs[0]).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

using AddMapper =
    SimpleOpWithFusedActiovationMapper<tim::vx::ops::Add, TfLiteAddParams>;
using SubMapper =
    SimpleOpWithFusedActiovationMapper<tim::vx::ops::Sub, TfLiteSubParams>;
using DivMapper =
    SimpleOpWithFusedActiovationMapper<tim::vx::ops::Div, TfLiteDivParams>;
using MulMapper =
    SimpleOpWithFusedActiovationMapper<tim::vx::ops::Multiply, TfLiteMulParams>;

template <tim::vx::ResizeType resizeType>
struct ResizeMapper : public OpMapperBase<TfLiteResizeNearestNeighborParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    TFLITE_LOG(
        TFLITE_LOG_INFO, "Check Resize(%d)", static_cast<int>(resizeType));

    int input_index = node->inputs->data[0];
    int shape_index = node->inputs->data[1];
    if (context->tensors[shape_index].allocation_type != kTfLiteMmapRo) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "shape tensor must be constant.");
      return false;
    }
    if ((context->tensors[input_index].type == kTfLiteInt8 ||
         context->tensors[input_index].type == kTfLiteUInt8) &&
        context->tensors[input_index].quantization.type ==
            kTfLiteNoQuantization) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "Int8 or uint8 input without quantization is not supported in "
          "Resize");
      return false;
    }

    int size_tensor_idx = node->inputs->data[1];
    const uint8_t* tensor_data = reinterpret_cast<const uint8_t*>(
        context->tensors[size_tensor_idx].data.raw_const);
    return tensor_data != nullptr;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(
        TFLITE_LOG_INFO, "Creating Resize(%d)", static_cast<int>(resizeType));
    auto input_shape = inputs[0]->GetShape();
    uint32_t resize_rank = inputs[1]->GetShape()[0];
    std::vector<int32_t> output_shape(resize_rank);
    inputs[1]->CopyDataFromTensor(output_shape.data());

    int32_t channel = input_shape[0];
    int32_t scale_w = output_shape[1] / input_shape[1];
    int32_t scale_h = output_shape[0] / input_shape[2];

    bool is_scale_integer = !((output_shape[1] % input_shape[1]) ||
                              (output_shape[0] % input_shape[2]));
    // turn off optimization by default.
    bool enable_bilinear = false;
    bool can_resize_to_transposeconv = false;
    // is_scale_integer &&
    // ((enable_bilinear && resizeType == tim::vx::ResizeType::BILINEAR) ||
    //  (resizeType == tim::vx::ResizeType::NEAREST_NEIGHBOR));

    if (can_resize_to_transposeconv) {
      return ResizeToTransposeConv(
          delegate, inputs, outputs, resizeType, channel, scale_w, scale_h);
    }
    const auto builtin =
        reinterpret_cast<const TfLiteResizeNearestNeighborParams*>(params);
    auto size_tensor = inputs[1];

    std::vector<int> size(size_tensor->GetShape()[0]);
    size_tensor->CopyDataFromTensor(size.data());

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Resize>(
        resizeType,
        0.0f,
        builtin->align_corners,
        builtin->half_pixel_centers,
        size[0],
        size[1],
        tim::vx::DataLayout::CWHN);

    (*op).BindInput(inputs[0]);
    (*op).BindOutput(outputs[0]);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct AddNMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating AddN op");
    auto output_tensor = outputs[0];
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::AddN>(
        inputs.size());

    (*op).BindInputs(inputs);
    (*op).BindOutput(output_tensor);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct SplitMapper : public OpMapperBase<TfLiteSplitParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    TfLiteTensor output_tensor = context->tensors[node->outputs->data[0]];
    if (output_tensor.allocation_type == kTfLiteDynamic) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "dynamic shpae is not supported in split.");
      return false;
    }
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if ((context->tensors[input_index].type == kTfLiteInt8 ||
           context->tensors[input_index].type == kTfLiteUInt8) &&
          context->tensors[input_index].quantization.type ==
              kTfLiteNoQuantization) {
        TFLITE_LOG_PROD(
            TFLITE_LOG_ERROR,
            "Int8 or uint8 input without quantization is not supported in "
            "Split");
        return false;
      }
    }

    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Split op");
    auto axis_tensor = inputs[0];
    auto input_tensor = inputs[1];

    int32_t axis = 0;
    axis_tensor->CopyDataFromTensor(&axis);
    axis =
        vx::delegate::utils::ConvertAxis(axis, input_tensor->GetShape().size());

    std::vector<uint32_t> slices;
    for (auto& o : outputs) {
      slices.push_back(o->GetShape()[axis]);
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Split>(
        axis, slices);

    (*op).BindInput(input_tensor);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct SqueezeMapper : public OpMapperBase<TfLiteSqueezeParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Squeeze op");
    auto input_shape = inputs[0]->GetShape();
    const auto builtin = reinterpret_cast<const TfLiteSqueezeParams*>(params);
    std::vector<uint32_t> vx_axis(builtin->num_squeeze_dims);
    if (builtin->num_squeeze_dims != 0) {
      for (int i = 0; i < builtin->num_squeeze_dims; ++i) {
        vx_axis[i] = vx::delegate::utils::ConvertAxis(builtin->squeeze_dims[i],
                                                      input_shape.size());
      }
    } else {  // tim-vx always needs axis.
      for (int i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] == 1) {
          vx_axis.push_back(i);
        }
      }
    }

    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::Squeeze>(vx_axis);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Space2DepthMapper : public OpMapperBase<TfLiteSpaceToDepthParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (context->tensors[input_index].type == kTfLiteInt32) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                        "Int32 input is not supported in Space2Depth");
        return false;
      }
      if (context->tensors[input_index].type == kTfLiteInt64) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                        "Int64 input is not supported in Space2Depth");
        return false;
      }
      if ((context->tensors[input_index].type == kTfLiteInt8 ||
           context->tensors[input_index].type == kTfLiteUInt8) &&
          context->tensors[input_index].quantization.type ==
              kTfLiteNoQuantization) {
        TFLITE_LOG_PROD(
            TFLITE_LOG_ERROR,
            "Int8 or uint8 input without quantization is not supported in "
            "Space2Depth");
        return false;
      }
    }

    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create SpaceToDepth op");
    const auto builtin =
        reinterpret_cast<const TfLiteSpaceToDepthParams*>(params);

    std::vector<int> block({builtin->block_size, builtin->block_size});
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::SpaceToDepth>(
        block, tim::vx::DataLayout::CWHN);
    (*op).BindInput(inputs[0]);
    (*op).BindOutput(outputs[0]);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Depth2SpaceMapper : public OpMapperBase<TfLiteDepthToSpaceParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (context->tensors[input_index].type == kTfLiteInt32) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                        "Int32 input is not supported in Depth2Space");
        return false;
      }
      if (context->tensors[input_index].type == kTfLiteInt64) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                        "Int64 input is not supported in Depth2Space");
        return false;
      }
      if ((context->tensors[input_index].type == kTfLiteInt8 ||
           context->tensors[input_index].type == kTfLiteUInt8) &&
          context->tensors[input_index].quantization.type ==
              kTfLiteNoQuantization) {
        TFLITE_LOG_PROD(
            TFLITE_LOG_ERROR,
            "Int8 or uint8 input without quantization is not supported in "
            "Depth2Space");
        return false;
      }
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create DepthToSpace op");
    const auto builtin =
        reinterpret_cast<const TfLiteDepthToSpaceParams*>(params);

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::DepthToSpace>(
        builtin->block_size, tim::vx::DataLayout::CWHN);

    (*op).BindInput(inputs[0]);
    (*op).BindOutput(outputs[0]);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct PreluMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create Prelu op");
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Prelu>(0);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Transpose : public OpMapperBase<TfLiteTransposeParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    int output_index = node->outputs->data[0];
    if (context->tensors[output_index].dims->size == 0){
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                        "Dynamic shape is not supported in transpose");
        return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create Transpose op");
    auto perm_tensor = inputs[1];
    std::vector<uint32_t> perm(perm_tensor->GetShape()[0]);
    perm_tensor->CopyDataFromTensor(perm.data());
    std::vector<uint32_t> ovx_perm =
        vx::delegate::utils::GetOvxTransposePerm(perm);
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Transpose>(
        ovx_perm);

    (*op).BindInput(inputs[0]);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct BatchMatmul : public OpMapperBase<TfLiteBatchMatMulParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    const auto builtin =
        reinterpret_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);
    bool adj_x = builtin->adj_x;
    bool adj_y = builtin->adj_y;
    if (context->tensors[node->outputs->data[0]].type == kTfLiteInt32)  {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "I32 outputs type is not supported in BatchMatmul");
      return false;
    }
    if ((context->tensors[node->inputs->data[0]].type == kTfLiteInt8 ||
         context->tensors[node->inputs->data[1]].type == kTfLiteInt8) ||
        (context->tensors[node->inputs->data[1]].type == kTfLiteFloat32 &&
         context->tensors[node->inputs->data[0]].type == kTfLiteInt8)) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "F32/I8 inputs type is not supported in BatchMatmul");
      return false;
    }
    if (adj_x && adj_y) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_ERROR,
          "Does not support adj_x and adj_y being true at the same time");
      return false;
    }

    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create BatchMatmul op");
    const auto builtin =
        reinterpret_cast<const TfLiteBatchMatMulParams*>(params);
    bool adj_x = builtin->adj_x;
    bool adj_y = builtin->adj_y;

    std::vector<std::vector<uint32_t>> in_shape = {inputs[0]->GetShape(),
                                                   inputs[1]->GetShape()};
    bool boradcast_required = false;

    // Need broadcast or not
    if (in_shape[0].size() != in_shape[1].size()) {
      boradcast_required = true;
    } else {
      for (int i = 2; i < in_shape[0].size(); ++i) {
        if (in_shape[0][i] != in_shape[1][i]) {
          boradcast_required = true;
        }
      }
    }

    std::vector<std::shared_ptr<tim::vx::Tensor>> broadcast_out;
    if (boradcast_required) {
      int out_cnt;
      auto dim_iter0 = in_shape[0].begin();
      auto dim_iter1 = in_shape[1].begin();
      // Minimum 2 dimensions do not require broadcast
      dim_iter0 += 2;
      dim_iter1 += 2;
      std::vector<std::vector<uint32_t>> out_shape = {
          {in_shape[0][0], in_shape[0][1]}, {in_shape[1][0], in_shape[1][1]}};
      while (1) {
        if (dim_iter0 != in_shape[0].end() && dim_iter1 != in_shape[1].end()) {
          out_shape[0].push_back(std::max(*dim_iter1, *dim_iter0));
          out_shape[1].push_back(std::max(*dim_iter1, *dim_iter0));
        } else {
          if (in_shape[0].size() > in_shape[1].size()) {
            out_shape[0].push_back(*dim_iter0);
            out_shape[1].push_back(*dim_iter0);
          } else {
            out_shape[0].push_back(*dim_iter1);
            out_shape[1].push_back(*dim_iter1);
          }
        }
        if (dim_iter0 != in_shape[0].end()) dim_iter0++;
        if (dim_iter1 != in_shape[1].end()) dim_iter1++;

        if (dim_iter0 == in_shape[0].end() && dim_iter1 == in_shape[1].end()) {
          break;
        }
      }
      for (int i = 0; i < inputs.size(); ++i) {
        if (out_shape[i] != in_shape[i]) {
          tim::vx::TensorSpec spec = inputs[i]->GetSpec();
          spec = spec.AsTransientSpec();
          broadcast_out.push_back(delegate->GetGraph()->CreateTensor(spec));
          std::vector<int32>
              broadcast_param;  // for Broadcast constructor parameters
          for (auto iter = out_shape[i].begin(); iter != out_shape[i].end();
               iter++) {
            broadcast_param.push_back(*iter);
          }
          auto op_broadcast =
              delegate->GetGraph()->CreateOperation<tim::vx::ops::Broadcast>(
                  broadcast_param);
          (*op_broadcast).BindInput(inputs[i]).BindOutput(broadcast_out[i]);
        } else {
          broadcast_out.push_back(inputs[i]);
        }
      }
    }
    // adj_x & adj_y both true are not supported
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Matmul>(
        adj_x, adj_y);
    boradcast_required
        ? (*op).BindInput(broadcast_out[0]).BindInput(broadcast_out[1])
        : (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Rnn : public OpMapperBase<TfLiteRNNParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create Rnn op");
    const auto builtin = reinterpret_cast<const TfLiteRNNParams*>(params);

    tim::vx::ops::RNNCell::ActivationType act;
    switch (builtin->activation) {
      case kTfLiteActRelu:
        act = tim::vx::ops::RNNCell::kRELU;
        break;
      case kTfLiteActReluN1To1:
        act = tim::vx::ops::RNNCell::kRELU1;
        break;
      case kTfLiteActRelu6:
        act = tim::vx::ops::RNNCell::kRELU6;
        break;
      case kTfLiteActTanh:
        act = tim::vx::ops::RNNCell::kTANH;
        break;
      case kTfLiteActSigmoid:
        act = tim::vx::ops::RNNCell::kSIGMOID;
        break;
      default:
        printf("Not supported activition type for Rnn = %d",
               static_cast<int32_t>(builtin->activation));
        break;
    }
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::RNNCell>(act);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Gather : public OpMapperBase<TfLiteGatherParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (context->tensors[input_index].type == kTfLiteString) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                        "String input is not supported");
        return false;
      }
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create Gather op");
    const auto builtin = reinterpret_cast<const TfLiteGatherParams*>(params);
    int batch_dims = builtin->batch_dims;
    batch_dims < 0 ? batch_dims += inputs[1]->GetShape().size() : batch_dims;

    int axis = vx::delegate::utils::ConvertAxis(builtin->axis,
                                                inputs[0]->GetShape().size());
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Gather>(
        axis, batch_dims);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct GatherNd : public OpMapperBase<EmptyStructPlaceholder> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if (context->tensors[input_index].type == kTfLiteString) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                        "String input is not supported");
        return false;
      }
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create GatherNd op");
    std::vector<int32_t> axis({0});
    inputs[1] = ReverseInputTensor(delegate, inputs[1], axis);
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::GatherNd>();

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct UnidirectionalSequenceLstm : public OpMapperBase<TfLiteUnidirectionalSequenceLSTMParams> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const {
    int h_state_index = node->inputs->data[lstm::full::kOutputStateTensor];
    int c_state_index = node->inputs->data[lstm::full::kCellStateTensor];
    int input_index = node->inputs->data[lstm::full::kInputTensor];
    int output_index = node->outputs->data[lstm::full::kOutputTensor];
    if (!(context->tensors[input_index].type == kTfLiteUInt8 ||
          context->tensors[input_index].type == kTfLiteInt8)) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "UnidirectionalLstm input is only support UInt8 || Int8");
      return false;
    }
    if (context->tensors[h_state_index].type !=
        context->tensors[output_index].type) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "OutputState should be the same type as output");
      return false;
    }
    if (!(context->tensors[c_state_index].type == kTfLiteFloat16 ||
          context->tensors[c_state_index].type == kTfLiteInt16)) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "Cell_state is only support Int16 || Float16 ");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create UnidirectionalSequenceLstm op");
    const auto builtin = reinterpret_cast<const TfLiteUnidirectionalSequenceLSTMParams*>(params);
    float cell_clip = builtin -> cell_clip;
    float proj_clip = builtin -> proj_clip;
    tim::vx::ops::UnidirectionalSequenceLstm::ActivationType act;
    float forget_bias = 0;
    bool time_major = builtin -> time_major;
    tim::vx::ops::UnidirectionalSequenceLstm::ActivationType recurrent_act_type = tim::vx::ops::UnidirectionalSequenceLstm::kSIGMOID;
    bool return_sequences = true;
    switch (builtin -> activation)
    {
    case kTfLiteActRelu:
      act = tim::vx::ops::UnidirectionalSequenceLstm::kRELU;
      break;
    case kTfLiteActRelu6:
      act = tim::vx::ops::UnidirectionalSequenceLstm::kRELU6;
      break;
    case kTfLiteActTanh:
      act = tim::vx::ops::UnidirectionalSequenceLstm::kTANH;
      break;
    case kTfLiteActSigmoid:
      act = tim::vx::ops::UnidirectionalSequenceLstm::kSIGMOID;
      break;
    default:
      printf("Not supported activition type for UnidirectionalSequenceLstm = %d", static_cast<int32_t>(builtin -> activation));
      break;
    }
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::UnidirectionalSequenceLstm>(
        cell_clip, proj_clip, act, forget_bias, time_major, recurrent_act_type, return_sequences);
    auto tensor_placeholder = delegate->GetGraph()->CreateTensorPlaceHolder();

    std::shared_ptr<tim::vx::Tensor> c_state_t = inputs[lstm::full::kCellStateTensor];
    std::shared_ptr<tim::vx::Tensor> h_state_t = inputs[lstm::full::kOutputStateTensor];

    const auto target_dtype = tim::vx::DataType::FLOAT16;
    if (c_state_t->GetSpec().datatype_ != target_dtype && c_state_t->GetSpec().datatype_ != tim::vx::DataType::FLOAT32) {
      auto dc_on_c = delegate->GetGraph()->CreateOperation<tim::vx::ops::DataConvert>();
      (*dc_on_c).BindInput(c_state_t);

      tim::vx::TensorSpec dc_out_tensor_spec = c_state_t->GetSpec();
      dc_out_tensor_spec.datatype_ = target_dtype;
      dc_out_tensor_spec.quantization_ = tim::vx::Quantization();
      auto dc_out_tensor = delegate->GetGraph()->CreateTensor(dc_out_tensor_spec);
      c_state_t = dc_out_tensor;
      (*dc_on_c).BindOutput(dc_out_tensor);
    }

    std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors = {
      inputs[lstm::full::kInputTensor],
      h_state_t,
      c_state_t,

      inputs[lstm::full::kInputToInputWeightsTensor],
      inputs[lstm::full::kInputToForgetWeightsTensor],
      inputs[lstm::full::kInputToCellWeightsTensor ],
      inputs[lstm::full::kInputToOutputWeightsTensor ],

      inputs[lstm::full::kRecurrentToInputWeightsTensor],
      inputs[lstm::full::kRecurrentToForgetWeightsTensor ],
      inputs[lstm::full::kRecurrentToCellWeightsTensor ],
      inputs[lstm::full::kRecurrentToOutputWeightsTensor ],

      // peephole weights : optional
      inputs[lstm::full::kCellToInputWeightsTensor ],
      inputs[lstm::full::kCellToForgetWeightsTensor ],
      inputs[lstm::full::kCellToOutputWeightsTensor ],

      //gate bias : optional
      inputs[lstm::full::kInputGateBiasTensor ],
      inputs[lstm::full::kForgetGateBiasTensor ],
      inputs[lstm::full::kCellGateBiasTensor ],
      inputs[lstm::full::kOutputGateBiasTensor ],

      // Projection : optional
      inputs[lstm::full::kProjectionWeightsTensor ],
      inputs[lstm::full::kProjectionBiasTensor ],

      // AUX?
    };

    if (inputs.size() == 24) {
      // LayerNorm: optional
      input_tensors.push_back(inputs[lstm::full::kInputLayerNormCoefficientsTensor  ]);
      input_tensors.push_back(inputs[lstm::full::kForgetLayerNormCoefficientsTensor ]);
      input_tensors.push_back(inputs[lstm::full::kCellLayerNormCoefficientsTensor   ]);
      input_tensors.push_back(inputs[lstm::full::kOutputLayerNormCoefficientsTensor ]);
    }

    (*op).BindInputs(
      input_tensors
      );
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct BidirectionalSequenceLstm : public OpMapperBase<TfLiteBidirectionalSequenceLSTMParams> {

  // Input Tensors
  constexpr  static int kInputTensor = 0;

  // Forward LSTM cell tensors.
  // Input weight tensors of size: {n_cell, n_input}
  constexpr static int kFwInputToInputWeightsTensor = 1;  // Optional
  constexpr static int kFwInputToForgetWeightsTensor = 2;
  constexpr static int kFwInputToCellWeightsTensor = 3;
  constexpr static int kFwInputToOutputWeightsTensor = 4;

  // Recurrent weight tensors of size {n_cell, n_output}
  constexpr static int kFwRecurrentToInputWeightsTensor = 5;  // Optional
  constexpr static int kFwRecurrentToForgetWeightsTensor = 6;
  constexpr static int kFwRecurrentToCellWeightsTensor = 7;
  constexpr static int kFwRecurrentToOutputWeightsTensor = 8;

  // Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
  constexpr static int kFwCellToInputWeightsTensor = 9;    // Optional
  constexpr static int kFwCellToForgetWeightsTensor = 10;  // Optional
  constexpr static int kFwCellToOutputWeightsTensor = 11;  // Optional

  // Gates bias tensors of size {n_cell}
  constexpr static int kFwInputGateBiasTensor = 12;  // Optional
  constexpr static int kFwForgetGateBiasTensor = 13;
  constexpr static int kFwCellGateBiasTensor = 14;
  constexpr static int kFwOutputGateBiasTensor = 15;

  // Projection weight tensor of size {n_output, n_cell}
  constexpr static int kFwProjectionWeightsTensor = 16;  // Optional
  // Projection bias tensor of size {n_output}
  constexpr static int kFwProjectionBiasTensor = 17;  // Optional

  // Backward LSTM cell tensors.
  // Input weight tensors of size: {n_cell, n_input}
  constexpr static int kBwInputToInputWeightsTensor = 18;  // Optional
  constexpr static int kBwInputToForgetWeightsTensor = 19;
  constexpr static int kBwInputToCellWeightsTensor = 20;
  constexpr static int kBwInputToOutputWeightsTensor = 21;

  // Recurrent weight tensors of size {n_cell, n_output}
  constexpr static int kBwRecurrentToInputWeightsTensor = 22;  // Optional
  constexpr static int kBwRecurrentToForgetWeightsTensor = 23;
  constexpr static int kBwRecurrentToCellWeightsTensor = 24;
  constexpr static int kBwRecurrentToOutputWeightsTensor = 25;

  // Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
  constexpr static int kBwCellToInputWeightsTensor = 26;   // Optional
  constexpr static int kBwCellToForgetWeightsTensor = 27;  // Optional
  constexpr static int kBwCellToOutputWeightsTensor = 28;  // Optional

  // Gates bias tensors of size {n_cell}
  constexpr static int kBwInputGateBiasTensor = 29;  // Optional
  constexpr static int kBwForgetGateBiasTensor = 30;
  constexpr static int kBwCellGateBiasTensor = 31;
  constexpr static int kBwOutputGateBiasTensor = 32;

  // Projection weight tensor of size {n_output, n_cell}
  constexpr static int kBwProjectionWeightsTensor = 33;  // Optional
  // Projection bias tensor of size {n_output}
  constexpr static int kBwProjectionBiasTensor = 34;  // Optional

  // Stateful input tensors that are variables and will be modified by the Op.
  // Activation state tensors of size {n_batch, n_output}
  constexpr static int kFwInputActivationStateTensor = 35;
  // Cell state tensors of size {n_batch, n_cell}
  constexpr static int kFwInputCellStateTensor = 36;
  // Activation state tensors of size {n_batch, n_output}
  constexpr static int kBwInputActivationStateTensor = 37;
  // Cell state tensors of size {n_batch, n_cell}
  constexpr static int kBwInputCellStateTensor = 38;

  // Used as auxiliary input and weights 
  constexpr static int kAuxInputTensor = 39;  // Optional
  // Forward weights.
  constexpr static int kFwAuxInputToInputWeightsTensor = 40;   // Optional
  constexpr static int kFwAuxInputToForgetWeightsTensor = 41;  // Optional
  constexpr static int kFwAuxInputToCellWeightsTensor = 42;    // Optional
  constexpr static int kFwAuxInputToOutputWeightsTensor = 43;  // Optional
  // Backward weights.
  constexpr static int kBwAuxInputToInputWeightsTensor = 44;   // Optional
  constexpr static int kBwAuxInputToForgetWeightsTensor = 45;  // Optional
  constexpr static int kBwAuxInputToCellWeightsTensor = 46;    // Optional
  constexpr static int kBwAuxInputToOutputWeightsTensor = 47;  // Optional

  // Output tensors.
  constexpr static int kFwOutputTensor = 0;
  constexpr static int kBwOutputTensor = 1;  // Ignored if merge_outputs is set.

  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const {
    int i2i_index = node->inputs->data[kFwInputToInputWeightsTensor];
    int fwprojection_weights_index = node->inputs->data[kFwProjectionWeightsTensor];
    int fwaux_weight_index = node->inputs->data[kFwAuxInputToCellWeightsTensor];
   
    if ( context->tensors[i2i_index].type != kTfLiteFloat32 ){
       TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "Quantized weights are not supported");
       return false;
    }
    if ( fwprojection_weights_index != -1 ){
       TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "Projection weights are not supported");
       return false;
    }
    if ( fwaux_weight_index != -1 ){
       TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "Aux weights are not supported");
       return false;
    }
    
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create BidirectionalSequenceLstm op");
    const auto builtin = reinterpret_cast<const TfLiteBidirectionalSequenceLSTMParams*>(params);
    float cell_clip = builtin -> cell_clip;
    float proj_clip = builtin -> proj_clip;
    tim::vx::ops::BidirectionalSequenceLstm::ActivationType act;
    float forget_bias = 0;
    bool time_major = builtin -> time_major;
    tim::vx::ops::BidirectionalSequenceLstm::ActivationType recurrent_act_type = tim::vx::ops::BidirectionalSequenceLstm::kSIGMOID;
    bool return_sequences = true;
    switch (builtin -> activation)
    {
    case kTfLiteActRelu:
      act = tim::vx::ops::BidirectionalSequenceLstm::kRELU;
      break;
    case kTfLiteActRelu6:
      act = tim::vx::ops::BidirectionalSequenceLstm::kRELU6;
      break;
    case kTfLiteActTanh:
      act = tim::vx::ops::BidirectionalSequenceLstm::kTANH;
      break;
    case kTfLiteActSigmoid:
      act = tim::vx::ops::BidirectionalSequenceLstm::kSIGMOID;
      break;
    default:
      printf("Not supported activition type for BidirectionalSequenceLstm = %d", static_cast<int32_t>(builtin -> activation));
      break;
    }
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::BidirectionalSequenceLstm>(
        cell_clip, proj_clip, act, forget_bias, time_major, recurrent_act_type, return_sequences);
    auto tensor_placeholder = delegate->GetGraph()->CreateTensorPlaceHolder();
  
    std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors = {
      inputs[kInputTensor],

      // Forward LSTM cell tennsors
      // Input weight tensors 
      inputs[kFwInputToInputWeightsTensor],
      inputs[kFwInputToForgetWeightsTensor],
      inputs[kFwInputToCellWeightsTensor ],
      inputs[kFwInputToOutputWeightsTensor ],

      // Recurrent weight tensors
      inputs[kFwRecurrentToInputWeightsTensor],
      inputs[kFwRecurrentToForgetWeightsTensor ],
      inputs[kFwRecurrentToCellWeightsTensor ],
      inputs[kFwRecurrentToOutputWeightsTensor ],

      // peephole weights : optional
      inputs[kFwCellToInputWeightsTensor ],
      inputs[kFwCellToForgetWeightsTensor ],
      inputs[kFwCellToOutputWeightsTensor ],

      // gate bias : optional
      inputs[kFwInputGateBiasTensor ],
      inputs[kFwForgetGateBiasTensor ],
      inputs[kFwCellGateBiasTensor ],
      inputs[kFwOutputGateBiasTensor ],

      // Projection : optional
      inputs[kFwProjectionWeightsTensor ],
      inputs[kFwProjectionBiasTensor ],

      // Backward LSTM cell tensors
      // Input weight tensors
      inputs[kBwInputToInputWeightsTensor],
      inputs[kBwInputToForgetWeightsTensor],
      inputs[kBwInputToCellWeightsTensor ],
      inputs[kBwInputToOutputWeightsTensor ],

      // Recurrent weight tensors
      inputs[kBwRecurrentToInputWeightsTensor],
      inputs[kBwRecurrentToForgetWeightsTensor ],
      inputs[kBwRecurrentToCellWeightsTensor ],
      inputs[kBwRecurrentToOutputWeightsTensor ],

      // peephole weights : optional
      inputs[kBwCellToInputWeightsTensor ],
      inputs[kBwCellToForgetWeightsTensor ],
      inputs[kBwCellToOutputWeightsTensor ],

      // gate bias : optional
      inputs[kBwInputGateBiasTensor ],
      inputs[kBwForgetGateBiasTensor ],
      inputs[kBwCellGateBiasTensor ],
      inputs[kBwOutputGateBiasTensor ],
     
      // Projection : optional
      inputs[kBwProjectionWeightsTensor ],
      inputs[kBwProjectionBiasTensor ], 
      
      // State Tensor
      inputs[kFwInputActivationStateTensor],
      inputs[kFwInputCellStateTensor],
      inputs[kBwInputActivationStateTensor],
      inputs[kBwInputCellStateTensor],

      // Aux
      inputs[kAuxInputTensor],
      inputs[kFwAuxInputToInputWeightsTensor],
      inputs[kFwAuxInputToForgetWeightsTensor],
      inputs[kFwAuxInputToCellWeightsTensor],
      inputs[kFwAuxInputToOutputWeightsTensor],
      inputs[kBwAuxInputToInputWeightsTensor],
      inputs[kBwAuxInputToForgetWeightsTensor],
      inputs[kBwAuxInputToCellWeightsTensor],
      inputs[kBwAuxInputToOutputWeightsTensor],
     
     // LayerNorm :not used
      tensor_placeholder,
      tensor_placeholder,
      tensor_placeholder,
      tensor_placeholder,
      tensor_placeholder,
      tensor_placeholder,
      tensor_placeholder,
      tensor_placeholder,
    };

    std::vector<std::shared_ptr<tim::vx::Tensor>> output_tensors={
      outputs[kFwOutputTensor],
      tensor_placeholder,
      tensor_placeholder,
      outputs[kBwOutputTensor],
      tensor_placeholder,
      tensor_placeholder,
    };
    (*op).BindInputs(
      input_tensors
      );
    (*op).BindOutputs(output_tensors);

    delegate->GetOps().push_back(std::move(op));

    return true;
   } //HandleMapOp
  };

struct Batch2Space : public OpMapperBase<TfLiteBatchToSpaceNDParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    int input_index = node->inputs->data[0];
    if (context->tensors[input_index].dims->size != 4) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "batch2space in vx-delagate only support 4D input");
      return false;
    }
    int block_index = node->inputs->data[1];
    if (context->tensors[block_index].dims->data[0] != 2) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "batch2space in vx-delagate only support the input whose "
                      "spatial dimensions is 2");
      return false;
    }
    int output_index = node->outputs->data[0];
    if(context->tensors[output_index].dims->size == 0){
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "dynamic shape in not support in batchtospace");
      return false;
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create Batch2Space op");
    // the value of block_size_num should be 2.
    int block_size_num = inputs[1]->GetShape()[0];
    std::vector<int> block_size(block_size_num);
    std::vector<int> crop(block_size_num * 2);
    inputs[1]->CopyDataFromTensor(block_size.data());
    inputs[2]->CopyDataFromTensor(crop.data());
    block_size = std::vector<int>(block_size.rbegin(), block_size.rend());
    std::vector<int> new_crop =
        vx::delegate::utils::TransposeVec<int>(crop, {2, 3, 0, 1});
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Batch2Space>(
        block_size, new_crop, tim::vx::DataLayout::CWHN);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Space2Batch : public OpMapperBase<TfLiteSpaceToBatchNDParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    int input_index = node->inputs->data[0];
    if (context->tensors[input_index].dims->size != 4) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "space2batch in vx-delegate only support 4D input");
      return false;
    }
    int output_index = node->outputs->data[0];
    if(context->tensors[output_index].dims->size == 0){
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "dynamic shape in not support in space2batch");
      return false;
    }
    int block_index = node->inputs->data[1];
    if (context->tensors[block_index].dims->data[0] != 2) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "space2batch in vx-delegate only support the input whose "
                      "spatial dimensions is 2");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create SpaceToBatch op");
    // the value of block_size_num should be 2.
    int block_size_num = inputs[1]->GetShape()[0];
    std::vector<int> block_size(block_size_num);
    std::vector<int> pad(block_size_num * 2);
    inputs[1]->CopyDataFromTensor(block_size.data());
    inputs[2]->CopyDataFromTensor(pad.data());
    block_size = std::vector<int>(block_size.rbegin(), block_size.rend());
    std::vector<int> new_pad =
        vx::delegate::utils::TransposeVec<int>(pad, {2, 3, 0, 1});
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Space2Batch>(
        block_size, new_pad, tim::vx::DataLayout::CWHN);
    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct ReverseV2 : public OpMapperBase<TfLiteReverseSequenceParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    TfLiteTensor param_tensor = context->tensors[node->inputs->data[1]];
    if (param_tensor.allocation_type != kTfLiteMmapRo) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "const axis_tensor is only supported in reverse.");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create ReverseV2 op");
    auto axis_tensor = inputs[1];
    std::vector<int> axis(axis_tensor->GetShape()[0]);
    axis_tensor->CopyDataFromTensor(axis.data());
    axis.data()[0] = vx::delegate::utils::ConvertAxis(
        axis.data()[0], inputs[0]->GetShape().size() + 1);

    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::Reverse>(axis);

    (*op).BindInput(inputs[0]).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct CustomOpMap : public OpMapperBase<EmptyStructPlaceholder> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    return false;
  }
};

template <typename T_OperationType>
struct ReduceOpMapper : public OpMapperBase<TfLiteReducerParams> {
  std::string name_;

  ReduceOpMapper(std::string name) : name_(name) {}
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    TfLiteTensor axis_tensor = context->tensors[node->inputs->data[1]];

    if (axis_tensor.allocation_type != kTfLiteMmapRo) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "const axis_tensor is only supported in reduce.");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create reduce_%s op", name_.c_str());
    const auto builtin = reinterpret_cast<const TfLiteReducerParams*>(params);
    auto keep_dims = builtin->keep_dims;

    uint32_t axis_num = inputs[1]->GetShape()[0];
    std::vector<int32_t> axis(axis_num);
    inputs[1]->CopyDataFromTensor(axis.data());
    for (uint32_t i = 0; i < axis_num; i++) {
      axis[i] = vx::delegate::utils::ConvertAxis(axis[i],
                                                 inputs[0]->GetShape().size());
    }
    auto op =
        delegate->GetGraph()->CreateOperation<T_OperationType>(axis, keep_dims);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct ExpandDimsMapper : public OpMapperBase<EmptyStructPlaceholder> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    TfLiteTensor axis_tensor = context->tensors[node->inputs->data[1]];

    if (axis_tensor.allocation_type != kTfLiteMmapRo) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "const axis_tensor is only supported in expand_dims.");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create ExpandDims op");

    auto input_shape = inputs[0]->GetShape();
    int axis = 0;
    inputs[1]->CopyDataFromTensor(&axis);
    auto output_shape = outputs[0]->GetShape();
    uint32_t new_axis =
        vx::delegate::utils::ConvertAxis(axis, output_shape.size());

    std::vector<uint32_t> expanded_shape(input_shape);
    expanded_shape.insert(expanded_shape.begin() + new_axis, 1);

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(
        expanded_shape);

    (*op).BindInput(inputs[0]);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct LeakyReluMapper : public OpMapperBase<TfLiteLeakyReluParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create LeakyRelu op");
    const auto builtin = reinterpret_cast<const TfLiteLeakyReluParams*>(params);
    auto alpha = builtin->alpha;
    auto op =
        delegate->GetGraph()->CreateOperation<tim::vx::ops::LeakyRelu>(alpha);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Slice : public OpMapperBase<EmptyStructPlaceholder> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    int input_index = node->inputs->data[0];
    int begin_index = node->inputs->data[1];
    int len_index = node->inputs->data[2];

    int output_index = node->outputs->data[0];
    int input_dim_size = context->tensors[input_index].dims->size;
    int batch_in = context->tensors[input_index].dims->data[0];
    int batch_out = context->tensors[output_index].dims->data[0];

    bool is_begin_dynamic = context->tensors[begin_index].allocation_type != kTfLiteMmapRo;
    bool is_len_dynamic = context->tensors[len_index].allocation_type != kTfLiteMmapRo;


    if( is_begin_dynamic || is_len_dynamic) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "vx-delegate cannot support dynamic shaped operator(slice), fallback it to CPU");
      return false;
    }
    if (input_dim_size > 3 && (batch_in != batch_out)) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "vx-delegate doesn't support slice in batch.");
      return false;
    }

    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create Slice op");
    auto input_tensor = inputs[0];
    auto begin_tensor = inputs[1];
    auto size_tensor = inputs[2];
    uint32_t input_dims = input_tensor->GetShape().size();
    uint32_t begin_size = begin_tensor->GetShape()[0];
    uint32_t size_size = size_tensor->GetShape()[0];
    std::vector<int32_t> begin(begin_size);
    std::vector<int32_t> size(size_size);
    begin_tensor->CopyDataFromTensor(begin.data());
    size_tensor->CopyDataFromTensor(size.data());

    std::reverse(begin.begin(), begin.end());
    std::reverse(size.begin(), size.end());

    for (int i = 0; i < size.size(); i++) {
      if (size[i] == -1) {  // If size[i] == -1, that means extract all
                            // elements of demension i.
        size[i] = input_tensor->GetShape()[i];
      }
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Slice>(
        input_dims, begin, size);

    (*op).BindInput(inputs[0]);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct SplitVMapper : public OpMapperBase<TfLiteSplitVParams> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    TfLiteTensor output_tensor = context->tensors[node->outputs->data[0]];
    if (output_tensor.allocation_type == kTfLiteDynamic) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "dynamic shpae is not supported in split.");
      return false;
    }
    for (int i = 0; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      if ((context->tensors[input_index].type == kTfLiteInt8 ||
           context->tensors[input_index].type == kTfLiteUInt8) &&
          context->tensors[input_index].quantization.type ==
              kTfLiteNoQuantization) {
        TFLITE_LOG_PROD(
            TFLITE_LOG_ERROR,
            "Int8 or uint8 input without quantization is not supported in "
            "Split");
        return false;
      }
    }

    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create SplitV op");

    const auto builtin = reinterpret_cast<const TfLiteSplitVParams*>(params);
    auto input_tensor = inputs[0];
    auto slices_tensor = inputs[1];
    auto axis_tensor = inputs[2];

    int32 axis = 0;
    axis_tensor->CopyDataFromTensor(&axis);
    axis =
        vx::delegate::utils::ConvertAxis(axis, input_tensor->GetShape().size());

    uint32_t total_slices_dim = inputs[0]->GetShape()[axis];
    std::vector<int32_t> slices_i32(builtin->num_splits);
    std::vector<uint32_t> slices;
    slices_tensor->CopyDataFromTensor(slices_i32.data());
    for (auto s : slices_i32) {
      if (s > 0) {
        total_slices_dim -= s;
        slices.push_back(s);
      } else {
        slices.push_back(total_slices_dim);
      }
    }

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Split>(
        axis, slices);

    (*op).BindInput(inputs[0]);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct Select : public OpMapperBase<EmptyStructPlaceholder> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    int condition_index = node->inputs->data[0];
    int input_x_index = node->inputs->data[1];
    if (context->tensors[condition_index].dims->size !=
        context->tensors[input_x_index].dims->size) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "condition and input must have the same rank");
      return false;
    }
    for (int i = 1; i < node->inputs->size; i++) {
      int input_index = node->inputs->data[i];
      auto input_type = context->tensors[input_index].type;
      if (input_type == kTfLiteBool || input_type == kTfLiteInt8 ||
          input_type == kTfLiteUInt8) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Bool type input is not supported");
        return false;
      }
    }
    for (int i = 0; i < node->outputs->size; i++) {
      int output_index = node->outputs->data[i];
      auto output_type = context->tensors[output_index].type;
      if (output_type == kTfLiteBool || output_type == kTfLiteInt8 ||
          output_type == kTfLiteUInt8) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Bool type output is not supported");
        return false;
      }
    }

    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create Select op");

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Select>();

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

template <typename T_OperationType>
struct LogicalOpMapper : public OpMapperBase<EmptyStructPlaceholder> {
  std::string name_;

  LogicalOpMapper(std::string name) : name_(name) {}
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Logical %s op", name_.c_str());

    auto op = delegate->GetGraph()->CreateOperation<T_OperationType>();
    (*op).BindInputs(inputs).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct PackMapper : public OpMapperBase<TfLitePackParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    if (input_tensor.type == kTfLiteInt32 ||
        (input_tensor.dims->size == 1 && (input_tensor.type == kTfLiteInt8 ||
                                          input_tensor.type == kTfLiteUInt8))) {
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Pack op");
    const auto builtin = reinterpret_cast<const TfLitePackParams*>(params);
    uint32_t axis = vx::delegate::utils::ConvertAxis(
        builtin->axis, inputs[0]->GetShape().size() + 1);

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Stack>(
        axis, inputs.size());
    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);
    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct UnpackMapper : public OpMapperBase<TfLiteUnpackParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Unpack op");
    const auto builtin = reinterpret_cast<const TfLiteUnpackParams*>(params);
    uint32_t axis = vx::delegate::utils::ConvertAxis(
        builtin->axis, inputs[0]->GetShape().size());
    uint32_t num = builtin->num;
    if(num==0){
      num = inputs[0]->GetShape()[axis];
    }
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Unstack>(
        axis, num);
    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);
    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct OneHotMapper : public OpMapperBase<TfLiteOneHotParams> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    auto depth_tensor = context->tensors[node->inputs->data[1]];
    if (depth_tensor.allocation_type != kTfLiteMmapRo) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "delegate only support parameters tensor as const input");
      return false;
    }
    auto output_tensor = context->tensors[node->outputs->data[0]];
    if (output_tensor.type == kTfLiteBool) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Bool type output is not supported");
      return false;
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating OneHot op");
    const auto builtin = reinterpret_cast<const TfLiteOneHotParams*>(params);
    auto depth = inputs[1];
    auto on_value = inputs[2];
    auto off_value = inputs[3];
    int32_t vx_axis = vx::delegate::utils::ConvertAxis(
        builtin->axis, outputs[0]->GetShape().size());

    int32_t depth_data;
    float on_value_data;
    float off_value_data;
    depth->CopyDataFromTensor(&depth_data);
    on_value->CopyDataFromTensor(&on_value_data);
    off_value->CopyDataFromTensor(&off_value_data);

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::OneHot>(
        depth_data, on_value_data, off_value_data, vx_axis);
    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);
    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

template <typename T_OperationType>
struct ArgOpMapper : public OpMapperBase<EmptyStructPlaceholder> {
  std::string name_;

  ArgOpMapper(std::string name) : name_(name) {}
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const override {
    if (context->tensors[node->inputs->data[0]].type == kTfLiteInt8 &&
        context->tensors[node->outputs->data[0]].type == kTfLiteInt32) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "I8 input/I32 output is not supported");
      return false;
    }
    if (0 == context->tensors[node->inputs->data[0]].dims->size ||
        0 == context->tensors[node->outputs->data[0]].dims->size) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "Arg cannot support dynamic shape");
      return false;
    }
    return true;
  }
  virtual bool HandleMapOp(
      vx::delegate::Delegate* delegate,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
      std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
      const void* params) {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Arg %s op", name_.c_str());

    auto axis_tensor = inputs[1];
    std::vector<int> axis(axis_tensor->GetShape()[0]);
    axis_tensor->CopyDataFromTensor(axis.data());

    auto transform_axis =
        vx::delegate::utils::ConvertAxis(axis[0], inputs[0]->GetShape().size());

    auto op =
        delegate->GetGraph()->CreateOperation<T_OperationType>(transform_axis);

    (*op).BindInput(inputs[0]);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

template <typename T_Param>
struct Conv3dKind
    : public OpMapperBase<T_Param, FusedActivationAction<0, T_Param>> {};
struct Conv3dMapper : public Conv3dKind<TfLiteConv3DParams> {
  virtual bool IsOpSupported(TfLiteContext* context,
                             TfLiteNode* node,
                             const TfLiteRegistration* registration) const {
    const auto builtin =
        reinterpret_cast<const TfLiteConv3DParams*>(node->builtin_data);
    if (builtin->dilation_width_factor > 1 ||
        builtin->dilation_height_factor > 1 ||
        builtin->dilation_depth_factor > 1) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "conv3d could not support dilation > 1.");
      return false;
    }
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto weight_tensor = context->tensors[node->inputs->data[1]];

    if (input_tensor.type != weight_tensor.type) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "hybrid data type is not supported in conv3d.");
      return false;
    }
    if (weight_tensor.allocation_type != kTfLiteMmapRo) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                      "weight tensor must be const in conv3d.");
      return false;
    }
    return true;
  }

  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Conv3d op");
    const auto builtin = reinterpret_cast<const TfLiteConv3DParams*>(params);

    // tensorflow kernel layout [kd, Kh, Kw, Ic, Oc] ---> TIM-VX [Oc, Ic, Kw,
    // Kh, Kd]
    int32_t weights = inputs[1]->GetShape()[0];
    int32_t kernel_w = inputs[1]->GetShape()[2];
    int32_t kernel_h = inputs[1]->GetShape()[3];
    int32_t kernel_d = inputs[1]->GetShape()[4];

    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::Conv3d>(
        static_cast<int32_t>(weights),
        TflitePadTypeToVsiPadType(builtin->padding),
        std::array<int32_t, 3>({kernel_w, kernel_h, kernel_d}),
        std::array<int32_t, 3>({builtin->stride_width,
                                builtin->stride_height,
                                builtin->stride_depth}),
        std::array<int32_t, 3>({builtin->dilation_width_factor,
                                builtin->dilation_height_factor,
                                builtin->dilation_depth_factor}),
        0,
        tim::vx::DataLayout::CWHDN,
        tim::vx::DataLayout::OcIcWHD);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

struct ShapeMapper : public OpMapperBase<TfLiteShapeParams> {
  bool IsOpSupported(TfLiteContext* context,
                     TfLiteNode* node,
                     const TfLiteRegistration* registration) const {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    for (int i = 0; i < input_tensor.dims->size; i++) {
      if (input_tensor.dims->data[i] <= 0) {
        TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                        "Negative shape values are not supported.");
        return false;
      }
    }
    return true;
  }
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Creating Shape op");
    std::vector<uint32_t> shape = inputs[0]->GetShape();
    tim::vx::TensorSpec shape_spec(tim::vx::DataType::INT32,
                                   {shape.size()},
                                   tim::vx::TensorAttribute::CONSTANT);
    auto shape_tensor = delegate->GetGraph()->CreateTensor(shape_spec, shape.data());
    delegate->GetTensors()[delegate->GetOperationOutput(0)] = shape_tensor;
    return true;
  }
};

using createIOpMapItemFunc = std::function<std::unique_ptr<IOpMapper>()>;
static const std::map<int, createIOpMapItemFunc> reg = {
#define REGISTER_OP_MAPPER(TFLITE_OP_CODE, MAPPER_TYPE, ...)                  \
  {                                                                           \
    TFLITE_OP_CODE, [] { return std::make_unique<MAPPER_TYPE>(__VA_ARGS__); } \
  }

    REGISTER_OP_MAPPER(kTfLiteBuiltinFullyConnected, FullyConnectedMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSoftmax, SoftmaxMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinConv2d, Conv2dMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinMaxPool2d,
                       Pool2dMapper<tim::vx::PoolType::MAX>),
    REGISTER_OP_MAPPER(kTfLiteBuiltinAveragePool2d,
                       Pool2dMapper<tim::vx::PoolType::AVG_ANDROID>),
    REGISTER_OP_MAPPER(kTfLiteBuiltinDepthwiseConv2d, DepthwiseConv2dMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinDequantize,
                       DequantizeMapper<tim::vx::ops::DataConvert>,
                       "Dequantize"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinQuantize,
                       QuantizeMapper<tim::vx::ops::DataConvert>,
                       "Quantize"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinConcatenation, ConcatenationMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLocalResponseNormalization,
                       LocalResponseNormalizationMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinL2Normalization, L2NormalizationMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReshape, ReshapeMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinStridedSlice, StridedSliceMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinPad, PadMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinExpandDims, ExpandDimsMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinOneHot, OneHotMapper),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinAbs, SimpleOpMapper<tim::vx::ops::Abs>, "Abs"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinSin, SimpleOpMapper<tim::vx::ops::Sin>, "Sin"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinExp, SimpleOpMapper<tim::vx::ops::Exp>, "Exp"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinLog, SimpleOpMapper<tim::vx::ops::Log>, "Log"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinSqrt, SimpleOpMapper<tim::vx::ops::Sqrt>, "Sqrt"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinRsqrt, SimpleOpMapper<tim::vx::ops::Rsqrt>, "Rsqrt"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinSquare, SimpleOpMapper<tim::vx::ops::Square>, "Square"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinFloor, SimpleOpMapper<tim::vx::ops::Floor>, "Floor"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinFloorDiv,
                       SimpleOpMapper<tim::vx::ops::FloorDiv>,
                       "FloorDiv"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinGreater,
                       SimpleOpMapper<tim::vx::ops::Greater>,
                       "Greater"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinGreaterEqual,
                       SimpleOpMapper<tim::vx::ops::GreaterOrEqual>,
                       "GreaterEqual"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinLess, SimpleOpMapper<tim::vx::ops::Less>, "Less"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLessEqual,
                       SimpleOpMapper<tim::vx::ops::LessOrEqual>,
                       "LessEqual"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinNotEqual,
                       SimpleOpMapper<tim::vx::ops::NotEqual>,
                       "NotEqual"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinEqual, SimpleOpMapper<tim::vx::ops::Equal>, "Equal"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLogicalNot,
                       SimpleOpMapper<tim::vx::ops::LogicalNot>,
                       "LogicalNot"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinHardSwish,
                       SimpleOpMapper<tim::vx::ops::HardSwish>,
                       "HardSwish"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinMinimum, MinimumMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinMaximum, MaximumMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinAdd, AddMapper, "Add"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSub, SubMapper, "Sub"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinDiv, DivMapper, "Div"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinMul, MulMapper, "Multiply"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinPow, PowMapper<tim::vx::ops::Pow>, "Pow"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinResizeNearestNeighbor,
                       ResizeMapper<tim::vx::ResizeType::NEAREST_NEIGHBOR>),
    REGISTER_OP_MAPPER(kTfLiteBuiltinResizeBilinear,
                       ResizeMapper<tim::vx::ResizeType::BILINEAR>),
    REGISTER_OP_MAPPER(kTfLiteBuiltinAddN, AddNMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSplit, SplitMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSqueeze, SqueezeMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSpaceToDepth, Space2DepthMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinDepthToSpace, Depth2SpaceMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinPrelu, PreluMapper),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinElu, SimpleOpMapper<tim::vx::ops::Elu>, "Elu"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinRelu, SimpleOpMapper<tim::vx::ops::Relu>, "Relu"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinReluN1To1, SimpleOpMapper<tim::vx::ops::Relu1>, "Relu1"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinRelu6, SimpleOpMapper<tim::vx::ops::Relu6>, "Relu6"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLogistic,
                       SimpleOpMapper<tim::vx::ops::Sigmoid>,
                       "Sigmoid"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinTranspose, Transpose),
    REGISTER_OP_MAPPER(kTfLiteBuiltinBatchMatmul, BatchMatmul),
    REGISTER_OP_MAPPER(kTfLiteBuiltinRnn, Rnn),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinNeg, SimpleOpMapper<tim::vx::ops::Neg>, "Neg"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinTanh, SimpleOpMapper<tim::vx::ops::Tanh>, "tanh"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinGather, Gather),
    REGISTER_OP_MAPPER(kTfLiteBuiltinGatherNd, GatherNd),
    REGISTER_OP_MAPPER(kTfLiteBuiltinUnidirectionalSequenceLstm,
                       UnidirectionalSequenceLstm),
    REGISTER_OP_MAPPER(kTfLiteBuiltinBidirectionalSequenceLstm,
                       BidirectionalSequenceLstm),
    REGISTER_OP_MAPPER(kTfLiteBuiltinBatchToSpaceNd, Batch2Space),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSpaceToBatchNd, Space2Batch),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReverseV2, ReverseV2),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReduceMin,
                       ReduceOpMapper<tim::vx::ops::ReduceMin>,
                       "Min"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinSum, ReduceOpMapper<tim::vx::ops::ReduceSum>, "Sum"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReduceMax,
                       ReduceOpMapper<tim::vx::ops::ReduceMax>,
                       "Max"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReduceAny,
                       ReduceOpMapper<tim::vx::ops::ReduceAny>,
                       "Any"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinReduceProd,
                       ReduceOpMapper<tim::vx::ops::ReduceProd>,
                       "Prod"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinMean, ReduceOpMapper<tim::vx::ops::ReduceMean>, "Mean"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLeakyRelu, LeakyReluMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLogicalAnd,
                       LogicalOpMapper<tim::vx::ops::LogicalAnd>,
                       "And"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinLogicalOr,
                       LogicalOpMapper<tim::vx::ops::LogicalOr>,
                       "Or"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSlice, Slice),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSplitV, SplitVMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinTransposeConv, TransposeConvMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSelect, Select),
    REGISTER_OP_MAPPER(kTfLiteBuiltinSelectV2, Select),
    REGISTER_OP_MAPPER(kTfLiteBuiltinPack, PackMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinUnpack, UnpackMapper),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinArgMin, ArgOpMapper<tim::vx::ops::ArgMin>, "Min"),
    REGISTER_OP_MAPPER(
        kTfLiteBuiltinArgMax, ArgOpMapper<tim::vx::ops::ArgMax>, "Max"),
    REGISTER_OP_MAPPER(kTfLiteBuiltinConv3d, Conv3dMapper),
    REGISTER_OP_MAPPER(kTfLiteBuiltinShape, ShapeMapper),

#undef REGISTER_OP_MAPPER
};

struct NBGOpMap : public OpMapperBase<TfLiteVsiNpuParams> {
  bool HandleMapOp(vx::delegate::Delegate* delegate,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const void* params) override {
    TFLITE_LOG(TFLITE_LOG_INFO, "Create NBG op");
    const auto builtin = reinterpret_cast<const TfLiteVsiNpuParams*>(params);
    auto op = delegate->GetGraph()->CreateOperation<tim::vx::ops::NBG>(
        reinterpret_cast<const char*>(builtin->binary),
        builtin->input_count,
        builtin->output_cout);

    (*op).BindInputs(inputs);
    (*op).BindOutputs(outputs);

    delegate->GetOps().push_back(std::move(op));

    return true;
  }
};

static const std::map<std::string, createIOpMapItemFunc> custom_reg = {
#define REGISTER_CUSTOM_OP(CUSTOM_NAME, MAPPER_TYPE, ...)                  \
  {                                                                        \
    CUSTOM_NAME, [] { return std::make_unique<MAPPER_TYPE>(__VA_ARGS__); } \
  }

    REGISTER_CUSTOM_OP("WRNN_BIDI_SEQGRU", CustomOpMap),
    REGISTER_CUSTOM_OP("vsi-npu", NBGOpMap),
#undef REGISTER_CUSTOM_OP
};

template <typename T>
struct OperationMapConstructor {
  T supported_builtins;
  OperationMapConstructor(
      const std::map<typename T::key_type, createIOpMapItemFunc> reg) {
    TFLITE_LOG(TFLITE_LOG_INFO, "Initialize supported_builtins");
    for (const auto& kv : reg) {
      supported_builtins.insert(std::make_pair(kv.first, kv.second()));
    }
  }
};

const OperationMapItemType& SupportedBuiltinOps() {
  static OperationMapConstructor<OperationMapItemType> c(reg);

  return c.supported_builtins;
}

const CustomOperationMapItemType& SupportedBuiltinCustomOps() {
  static OperationMapConstructor<CustomOperationMapItemType> c(custom_reg);

  return c.supported_builtins;
}

}  // namespace op_map
}  // namespace vx
