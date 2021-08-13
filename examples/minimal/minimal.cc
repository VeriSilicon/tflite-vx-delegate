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
#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>
#include <numeric>
#include <thread>
#include <chrono>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "tensorflow/lite/delegates/external/external_delegate.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

template <typename T>
std::vector<T> read_data(const char * filename, size_t required)
{
    // open the file:
    std::ifstream file(filename, std::ios::binary);

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if(fileSize != required) {
      std::cout << "Fatal: input size not matched" << std::endl;
      return std::vector<T>();
    }

    // reserve capacity
    std::vector<uint8_t> vec;
    vec.reserve(fileSize);

    // read the data:
    vec.insert(vec.begin(),
               std::istream_iterator<uint8_t>(file),
               std::istream_iterator<uint8_t>());

    std::vector<T> ret;
    ret.reserve(fileSize/sizeof(T));
    memcpy(ret.data(), vec.data(), fileSize);

    return ret;
}

template< typename T>
float cosine(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  auto calc_m = [](const std::vector<T>& lhs) {
    float lhs_m = 0.0f;

    for(auto iter = lhs.begin(); iter != lhs.end(); ++iter) {
      lhs_m += std::pow(*iter, 2);
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

void setupInput(int argc, char* argv[], const std::unique_ptr<tflite::Interpreter>& interpreter) {
  auto input_list = interpreter->inputs();
  bool use_random_input = false;
  if (input_list.size() != argc - 3) {
    std::cout << "Warning: input count not match between command line and model -> generate random data for inputs" << std::endl;
    use_random_input = true;
  }

  uint32_t i = 3; // argv index

  for (auto input_idx = 0; input_idx < input_list.size(); input_idx++) {
    auto in_tensor = interpreter->input_tensor(input_idx);

    std::cout << "Setup intput[" << std::string(interpreter->GetInputName(input_idx)) << "]" << std::endl;
    const char* input_data =  use_random_input ? "/dev/urandom" : argv[i];

    switch (in_tensor->type) {
      case kTfLiteFloat32:
      {
        auto in = read_data<float>(input_data, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<uint8_t>(input_idx), in.data(), sizeof(float)*in.size());
        break;
      }
      case kTfLiteUInt8:
      case kTfLiteInt8: {
        auto in = read_data<uint8_t>(input_data, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<uint8_t>(input_idx), in.data(), in.size());
        break;
      }
      default: {
        std::cout << "Fatal: datatype for input not implemented" << std::endl;
        TFLITE_MINIMAL_CHECK(false);
        break;
      }
    }

    i += 1;
  }
}

int main(int argc, char* argv[]) {
  if (argc <= 2) {
    fprintf(stderr, "minimal <external_delegate.so> <tflite model> <inputs>\n");
    return 1;
  }
  const char* delegate_so = argv[1];
  const char* filename = argv[2];
  // start from argv[3] to argv[N] is input tensor

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Prepare external delegate
  // TfLiteExternalDelegateOptions options = {
  //   .lib_path = argv[1], .count = 1, .keys = nullptr, .values = nullptr, .insert = nullptr
  // };

  auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault(argv[1]);
  auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::Interpreter> interpreter_cp;
  builder(&interpreter);
  builder(&interpreter_cp);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);
  interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
  interpreter_cp->ModifyGraphWithDelegate(ext_delegate_ptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  TFLITE_MINIMAL_CHECK(interpreter_cp->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  setupInput(argc, argv, interpreter);
  setupInput(argc, argv, interpreter_cp);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  TFLITE_MINIMAL_CHECK(interpreter_cp->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // CPU
  tflite::ops::builtin::BuiltinOpResolver cpu_resolver;
  tflite::InterpreterBuilder cpu_builder(*model, cpu_resolver);
  std::unique_ptr<tflite::Interpreter> cpu_interpreter;
  cpu_builder(&cpu_interpreter);
  TFLITE_MINIMAL_CHECK(cpu_interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(cpu_interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(cpu_interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  setupInput(argc, argv, cpu_interpreter);

  // Run inference
  TFLITE_MINIMAL_CHECK(cpu_interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke CPU Interpreter State ===\n");
  tflite::PrintInterpreterState(cpu_interpreter.get());

  if(0){// do run-sleep-run loop
    const uint32_t loop_count = 10;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds sleep_{1000};

    for(uint32_t i = 0; i < loop_count; ++i) {
    auto start_0 = std::chrono::high_resolution_clock::now();
      interpreter->Invoke();
    auto end_0 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> cost_0, cost_1;
      cost_0 = end_0 - start_0;
      std::cout << "[No switch]" << i << "  :cost = " << cost_0.count() << "ms" << std::endl;
    }
  }
  if(0){// do run-sleep-run loop
    const uint32_t loop_count = 10;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds sleep_{1000};

    for(uint32_t i = 0; i < loop_count; ++i) {
    auto start_0 = std::chrono::high_resolution_clock::now();
      interpreter->Invoke();
    auto end_0 = std::chrono::high_resolution_clock::now();
    auto start_1 = std::chrono::high_resolution_clock::now();
      interpreter_cp->Invoke();
    auto end_1 = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> cost_0, cost_1;
      cost_0 = end_0 - start_0;
      cost_1 = end_1 - start_1;
      std::cout << "[No Sleep]" << i << "  :cost 0 = " << cost_0.count() << "ms\t, cost 1 = " << cost_1.count() << "ms" << std::endl;
    }
  }
  if(0){// do run-sleep-run loop
    const uint32_t loop_count = 10;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds sleep_{1000};

    for(uint32_t i = 0; i < loop_count; ++i) {
    auto start_0 = std::chrono::high_resolution_clock::now();
      interpreter->Invoke();
    auto end_0 = std::chrono::high_resolution_clock::now();
      std::this_thread::sleep_for(sleep_);
    auto start_1 = std::chrono::high_resolution_clock::now();
      interpreter_cp->Invoke();
    auto end_1 = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> cost_0, cost_1;
      cost_0 = end_0 - start_0;
      cost_1 = end_1 - start_1;
      std::cout << "[UseSleep] " << i << "  :cost 0 = " << cost_0.count() << "ms\t, cost 1 = " << cost_1.count() << "ms" << std::endl;
    }
  }

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

  auto output_idx_list = interpreter->outputs();
  TFLITE_MINIMAL_CHECK(interpreter->outputs().size() == cpu_interpreter->outputs().size());
  {  // compare result cosine similarity
    for (auto idx = 0; idx < output_idx_list.size(); ++idx) {
      // std::vector<float> result;
      switch (interpreter->output_tensor(idx)->type) {
        // case kTfLiteInt8:
        case kTfLiteUInt8: {
          auto npu_out_buf = interpreter->typed_output_tensor<uint8_t>(idx);
          auto cpu_out_buf = cpu_interpreter->typed_output_tensor<uint8_t>(idx);
          TFLITE_MINIMAL_CHECK(interpreter->output_tensor(idx)->bytes ==
                               cpu_interpreter->output_tensor(idx)->bytes);

          auto bytes = interpreter->output_tensor(idx)->bytes;
          for (auto j = 0; j < bytes; ++j) {
            if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 2) {
              std::cout << "[Result mismatch]: Output[" << idx
                        << "], CPU vx NPU("
                        << static_cast<int32_t>(cpu_out_buf[j]) << ","
                        << static_cast<int32_t>(npu_out_buf[j]) << ")"
                        << std::endl;
            }
          }

          {
            std::vector<uint8_t> lhs(bytes);
            auto lquant =
                interpreter->tensor(output_idx_list[idx])->quantization;
            std::vector<uint8_t> rhs(bytes);
            // auto rquant =
            //     interpreter->tensor(output_idx_list[idx])->quantization;
            // auto dequantize = [](const std::vector<uint8_t>& quant_data,
            //                      const TfLiteQuantization& quant_param)
            //     -> std::vector<float> {
            //   std::vector<float> ret;

            //   std::transform(
            //       quant_data.cbegin(),
            //       quant_data.cend(),
            //       std::back_inserter(ret),
            //       [&quant_param](const uint8_t& d) {
            //         return ((TfLiteAffineQuantization*)quant_param.params)
            //                    ->scale->data[0] *
            //                (d - ((TfLiteAffineQuantization*)quant_param.params)
            //                         ->zero_point->data[0]);
            //       });

            //   return ret;
            // };

            memcpy(lhs.data(), cpu_out_buf, bytes);
            memcpy(rhs.data(), npu_out_buf, bytes);

            // std::vector<float> lhs_f;
            // std::vector<float> rhs_f;

            // lhs_f = dequantize(lhs, lquant);
            // rhs_f = dequantize(rhs, rquant);

            std::cout << "CosineCosineSimilarity = " << cosine(lhs, rhs)
                      << std::endl;
          //   std::cout << "CosineCosineSimilarity = " << cosine(lhs_f, rhs_f)
          //             << std::endl;
          }

          break;
        }
        case kTfLiteFloat32: {
          auto npu_out_buf = interpreter->typed_output_tensor<float>(idx);
          auto cpu_out_buf = cpu_interpreter->typed_output_tensor<float>(idx);
          TFLITE_MINIMAL_CHECK(interpreter->output_tensor(idx)->bytes ==
                               cpu_interpreter->output_tensor(idx)->bytes);

          auto bytes = interpreter->output_tensor(idx)->bytes;
          for (auto j = 0; j < bytes / sizeof(float); ++j) {
            if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) >
                0.001f) {  // TODO{sven}: not accurate
              std::cout << "[Result mismatch]: Output[" << idx
                        << "], CPU vx NPU(" << cpu_out_buf[j] << ","
                        << npu_out_buf[j] << ")" << std::endl;
            }
          }

          if(0){
            std::vector<float> lhs(bytes / sizeof(float));
            std::vector<float> rhs(bytes / sizeof(float));

            memcpy(lhs.data(), cpu_out_buf, bytes / sizeof(float));
            memcpy(rhs.data(), npu_out_buf, bytes / sizeof(float));

            std::cout << "CosineCosineSimilarity = " << cosine(lhs, rhs)
                      << std::endl;
          }
          break;
        }
        default: {
          TFLITE_MINIMAL_CHECK(false);
        }
      }
    }
  }

  TfLiteExternalDelegateDelete(ext_delegate_ptr);
  return 0;
}

