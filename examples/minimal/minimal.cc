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
#include <map>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/minimal_logging.h"

#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "vsi_npu_custom_op.h"
#include "util.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

void setupInput(int argc,
                char* argv[],
                const std::unique_ptr<tflite::Interpreter>& interpreter,
                bool is_cache_mode) {
  auto input_list = interpreter->inputs();
  bool use_random_input = false;

  if ((!is_cache_mode && input_list.size() != argc - 3) ||
      (is_cache_mode && input_list.size() != argc - 5)) {
    std::cout << "Warning: input count not match between command line and "
                 "model -> generate random data for inputs"
              << std::endl;
    use_random_input = true;
  }
  uint32_t i = is_cache_mode ? 5 : 3;
  //uint32_t i = 4; // argv index

  for (auto input_idx = 0; input_idx < input_list.size(); input_idx++) {
    auto in_tensor = interpreter->input_tensor(input_idx);

    std::cout << "Setup intput[" << std::string(interpreter->GetInputName(input_idx)) << "]" << std::endl;
    const char* input_data =  use_random_input ? "/dev/urandom" : argv[i];

    if (!use_random_input) {
      // get its size:
      std::ifstream file(input_data, std::ios::binary);
      std::streampos fileSize;

      file.seekg(0, std::ios::end);
      fileSize = file.tellg();
      file.seekg(0, std::ios::beg);

      if (fileSize != in_tensor->bytes) {
        std::cout << "Fatal: input size not matched" << std::endl;
        assert(false);
      }
    }

    switch (in_tensor->type) {
      case kTfLiteFloat32:
      {
        auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<float>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteUInt8:
      {
        auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<uint8_t>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteInt8: {
        auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<int8_t>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteInt32:
      {
        auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<int32_t>(input_idx), in.data(), in.size());
        break;
      }
      default: {
        std::cout << "Fatal: datatype for input not implemented" << std::endl;
        TFLITE_EXAMPLE_CHECK(false);
        break;
      }
    }

    i += 1;
  }
}

int main(int argc, char* argv[]) {
  if (argc <= 2) {
    fprintf(stderr, "minimal <external_delegate.so> <tflite model> <use_cache_mode> <cache file> <inputs>\n");
    return 1;
  }
  const char* delegate_so = argv[1];
  const char* filename = argv[2];
  bool is_use_cache_mode = false;
  const char* cachename;
  if(argc >= 5){
    int is_match = std::strcmp(argv[3],"use_cache_mode");
    if(is_match == 0){
      is_use_cache_mode = true;
      cachename = argv[4];
    }
  }

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_EXAMPLE_CHECK(model != nullptr);

  auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault(argv[1]);
  if(is_use_cache_mode){
    const char* allow_cache_key = "allowed_cache_mode";
    const char* allow_cache_value = "true";
    const char* cache_file_key = "cache_file_path";
    const char* cache_file_value = cachename;
    ext_delegate_option.insert(&ext_delegate_option,allow_cache_key,allow_cache_value);
    ext_delegate_option.insert(&ext_delegate_option,cache_file_key,cache_file_value);
  }

  auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(kNbgCustomOp, tflite::ops::custom::Register_VSI_NPU_PRECOMPILED());

  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> npu_interpreter;
  builder(&npu_interpreter);
  TFLITE_EXAMPLE_CHECK(npu_interpreter != nullptr);
  npu_interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);

  // Allocate tensor buffers.
  TFLITE_EXAMPLE_CHECK(npu_interpreter->AllocateTensors() == kTfLiteOk);
  TFLITE_LOG(tflite::TFLITE_LOG_INFO, "=== Pre-invoke NPU Interpreter State ===");
  tflite::PrintInterpreterState(npu_interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  setupInput(argc, argv, npu_interpreter,is_use_cache_mode);

  // Run inference
  TFLITE_EXAMPLE_CHECK(npu_interpreter->Invoke() == kTfLiteOk);

  // Get performance
  // {
  //   const uint32_t loop_cout = 10;
  //   auto start = std::chrono::high_resolution_clock::now();
  //   for (uint32_t i = 0; i < loop_cout; i++) {
  //     npu_interpreter->Invoke();
  //   }
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::cout << "[NPU Performance] Run " << loop_cout << " times, average time: " << (end - start).count() << " ms" << std::endl;
  // }

  TFLITE_LOG(tflite::TFLITE_LOG_INFO, "=== Post-invoke NPU Interpreter State ===");
  tflite::PrintInterpreterState(npu_interpreter.get());

  // CPU
  tflite::ops::builtin::BuiltinOpResolver cpu_resolver;
  tflite::InterpreterBuilder cpu_builder(*model, cpu_resolver);
  std::unique_ptr<tflite::Interpreter> cpu_interpreter;
  cpu_builder(&cpu_interpreter);
  TFLITE_EXAMPLE_CHECK(cpu_interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_EXAMPLE_CHECK(cpu_interpreter->AllocateTensors() == kTfLiteOk);
  TFLITE_LOG(tflite::TFLITE_LOG_INFO, "=== Pre-invoke CPU Interpreter State ===");
  tflite::PrintInterpreterState(cpu_interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  setupInput(argc, argv, cpu_interpreter,is_use_cache_mode);

  // Run inference
  TFLITE_EXAMPLE_CHECK(cpu_interpreter->Invoke() == kTfLiteOk);

  // Get performance
  // {
  //   const uint32_t loop_cout = 10;
  //   auto start = std::chrono::high_resolution_clock::now();
  //   for (uint32_t i = 0; i < loop_cout; i++) {
  //     cpu_interpreter->Invoke();
  //   }
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::cout << "[CPU Performance] Run " << loop_cout << " times, average time: " << (end - start).count() << " ms" << std::endl;
  // }

  TFLITE_LOG(tflite::TFLITE_LOG_INFO, "=== Post-invoke CPU Interpreter State ===");
  tflite::PrintInterpreterState(cpu_interpreter.get());

  auto output_idx_list = npu_interpreter->outputs();
  TFLITE_EXAMPLE_CHECK(npu_interpreter->outputs().size() ==
                       cpu_interpreter->outputs().size());
  for (size_t idx = 0; idx < output_idx_list.size(); idx++) {
    TFLITE_EXAMPLE_CHECK(npu_interpreter->output_tensor(idx)->bytes ==
                         cpu_interpreter->output_tensor(idx)->bytes);
    auto bytes = npu_interpreter->output_tensor(idx)->bytes;
    auto tensor_location = output_idx_list[idx];
    auto tensor_name = npu_interpreter->GetOutputName(idx);
    std::cout<<"Checking "<<idx <<" output. In tflite model, the location is "<<tensor_location<< ", tensor name is: "
             <<tensor_name<<std::endl;
    switch (npu_interpreter->output_tensor(idx)->type) {
      case kTfLiteInt8: {
        auto npu_out_buf = npu_interpreter->typed_output_tensor<int8_t>(idx);
        auto cpu_out_buf = cpu_interpreter->typed_output_tensor<int8_t>(idx);

        CompareTensorResult(idx, npu_out_buf, cpu_out_buf, bytes);
        break;
      }
      case kTfLiteUInt8: {
        auto npu_out_buf = npu_interpreter->typed_output_tensor<uint8_t>(idx);
        auto cpu_out_buf = cpu_interpreter->typed_output_tensor<uint8_t>(idx);

        CompareTensorResult(idx, npu_out_buf, cpu_out_buf, bytes);
        break;
      }
      case kTfLiteFloat32: {
        auto npu_out_buf = npu_interpreter->typed_output_tensor<float_t>(idx);
        auto cpu_out_buf = cpu_interpreter->typed_output_tensor<float_t>(idx);

        CompareTensorResult(idx, npu_out_buf, cpu_out_buf, bytes);
        break;
      }
       case kTfLiteInt32: {
        auto npu_out_buf = npu_interpreter->typed_output_tensor<int32_t>(idx);
        auto cpu_out_buf = cpu_interpreter->typed_output_tensor<int32_t>(idx);

        CompareTensorResult(idx, npu_out_buf, cpu_out_buf, bytes);
        break;
      }
      default: {
        TFLITE_EXAMPLE_CHECK(false);
      }
    }
  }
  TfLiteExternalDelegateDelete(ext_delegate_ptr);
  return 0;
}
