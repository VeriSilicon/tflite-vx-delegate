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

#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "vsi_npu_custom_op.h"

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

std::vector<uint8_t> read_data(const char * filename, size_t required)
{
    static std::map<std::string, std::vector<uint8_t>> cached_data;

    if (cached_data.find(filename) != cached_data.end()) {
      return cached_data[filename];
    }
    // open the file:
    std::ifstream file(filename, std::ios::binary);

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // reserve capacity not change size, memory copy in setInput will fail, so use resize()
    std::vector<uint8_t> vec;
    vec.resize(required);

    // read the data:
    file.read(reinterpret_cast<char *>(vec.data()), required);

    cached_data.insert(std::make_pair(std::string(filename), vec));

    return vec;
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

  auto lhs_m = calc_m(lhs);
  auto rhs_m = calc_m(rhs);

  float element_sum = 0.f;
  for(auto i = 0U; i < lhs.size(); ++i) {
    element_sum += lhs[i]*rhs[i];
  }

  return element_sum/(lhs_m*rhs_m);
}

void setupInput(int argc,
                char* argv[],
                const std::unique_ptr<tflite::Interpreter>& interpreter,
                bool is_cache_mode) {
  auto input_list = interpreter->inputs();
  bool use_random_input = false;

  if ((!is_cache_mode && input_list.size() != argc - 4) ||
      (is_cache_mode && input_list.size() != argc - 6)) {
    std::cout << "Warning: input count not match between command line and "
                 "model -> generate random data for inputs"
              << std::endl;
    use_random_input = true;
  }
  uint32_t i = is_cache_mode ? 6 : 4;
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
        auto in = read_data(input_data, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<float>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteUInt8:
      {
        auto in = read_data(input_data, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<uint8_t>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteInt8: {
        auto in = read_data(input_data, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<int8_t>(input_idx), in.data(), in.size());
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
  TFLITE_MINIMAL_CHECK(model != nullptr);

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
  TFLITE_MINIMAL_CHECK(npu_interpreter != nullptr);
  npu_interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(npu_interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(npu_interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  setupInput(argc, argv, npu_interpreter,is_use_cache_mode);

  // Run inference
  TFLITE_MINIMAL_CHECK(npu_interpreter->Invoke() == kTfLiteOk);

  // Get performance
  {
    const uint32_t loop_cout = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < loop_cout; i++) {
      npu_interpreter->Invoke();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[NPU Performance] Run " << loop_cout << " times, average time: " << (end - start).count() << " ms" << std::endl;
  }

  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(npu_interpreter.get());

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
  setupInput(argc, argv, cpu_interpreter,is_use_cache_mode);

  // Run inference
  TFLITE_MINIMAL_CHECK(cpu_interpreter->Invoke() == kTfLiteOk);

  // Get performance
  {
    const uint32_t loop_cout = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < loop_cout; i++) {
      cpu_interpreter->Invoke();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[CPU Performance] Run " << loop_cout << " times, average time: " << (end - start).count() << " ms" << std::endl;
  }

  printf("\n\n=== Post-invoke CPU Interpreter State ===\n");
  tflite::PrintInterpreterState(cpu_interpreter.get());

  auto output_idx_list = npu_interpreter->outputs();
  TFLITE_MINIMAL_CHECK(npu_interpreter->outputs().size() == cpu_interpreter->outputs().size());
  {  // compare result cosine similarity
    for (auto idx = 0; idx < output_idx_list.size(); ++idx) {
      // std::vector<float> result;
      switch (npu_interpreter->output_tensor(idx)->type) {
        case kTfLiteInt8: {
          auto npu_out_buf = npu_interpreter->typed_output_tensor<int8_t>(idx);
          auto cpu_out_buf = cpu_interpreter->typed_output_tensor<int8_t>(idx);
          TFLITE_MINIMAL_CHECK(npu_interpreter->output_tensor(idx)->bytes ==
                               cpu_interpreter->output_tensor(idx)->bytes);

          auto bytes = npu_interpreter->output_tensor(idx)->bytes;
          for (auto j = 0; j < bytes; ++j) {
            int count = 0;
            if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 2 && count < 100)
             {
              std::cout << "[Result mismatch]: Output[" << idx
                        << "], CPU vs NPU("
                        << static_cast<int32_t>(cpu_out_buf[j]) << ","
                        << static_cast<int32_t>(npu_out_buf[j]) << ")"
                        << std::endl;

              count++ ;
            }
          }

          std::vector<int8_t> lhs(bytes);
          auto lquant =
              npu_interpreter->tensor(output_idx_list[idx])->quantization;
          std::vector<int8_t> rhs(bytes);

          memcpy(lhs.data(), cpu_out_buf, bytes);
          memcpy(rhs.data(), npu_out_buf, bytes);

          std::cout << "CosineCosineSimilarity = " << cosine(lhs, rhs)
                    << std::endl;

          break;
        }
        case kTfLiteUInt8: {
          auto npu_out_buf = npu_interpreter->typed_output_tensor<uint8_t>(idx);
          auto cpu_out_buf = cpu_interpreter->typed_output_tensor<uint8_t>(idx);
          TFLITE_MINIMAL_CHECK(npu_interpreter->output_tensor(idx)->bytes ==
                               cpu_interpreter->output_tensor(idx)->bytes);

          auto bytes = npu_interpreter->output_tensor(idx)->bytes;
          for (auto j = 0; j < bytes; ++j) {
            int count = 0;
            if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 2 && count < 100) {
              std::cout << "[Result mismatch]: Output[" << idx
                        << "], CPU vs NPU("
                        << static_cast<int32_t>(cpu_out_buf[j]) << ","
                        << static_cast<int32_t>(npu_out_buf[j]) << ")"
                        << std::endl;

              count++ ;
            }
          }

          std::vector<uint8_t> lhs(bytes);
          auto lquant =
              npu_interpreter->tensor(output_idx_list[idx])->quantization;
          std::vector<uint8_t> rhs(bytes);

          memcpy(lhs.data(), cpu_out_buf, bytes);
          memcpy(rhs.data(), npu_out_buf, bytes);

          std::cout << "CosineCosineSimilarity = " << cosine(lhs, rhs)
                    << std::endl;

          break;
        }
        case kTfLiteFloat32: {
          auto npu_out_buf = npu_interpreter->typed_output_tensor<float>(idx);
          auto cpu_out_buf = cpu_interpreter->typed_output_tensor<float>(idx);
          TFLITE_MINIMAL_CHECK(npu_interpreter->output_tensor(idx)->bytes ==
                               cpu_interpreter->output_tensor(idx)->bytes);

          auto bytes = npu_interpreter->output_tensor(idx)->bytes;
          for (auto j = 0; j < bytes / sizeof(float); ++j) {
            if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) >
                0.001f) {  // TODO{sven}: not accurate
              std::cout << "[Result mismatch]: Output[" << idx
                        << "], CPU vx NPU(" << cpu_out_buf[j] << ","
                        << npu_out_buf[j] << ")" << std::endl;
            }
          }

          std::vector<float> lhs(bytes / sizeof(float));
          std::vector<float> rhs(bytes / sizeof(float));

          memcpy(lhs.data(), cpu_out_buf, bytes);
          memcpy(rhs.data(), npu_out_buf, bytes);

          std::cout << "CosineCosineSimilarity = " << cosine(lhs, rhs)
                    << std::endl;

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
