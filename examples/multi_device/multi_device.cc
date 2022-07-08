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
#include <stdlib.h>
#include <iostream>
#include <functional>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/minimal_logging.h"

#include "vsi_npu_custom_op.h"
#include "utils.h"

// This is an example that is multi device to run model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the multi device makefile. This example must remain trivial to build with
// the multi device build tool.
//
// Usage: multi_device <vxdelegate.so> <config path>


void setupInput(const char* model_location,
                std::vector<std::string> input_files,
                const std::unique_ptr<tflite::Interpreter>& interpreter) {
  auto input_list = interpreter->inputs();
  bool use_random_input = false;
  if(input_files.size() == 1 && input_files[0].size() == 0){
    use_random_input = true;
  }

  for (auto input_idx = 0; input_idx < input_list.size(); input_idx++) {
    auto in_tensor = interpreter->input_tensor(input_idx);

    std::cout << "Setup intput[" << std::string(interpreter->GetInputName(input_idx)) << "]" << std::endl;
    const char* input_data =  use_random_input ? "/dev/urandom" : input_files[input_idx].c_str();

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
        auto in = vx::delegate::utils::ReadData(model_location, input_data, input_idx, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<float>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteUInt8:
      {
        auto in = vx::delegate::utils::ReadData(model_location, input_data, input_idx, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<uint8_t>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteInt8: {
        auto in = vx::delegate::utils::ReadData(model_location, input_data, input_idx, in_tensor->bytes);
        memcpy(interpreter->typed_input_tensor<int8_t>(input_idx), in.data(), in.size());
        break;
      }
      default: {
        std::cout << "Fatal: datatype for input not implemented" << std::endl;
        TFLITE_EXAMPLE_CHECK(false);
        break;
      }
    }
  }
}

void runSingleWork(const char* model_location,
                   std::vector<std::string> input_files,
                   TfLiteExternalDelegateOptions options) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_location);
  TFLITE_EXAMPLE_CHECK(model != nullptr);
  auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&options);

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> npu_interpreter;
  builder(&npu_interpreter);

  TFLITE_EXAMPLE_CHECK(npu_interpreter != nullptr);
  npu_interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);

  TFLITE_EXAMPLE_CHECK(npu_interpreter->AllocateTensors() == kTfLiteOk);
  TFLITE_LOG(tflite::TFLITE_LOG_INFO, "=== Pre-invoke NPU Interpreter State ===");
  setupInput(model_location, input_files, npu_interpreter);
  
  TFLITE_EXAMPLE_CHECK(npu_interpreter->Invoke() == kTfLiteOk);
  
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
  setupInput(model_location, input_files, cpu_interpreter);

  // Run inference
  TFLITE_EXAMPLE_CHECK(cpu_interpreter->Invoke() == kTfLiteOk);

  TFLITE_LOG(tflite::TFLITE_LOG_INFO, "=== Post-invoke CPU Interpreter State ===");

  tflite::PrintInterpreterState(cpu_interpreter.get());

  vx::delegate::utils::CompareInterpreterResult(cpu_interpreter,npu_interpreter);
  TfLiteExternalDelegateDelete(ext_delegate_ptr);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    TFLITE_LOG(tflite::TFLITE_LOG_ERROR,
               "multi device demo <external_delegate.so> <config.txt file>");
    return 1;
  }

  const char* delegate_so = argv[1];
  const char* configfile = argv[2];

  std::vector<std::string> model_locations;
  std::vector<uint32_t> repeat_num;
  std::vector<uint32_t> devs_id;
  std::vector<std::vector<std::string>> inputs_data_files;
  vx::delegate::utils::UnpackConfig(
      configfile, model_locations, repeat_num, devs_id, inputs_data_files);

  for (size_t i = 0; i < model_locations.size(); i++) {
    for (size_t j = 0; j < repeat_num[i]; j++) {
      TfLiteExternalDelegateOptions options =
          TfLiteExternalDelegateOptionsDefault(delegate_so);
      const char* allow_multi_device_key = "allowed_multi_device_mode";
      const char* allow_multi_device_value = "true";
      const char* device_id_key = "device_id";
      const char* device_id_value = std::to_string(devs_id[i]).c_str();
      const char* model_localtion_key = "model_location";
      const char* model_location_value = model_locations[i].c_str();

      options.insert(
          &options, allow_multi_device_key, allow_multi_device_value);
      options.insert(&options, device_id_key, device_id_value);
      options.insert(&options, model_localtion_key, model_location_value);

      runSingleWork(model_locations[i].c_str(), inputs_data_files[i], options);
    }
  }
  return 0;
}
