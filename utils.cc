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

#include "utils.h"
#include "tensorflow/lite/minimal_logging.h"

using namespace tflite;

namespace vx {
namespace delegate {
namespace utils {

// transpose channel_dim while doing transpose operation.
int32_t TransposeChannelDim(const std::vector<uint32_t>& perm,
                                   int32_t channel_dim) {
  if (channel_dim < 0) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "invalid channel_dim");
    return -1;
  }
  for (uint32_t i = 0; i < perm.size(); i++) {
    if (channel_dim == perm.at(i)) {
      return i;
    }
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Can't find channle_dim");
  return -1;
}

// Convert the perm in TfLite to the perm in vx-delegate when transpose.
std::vector<uint32_t> GetOvxTransposePerm(const std::vector<uint32_t>& perm) {
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


void GenerateWeightsDataForBilinear(float* data,
                                    const std::vector<uint32_t>& weight_shape,
                                    uint32_t scale_w,
                                    uint32_t scale_h) {
  int32_t width = weight_shape[0];
  int32_t height = weight_shape[1];
  int32_t channel_in = weight_shape[2];
  int32_t channel_out = weight_shape[3];
  for (int o = 0; o < channel_out; o++) {
    for (int h = 0; h < height; h++) {
      float center_w = width % 2 == 1 ? scale_w - 1.0 : scale_w - 0.5;
      float center_h = height % 2 == 1 ? scale_h - 1.0 : scale_h - 0.5;

      for (int w = 0; w < width; w++) {
        data[o * (channel_in + 1) * width * height + h * width + w] =
            (1 - std::abs(w - center_w) / scale_w) *
            (1 - std::abs(h - center_h) / scale_h);
      }
    }
  }

  return;
}

void GenerateWeightDataForNearest(float* data,
                                  const std::vector<uint32_t>& weight_shape) {
  uint32_t width = weight_shape[0];
  uint32_t height = weight_shape[1];
  uint32_t channel_in = weight_shape[2];
  uint32_t channel_out = weight_shape[3];

  for (int o = 0; o < channel_out; o++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        data[o * (channel_in + 1) * width * height + h * width + w] = 1;
      }
    }
  }

  return;
}

static std::map<std::string, std::vector<std::vector<uint8_t>>> cached_data_;

std::vector<uint8_t> ReadData(const char* model_location,
                              const char* filename,
                              size_t input_id,
                              size_t required) {
  if (cached_data_.find(model_location) != cached_data_.end() &&
      input_id < cached_data_[model_location].size()) {
    return cached_data_[model_location][input_id];
  }
  // open the file:
  std::ifstream file(filename, std::ios::binary);

  // Stop eating new lines in binary mode!!!
  file.unsetf(std::ios::skipws);

  // reserve capacity not change size, memory copy in setInput will fail, so use
  // resize()
  std::vector<uint8_t> vec;
  vec.resize(required);

  // read the data:
  file.read(reinterpret_cast<char*>(vec.data()), required);

  if (cached_data_.find(model_location) == cached_data_.end()) {
    std::vector<std::vector<uint8_t>> input_datas;
    input_datas.push_back(vec);
    cached_data_.insert(
        std::make_pair(std::string(model_location), input_datas));
  } else {
    cached_data_[model_location].push_back(vec);
  }
  return vec;
}

std::vector<uint32_t> StringToInt(std::string string)
{
	std::vector <uint32_t> nums;

	int len_s = string.size();
	int i=0, j=0;
	while (i < len_s)
	{
		if (string[i] >= '0'&& string[i] <= '9')
		{
			j = i;
			int len = 0;
			while (string[i] >= '0'&& string[i] <= '9')
			{
				i++;
				len++;
			}
			std::string s0 = string.substr(j, len);
            int num=0;
			std::stringstream s1(s0);
			s1 >> num;
			nums.push_back(num);
		}
		else
		{
			i++;
		}
	}
	return nums;
}

void UnpackConfig(const char* filename,
                   std::vector<std::string>& model_locations,
                   std::vector<uint32_t>& model_num,
                   std::vector<uint32_t>& devs_id,
                   std::vector<std::vector<std::string>>& inputs_datas) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cout << "can not fine this file " << std::endl;
    assert(true);
    return;
  } else {
    std::string string_line;
    while (getline(file, string_line)) {
      if (string_line.empty()) continue;
      char* strs = new char[string_line.length() + 1];
      strcpy(strs, string_line.c_str());

      char* delim = (char*)" ";
      char* p = strtok(strs, delim);

      if (p) {
        std::string s = p;
        model_locations.push_back(s);
        p = strtok(NULL, delim);
      } else {
        std::cout << "wrong model location format in config.txt " << std::endl;
        assert(true);
        return;
      }

      if (p) {
        model_num.push_back(atoi(p));
        p = strtok(NULL, delim);
      } else {
        std::cout << "wrong model number format in config.txt" << std::endl;
        assert(true);
        return;
      }

      if (p) {
        std::string s = p;
        auto nums = StringToInt(s);
        devs_id.push_back(nums[0]);
        p = strtok(NULL, delim);
      } else {
        std::cout << "wrong device Id format in config.txt" << std::endl;
        assert(true);
        return;
      }

      std::vector<std::string> input_datas;
      while(p) {
        std::string s = p;
        if (s == "NULL") {
          input_datas.push_back("");
          std::cout << "Using ramdom input data" << std::endl;
          break;
        } else {
          input_datas.push_back(s);
          p = strtok(NULL, delim);
        }
      }
      inputs_datas.push_back(input_datas);
    }
  }
  return;
}

void CompareInterpreterResult(
    std::unique_ptr<tflite::Interpreter>& cpu_interpreter,
    std::unique_ptr<tflite::Interpreter>& npu_interpreter) {
  auto output_idx_list = npu_interpreter->outputs();
  TFLITE_EXAMPLE_CHECK(npu_interpreter->outputs().size() ==
                       cpu_interpreter->outputs().size());
  for (auto idx = 0; idx < output_idx_list.size(); ++idx) {
    // std::vector<float> result;
    switch (npu_interpreter->output_tensor(idx)->type) {
      case kTfLiteInt8: {
        auto npu_out_buf = npu_interpreter->typed_output_tensor<int8_t>(idx);
        auto cpu_out_buf = cpu_interpreter->typed_output_tensor<int8_t>(idx);
        TFLITE_EXAMPLE_CHECK(npu_interpreter->output_tensor(idx)->bytes ==
                             cpu_interpreter->output_tensor(idx)->bytes);

        auto bytes = npu_interpreter->output_tensor(idx)->bytes;
        for (auto j = 0; j < bytes; ++j) {
          int count = 0;
          if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 2 && count < 100) {
            std::cout << "[Result mismatch]: Output[" << idx << "], CPU vs NPU("
                      << static_cast<int32_t>(cpu_out_buf[j]) << ","
                      << static_cast<int32_t>(npu_out_buf[j]) << ")"
                      << std::endl;

            count++;
          }
        }

        std::vector<int8_t> lhs(bytes);
        auto lquant =
            npu_interpreter->tensor(output_idx_list[idx])->quantization;
        std::vector<int8_t> rhs(bytes);

        memcpy(lhs.data(), cpu_out_buf, bytes);
        memcpy(rhs.data(), npu_out_buf, bytes);

        std::cout << "CosineCosineSimilarity = "
                  << vx::delegate::utils::cosine(lhs, rhs) << std::endl;

        break;
      }
      case kTfLiteUInt8: {
        auto npu_out_buf = npu_interpreter->typed_output_tensor<uint8_t>(idx);
        auto cpu_out_buf = cpu_interpreter->typed_output_tensor<uint8_t>(idx);
        TFLITE_EXAMPLE_CHECK(npu_interpreter->output_tensor(idx)->bytes ==
                             cpu_interpreter->output_tensor(idx)->bytes);

        auto bytes = npu_interpreter->output_tensor(idx)->bytes;
        for (auto j = 0; j < bytes; ++j) {
          int count = 0;
          if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) > 2 && count < 100) {
            std::cout << "[Result mismatch]: Output[" << idx << "], CPU vs NPU("
                      << static_cast<int32_t>(cpu_out_buf[j]) << ","
                      << static_cast<int32_t>(npu_out_buf[j]) << ")"
                      << std::endl;

            count++;
          }
        }

        std::vector<uint8_t> lhs(bytes);
        auto lquant =
            npu_interpreter->tensor(output_idx_list[idx])->quantization;
        std::vector<uint8_t> rhs(bytes);

        memcpy(lhs.data(), cpu_out_buf, bytes);
        memcpy(rhs.data(), npu_out_buf, bytes);

        std::cout << "CosineCosineSimilarity = "
                  << vx::delegate::utils::cosine(lhs, rhs) << std::endl;

        break;
      }
      case kTfLiteFloat32: {
        auto npu_out_buf = npu_interpreter->typed_output_tensor<float>(idx);
        auto cpu_out_buf = cpu_interpreter->typed_output_tensor<float>(idx);
        TFLITE_EXAMPLE_CHECK(npu_interpreter->output_tensor(idx)->bytes ==
                             cpu_interpreter->output_tensor(idx)->bytes);

        auto bytes = npu_interpreter->output_tensor(idx)->bytes;
        for (auto j = 0; j < bytes / sizeof(float); ++j) {
          if (std::abs(npu_out_buf[j] - cpu_out_buf[j]) >
              0.001f) {  // TODO{sven}: not accurate
            std::cout << "[Result mismatch]: Output[" << idx << "], CPU vx NPU("
                      << cpu_out_buf[j] << "," << npu_out_buf[j] << ")"
                      << std::endl;
          }
        }

        std::vector<float> lhs(bytes / sizeof(float));
        std::vector<float> rhs(bytes / sizeof(float));

        memcpy(lhs.data(), cpu_out_buf, bytes);
        memcpy(rhs.data(), npu_out_buf, bytes);

        std::cout << "CosineCosineSimilarity = "
                  << vx::delegate::utils::cosine(lhs, rhs) << std::endl;

        break;
      }
      default: {
        TFLITE_EXAMPLE_CHECK(false);
      }
    }
  }
}

}  // namespace utils
}  // namespace delegate
}  // namespace vx
