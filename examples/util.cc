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

#include "util.h"

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