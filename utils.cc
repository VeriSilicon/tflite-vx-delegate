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

#ifdef NODE_TRACE_DB_MODE
#include "json/json.h"
#endif

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

#ifdef NODE_TRACE_DB_MODE
void MapTfliteNodeToTimVxNode(
    const std::vector<std::shared_ptr<tim::vx::Operation>>& before_op_vector,
    const std::vector<std::shared_ptr<tim::vx::Operation>>& after_op_vector,
    std::vector<vx::delegate::TfliteNodeIDPair>& tflite_node_id_map) {
  size_t new_operation_size = after_op_vector.size() - before_op_vector.size();
  size_t i = 0;
  std::vector<uint32_t> new_operation;
  if (new_operation_size <= 0 || tflite_node_id_map.size() == 0) {
    return;
  }

  for (i = 0; i < new_operation_size; i++) {
    size_t new_operation_index = before_op_vector.size();
    uint32_t uid = after_op_vector[new_operation_index + i]->uid();
    tflite_node_id_map[tflite_node_id_map.size() - 1].op_uids.push_back(uid);
  }
  return;
}

void GenerateVxNodeTraceDb(
    std::vector<vx::delegate::TfliteNodeIDPair>& tflite_node_id_map) {
  Json::Value root;

  Json::StyledWriter sw;
  uint32_t i = 0;
  std::fstream fs;
  fs.open("vx_node_trace_db.json", std::ios::out | std::ios::trunc);

  for (auto tflite_node_id_pair : tflite_node_id_map) {
    Json::Value tflite_node_uid;
    Json::Value tim_vx_uids;

    Json::Value inputs_ids;
    Json::Value outputs_ids;
    Json::Value tflite_node_builtin_code;

    Json::Value map_pair;
    for (i = 0; i < tflite_node_id_pair.inputs.size(); i++) {
      inputs_ids[i] = tflite_node_id_pair.inputs[i];
    }
    for (i = 0; i < tflite_node_id_pair.outputs.size(); i++) {
      outputs_ids[i] = tflite_node_id_pair.outputs[i];
    }
    tflite_node_builtin_code = tflite_node_id_pair.builtin_code;
    tflite_node_uid["inputs"] = inputs_ids;
    tflite_node_uid["outputs"] = outputs_ids;
    tflite_node_uid["builtin_code"] = tflite_node_id_pair.builtin_code;

    for (i = 0; i < tflite_node_id_pair.op_uids.size(); i++) {
      tim_vx_uids[i] = tflite_node_id_pair.op_uids[i];
    }

    map_pair["tflite_node_id"] = tflite_node_uid;
    map_pair["tim_vx_uid"] = tim_vx_uids;
    root.append(map_pair);
  }

  fs << sw.write(root);
  fs.close();
  return;
}
#endif

}  // namespace utils
}  // namespace delegate
}  // namespace vx
