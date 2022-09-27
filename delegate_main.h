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


#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

#include "absl/types/optional.h"
#include "vsi_npu_custom_op.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/interpreter.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/tensor.h"

#ifdef MULTI_DEVICE_FEATURE_MODE
  #include "tim/vx/platform/platform.h"
  #include "tim/vx/platform/native.h"
#endif

namespace vx {
namespace delegate {

typedef struct {
  //Allowed save or load nbg binary
  bool allowed_cache_mode;
  //Device in multi device mode
  int32_t device_id;
  //nbg binary path
  std::string cache_file_path;
  // Allowed ops to delegate.
  int allowed_builtin_code;
  // Report error during init.
  bool error_during_init;
  // Report error during prepare.
  bool error_during_prepare;
  // Report error during invoke.
  bool error_during_invoke;
} VxDelegateOptions;


class Delegate;

struct OpData {
  std::vector<int> subgraph_inputs;
  std::vector<int> subgraph_outputs;
  std::vector<int> subgraph_states;

  std::unique_ptr<Delegate> delegate;
};

struct DerivedDelegateData {
    TfLiteDelegate parent;
    bool allow_cache_mode;
    int32_t device_id;
    std::string cache_path;
};

TfLiteDelegate* VxDelegate(const VxDelegateOptions* options);

VxDelegateOptions VxDelegateOptionsDefault();

TfLiteDelegate* VxDelegateCreate(const VxDelegateOptions* options);

void VxDelegateDelete(TfLiteDelegate* delegate);
class Delegate {
 public:
  static TfLiteDelegate* Create(const VxDelegateOptions* options);
  static bool SupportedOp(TfLiteContext* context,
                          TfLiteNode* node,
                          const TfLiteRegistration* registration);

  Delegate();
  ~Delegate() {}

  std::unique_ptr<OpData> Init(TfLiteContext* context,
                               const TfLiteDelegateParams* params);
  TfLiteStatus Prepare(const OpData& op_data,
                       TfLiteContext* context,
                       TfLiteNode* node);
  TfLiteStatus Invoke(const OpData& op_data,
                      TfLiteContext* context,
                      TfLiteNode* node);
  void CreateCacheOp(const OpData& op_data);

  std::vector<std::shared_ptr<tim::vx::Operation>>& GetOps() { return ops_; }
  int GetOperationOutput(uint32_t index) { return op_info_.outputs[index]; }
  std::shared_ptr<tim::vx::Graph>& GetGraph() { return graph_; }
  std::map<int32_t, std::shared_ptr<tim::vx::Tensor>>& GetTensors() {
    return tensors_;
  }

 private:
  struct OperationDataType {
    int builtin_code;
    std::string custom_name;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<int> states;
    std::vector<uint8_t> builtin_data;
  };

#ifdef MULTI_DEVICE_FEATURE_MODE
  std::vector<std::shared_ptr<tim::vx::platform::IDevice>> devices_;
  std::shared_ptr<tim::vx::platform::IExecutor> executor_;
  std::shared_ptr<tim::vx::platform::IExecutable> executable_;
  std::vector<std::shared_ptr<tim::vx::platform::ITensorHandle>> inputs_;
  std::vector<std::shared_ptr<tim::vx::platform::ITensorHandle>> outputs_;
#endif

  std::shared_ptr<tim::vx::Context> context_;
  std::shared_ptr<tim::vx::Graph> graph_;
  //first: layout infered graph; second: map from src_tensor to infered_tensor.
  std::pair<std::shared_ptr<tim::vx::Graph>,
          std::map<std::shared_ptr<tim::vx::Tensor>,
                   std::shared_ptr<tim::vx::Tensor>>> layout_infered_;
  std::map<int32_t, std::shared_ptr<tim::vx::Tensor>> tensors_;
  int32_t placeholder_tensor_idx_{-2};
  std::map<int32_t, std::shared_ptr<tim::vx::Tensor>> state_tensors_;
  std::vector<std::shared_ptr<tim::vx::Operation>> ops_;
  std::vector<OperationDataType> operations_;
  struct OperationDataType op_info_;
  bool compiled_;

  absl::optional<bool> is_cache_present_;
  uint32_t device_id_;

  size_t nbg_size_;
  std::fstream fs_;
};

}  // namespace delegate
}  // namespace vx

#endif /* TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H */
