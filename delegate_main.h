/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/tensor.h"

namespace vx {
namespace delegate {

TfLiteDelegate* VxDelegate();

class Delegate;

struct OpData {
  std::vector<int> subgraph_inputs;
  std::vector<int> subgraph_outputs;
  std::vector<int> subgraph_states;

  std::unique_ptr<Delegate> delegate;
};

typedef struct {
  // Allowed ops to delegate.
  int allowed_builtin_code;
  // Report error during init.
  bool error_during_init;
  // Report error during prepare.
  bool error_during_prepare;
  // Report error during invoke.
  bool error_during_invoke;
} VxDelegateOptions;

VxDelegateOptions VxDelegateOptionsDefault();

TfLiteDelegate* VxDelegateCreate(const VxDelegateOptions* options);

void VxDelegateDelete(TfLiteDelegate* delegate);
class Delegate {
 public:
  static TfLiteDelegate* Create();
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
  std::vector<std::shared_ptr<tim::vx::Operation>>& GetOps() { return ops_; }
  std::shared_ptr<tim::vx::Graph>& GetGraph() { return graph_; }
  std::vector<std::shared_ptr<tim::vx::Tensor>>& GetTensors() {
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

  std::shared_ptr<tim::vx::Context> context_;
  std::shared_ptr<tim::vx::Graph> graph_;
  //first: layout infered graph; second: map from src_tensor to infered_tensor.
  std::pair<std::shared_ptr<tim::vx::Graph>,
          std::map<std::shared_ptr<tim::vx::Tensor>,
                   std::shared_ptr<tim::vx::Tensor>>> layout_infered_;
  std::vector<std::shared_ptr<tim::vx::Tensor>> tensors_;
  std::vector<std::shared_ptr<tim::vx::Tensor>> state_tensors_;
  std::vector<std::shared_ptr<tim::vx::Operation>> ops_;
  std::vector<OperationDataType> operations_;
  bool compiled_;
};

}  // namespace delegate
}  // namespace vx

#endif /* TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_DELEGATE_MAIN_H */
