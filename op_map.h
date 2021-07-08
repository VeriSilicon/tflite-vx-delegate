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

#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_OP_MAP_H_
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_OP_MAP_H_

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "delegate_main.h"
#include "tim/vx/operation.h"

namespace vx {
namespace op_map {

struct IOpMapper {
  IOpMapper() {}
  virtual ~IOpMapper() {}

  virtual bool IsSupported(TfLiteContext* context,
                           TfLiteNode* node,
                           const TfLiteRegistration* registration) const {
    return true;
  }

  virtual bool GetStateTensorIndexes(TfLiteContext* context,
                                     TfLiteNode* node,
                                     const TfLiteRegistration* registration,
                                     std::vector<int>& states) const {
    return false;
  }

  virtual size_t GetParamSize() const { return 0; }

  virtual bool MapOp(vx::delegate::Delegate* delegate,
                     std::vector<std::shared_ptr<tim::vx::Tensor>> inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>> outputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>> states,
                     const void* params) = 0;
};

using OperationMapItemType = std::map<int, std::unique_ptr<IOpMapper>>;
using CustomOperationMapItemType =
    std::map<std::string, std::unique_ptr<IOpMapper>>;

const OperationMapItemType& SupportedBuiltinOps();
const CustomOperationMapItemType& SupportedBuiltinCustomOps();

}  // namespace op_map
}  // namespace vx
#endif /* TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_OP_MAP_H_ */
