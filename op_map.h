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
