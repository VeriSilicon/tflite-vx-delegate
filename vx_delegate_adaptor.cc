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
#include <string>
#include <vector>
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "delegate_main.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"

namespace vx {
namespace delegate {

TfLiteDelegate* CreateVxDelegateFromOptions(char** options_keys,
                                               char** options_values,
                                               size_t num_options) {
  VxDelegateOptions options = VxDelegateOptionsDefault();

  // Parse key-values options to VxDelegateOptions by mimicking them as
  // command-line flags.
  const char** argv;
  argv = new const char*[num_options + 1];
  constexpr char kVxDelegateParsing[] = "vx_delegate_parsing";
  argv[0] = kVxDelegateParsing;

  std::vector<std::string> option_args;
  option_args.reserve(num_options);
  for (int i = 0; i < num_options; ++i) {
    option_args.emplace_back("--");
    option_args.rbegin()->append(options_keys[i]);
    option_args.rbegin()->push_back('=');
    option_args.rbegin()->append(options_values[i]);
    argv[i + 1] = option_args.rbegin()->c_str();
  }

  constexpr char kAllowedSaveLoadNBG[] = "allowed_cache_mode";
  constexpr char kAllowedBuiltinOp[] = "allowed_builtin_code";
  constexpr char kReportErrorDuingInit[] = "error_during_init";
  constexpr char kReportErrorDuingPrepare[] = "error_during_prepare";
  constexpr char kReportErrorDuingInvoke[] = "error_during_invoke";

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kAllowedSaveLoadNBG, &options.allowed_cache_mode,
                               "Allowed save load nbg."),
      tflite::Flag::CreateFlag(kAllowedBuiltinOp, &options.allowed_builtin_code,
                               "Allowed builtin code."),
      tflite::Flag::CreateFlag(kReportErrorDuingInit,
                               &options.error_during_init,
                               "Report error during init."),
      tflite::Flag::CreateFlag(kReportErrorDuingPrepare,
                               &options.error_during_prepare,
                               "Report error during prepare."),
      tflite::Flag::CreateFlag(kReportErrorDuingInvoke,
                               &options.error_during_invoke,
                               "Report error during invoke."),
  };

  int argc = num_options + 1;
  if (!tflite::Flags::Parse(&argc, argv, flag_list)) {
    return nullptr;
  }

  TFLITE_LOG(INFO) << "Vx delegate: allowed_cache_mode set to "
                   << options.allowed_cache_mode << ".";
  TFLITE_LOG(INFO) << "Vx delegate: allowed_builtin_code set to "
                   << options.allowed_builtin_code << ".";
  TFLITE_LOG(INFO) << "Vx delegate: error_during_init set to "
                   << options.error_during_init << ".";
  TFLITE_LOG(INFO) << "Vx delegate: error_during_prepare set to "
                   << options.error_during_prepare << ".";
  TFLITE_LOG(INFO) << "Vx delegate: error_during_invoke set to "
                   << options.error_during_invoke << ".";

  if (options.allowed_cache_mode) {
    for (int i = 0; i < num_options; ++i) {
      if(strcmp(options_keys[i],"cache_file_path") == 0){
        options.cache_file_path = options_values[i];
        break;
      }
    }
  }

  delete []argv;
  return VxDelegateCreate(&options);
}

}  // namespace delegate
}  // namespace vx

extern "C" {

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return vx::delegate::CreateVxDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  vx::delegate::VxDelegateDelete(delegate);
}

}  // extern "C"
