# TfLite-vx-delegate
TfLite-vx-delegate constructed with TIM-VX as an openvx delegate for tensorflow lite.
# Use tflite-vx-delegate
## Get source code of tflite-vx-delegate and tim-vx
```sh
git clone https://github.com/VeriSilicon/tflite-vx-delegate.git
cd tflite-vx-delegate
git clone https://github.com/VeriSilicon/TIM-VX.git tim-vx
```
## build from source
```sh
bazel build vx_delegate.so --experimental_repo_remote_exec
```
## build benchmark_model
```sh
bazel build @org_tensorflow//tensorflow/lite/tools/benchmark:benchmark_model --experimental_repo_remote_exec
cp bazel-bin/external/org_tensorflow/tensorflow/lite/tools/benchmark/benchmark_model bazel-bin
```
## run with benchmark_model
```sh
export VIVANTE_SDK_DIR=`pwd`/tim-vx/prebuilt-sdk/x86_64_linux
export LD_LIBRARY_PATH=${VIVANTE_SDK_DIR}/lib/:${LD_LIBRARY_PATH}
./bazel-bin/benchmark_model --external_delegate_path=bazel-bin/vx_delegate.so --graph=/path/to/your/model.tflite
```
# Build with cmake
```sh
mkdir build && cd build
cmake ..
make vx_delegate -j4
```