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
If you would like to build using local version of tensorflow, you can use `FETCHCONTENT_SOURCE_DIR_TENSORFLOW` cmake variable. Point this variable to your tensorflow tree. For additional details on this variable please see the [official cmake documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html#command:fetchcontent_populate)

``` sh
cmake -DFETCHCONTENT_SOURCE_DIR_TENSORFLOW=/my/copy/of/tensorflow \
    -DOTHER_CMAKE_DEFINES...\
    ..
```
After cmake execution completes, build and run as usual. Beware that cmake process will apply a patch to your tensorflow tree. The patch is requred to enable the external delegate support and the NBG support.

# Examples
examples/python/label_image.py
modified based on [offical label_image](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py)

    1. build tensorflow-lite runtime python package follow by [offical build instruction](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/pip_package#readme)
    2. Added "-e" option to provide external provider, [Offical Label Image Instruction](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/README.md)

examples/minimal
modified based on [offical minimal](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/lite/examples/minimal/)

```sh
minimal libvx_delegate.so mobilenet_v2_1.0_224_quant.tflite
```
