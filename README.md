# TfLite-vx-delegate
TfLite-vx-delegate constructed with TIM-VX as an openvx delegate for tensorflow lite. Before vx-delegate, you may have nnapi-linux version from Verisilicon, we suggest you move to this new delegate because:

    1. without nnapi, it's flexible to enable more AI operators.
    2. vx-delegate is opensourced, and will promised compatible with latest tensorflow release(currently v2.8.0).
# Use tflite-vx-delegate

## Prepare source code
```sh
mkdir wksp && cd wksp
# tim-vx is optional, it will be downloaded by CMake automatically for none-cross build
# if you want to do cross build with cmake, you have to build tim-vx firstly
git clone https://github.com/VeriSilicon/TIM-VX.git tim-vx
git clone https://github.com/VeriSilicon/tflite-vx-delegate.git
# tensorflow is optional, it will be downloaded automatically if not present
git clone https://github.com/tensorflow/tensorflow.git
```
# Build from source with cmake

```sh
# default built for x86-64 simulator
cd tflite-vx-delegate
mkdir build && cd build
cmake ..
make vx_delegate -j12

# benchmark_model
make benchmark_model -j12
# label_image
make lable_image -j12
```
If you would like to build with your own vivante driver sdk and tim-vx build, you need do cross-build as
```sh
cd tim-vx
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<toolchain.cmake> -DEXTERNAL_VIV_SDK=<sdk_root>
# we can also build from a specific ovxlib instead of use default one by set
# TIM_VX_USE_EXTERNAL_OVXLIB=ON
# OVXLIB_INC=<direct_to_ovxlib_include>
# OVXLIB_LIB=<full_patch_to_libovxlib.so>
```

If you would like to build using local version of tensorflow, you can use `FETCHCONTENT_SOURCE_DIR_TENSORFLOW` cmake variable. Point this variable to your tensorflow tree. For additional details on this variable please see the [official cmake documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html#command:fetchcontent_populate)

``` sh
cmake -DFETCHCONTENT_SOURCE_DIR_TENSORFLOW=/my/copy/of/tensorflow \
    -DOTHER_CMAKE_DEFINES...\
    ..
```
After cmake execution completes, build and run as usual. Beware that cmake process will apply a patch to your tensorflow tree. The patch is requred to enable the external delegate support and the NBG support.

## Enable external delegate support in benchmark_model/label_image

For tensorflow v2.8.0, addtional patch `pwd`/patches/0001-TensorFlow-V280-Enable-External-Delegate.patch requred to enable enable external delegate in benchmark_model/label_image.
If tensorflow source code downloaded by cmake, you can find it in <build_output_dir>/_deps/tensorflow-src

The patch get merged into Tensorflow master branch, no patch required for master branch.

## benchmark_model/Label_image compatible with Tflite+NBG
With our Acuity Toolkit, you can generate tflite file with compiled NBG(**N**etwork **B**inary **G**raph) as a custom operator. To support this special format, you should build benchmark_model/label_image from our delegate repo not use the offical one.

## Run
```sh
# For default x86 build, you can find prebuilt sdk from tim-vx
# export VSIMULATOR_CONFIG=<your_target_npu_id> for x86-simulator
export VIVANTE_SDK_DIR=<direct_to_sdk_root>
# Please copy libtim-vx.so to drivers/ directory
export LD_LIBRARY_PATH=${VIVANTE_SDK_DIR}/drivers:$LD_LIBRARY_PATH # the "drivers" maybe named as lib
./benchmark_model --external_delegate_path=<patch_to_libvx_delegate.so> --graph=<tflite_model.tflite>
# If you would like to use cache mode which save and load binary graph in local disk
./benchmark_model --external_delegate_path=<patch_to_libvx_delegate.so> \
                  --external_delegate_path='allowed_cache_mode:true;cache_file_path:<cache_file>' \
                  --graph=<tflite_model.tflite>
```

## Test
Introduced unit test with tensorflow keras api and convert it to tflite with quantized or none-quantized model,
Golden generated from CPU implementation of tflite
[Details for run test](./test/python/READMME.md)

[Model verification script](./test/python/run_model.py) to compare NPU result with CPU result

# Examples
examples/python/label_image.py
modified based on [offical label_image](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py)

    1. build tensorflow-lite runtime python package follow by [offical build instruction](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/pip_package#readme)
    2. Added "-e" option to provide external provider, [Offical Label Image Instruction](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/README.md)

examples/minimal
modified based on [offical minimal](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/lite/examples/minimal/)

```sh
minimal <patch_to_libvx_delegate.so> <tflite_model.tflite>
# If you would like to use cache mode which save and load binary graph in local disk
minimal <patch_to_libvx_delegate.so> <tflite_model.tflite> use_cache_mode <cache_file>
```
