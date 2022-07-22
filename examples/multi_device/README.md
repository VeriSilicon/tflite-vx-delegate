# TensorFlow Lite C++ multi device example

This example shows how you can build and run  TensorFlow Lite models on multi device. The models located in: (https://github.com/sunshinemyson/TIM-VX/releases)

#### Step 1. Build

1. Trun option TFLITE_ENABLE_MULTI_DEVICE to On in ./CMakeLists.txt
2. Only 40 bit driver support this feature, EXTERNAL_VIV_SDK should be setted to point to 40 bit driver location when build cmake
3. If using TIM_VX_INSTALL，TIM_VX should open TIM_VX_ENABLE_PLATFORM

#### Step 2. Run

    The config.txt is used for store models information.Every line repreasents one model information, the format is:

    model_location   run_repeat_num   [device_id]   input_data

   If input_data is NULL, we will run model with random data. for example:

    ${WORKESPACE}/mobilenet_v2_quant.tflite 1 [3] NULL
    ${WORKESPACE}/inception_v3_quant.tflite  1 [0]  ./input_data.bin

```sh
export VSIMULATOR_CONFIG=VIP9400O_PID0XD9
export VIV_VX_ENABLE_VA40=1
export NBG_40BIT_VA_SUPPORT=1
export VIV_MGPU_AFFINITY=1:0
export VIV_OVX_USE_MULTI_DEVICE=1:1
export VIVANTE_SDK_DIR=${40_bit_driver_location}
export LD_LIBRARY_PATH=${40_bit_driver_location}/lib:$LD_LIBRARY_PATH
./multi_device <patch_to_libvx_delegate.so> <config.txt>
```
