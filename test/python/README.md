# How to build python package tfilte_runtime

[Office build guide can be found at here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/pip_package/README.md)

# Run with vx_delegate library
```sh
# setup LD_LIBRARY_PATH/VIVANTE_SDK_DIR/VSIMULATOR_CONFIG properly
# run test case with pytest
pytest test_conv2d.py --external_delegate <full path to your libvx_delegate.so>
# - run single test with -k
pytest -k test_conv2d[True-1-1-224-224-3-3-1] test_conv2d.py --external_delegate <full path to your libvx_delegate.so>
# - collect test case with --co
```

# Options
--save_test_model=<directory_to_save_test model in tflite format>
