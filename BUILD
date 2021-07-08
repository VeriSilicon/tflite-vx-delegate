package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vx_delegate",
    copts = ["-std=c++14","-w"],
    srcs = [
        "delegate_main.cc",
        "op_map.cc",
        "utils.cc",
    ],
    hdrs = [
        "delegate_main.h",
        "op_map.h",
        "utils.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels/internal:reference_base",
        "@org_tensorflow//tensorflow/lite/tools:logging",
        "@tim_vx//prebuilt-sdk:VIV_SDK_LIB",
        "@tim_vx//:tim-vx_interface",
    ],
    linkstatic=True,
)

cc_binary(
    name = "vx_delegate.so",
    copts = ["-std=c++14","-w"],
    srcs = [
        "vx_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":vx_delegate",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/tools:command_line_flags",
    ],
)

cc_test(
    name = "vx_delegate_test",
    copts = ["-std=c++14","-w"],
    size = "small",
    srcs = [
        "vx_delegate_test.cc",
    ],
    deps = [
        ":vx_delegate",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:minimal_logging",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/nnapi:nnapi_implementation",
        "@org_tensorflow//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_googletest//:gtest",
    ],
)
