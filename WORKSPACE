workspace(name = "tflite_vx_delegate")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

"""Loads TensorFlow."""
http_archive(
    name = "org_tensorflow",
    urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.5.0.tar.gz"],
    sha256 = "233875ea27fc357f6b714b2a0de5f6ff124b50c1ee9b3b41f9e726e9e677b86c",
    strip_prefix = "tensorflow-2.5.0"
)
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

"""Loads Verisilicon TIM_VX."""
# http_archive(
#     name  = "tim_vx",
#     urls = ["https://github.com/VeriSilicon/TIM-VX/archive/refs/tags/v1.1.30.3.tar.gz"],
#     sha256 = "2c931684658d68fc51853f3d6ccad05b672f67f03b5c75bb634fbd88e9a568ee",
#     strip_prefix = "TIM-VX-1.1.30.3"
# )

# Uncomment for local development
local_repository(
    name = "tim_vx",    
    path = "tim-vx",
)
