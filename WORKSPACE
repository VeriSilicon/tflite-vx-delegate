#
# tflite_delegate
#
# This is a container project to demonstrate how to manage a Tensorflow-lite build
# based on TIM-VX and Tensorflow vx-delegate. By following example provided in this
# WORKSPACE, one should be able to build tflite with acceleration by Verisilicon NPU
# and deploy onto respective target.
#
workspace(name = "tflite_delegate")

#
# Load Bazel tools for different repository rules
#
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# Bazel Skylib and repository rules for Tensorflow 2.3.0. These packages may need to be
# updated if Tensorflow version is updated. Please check Tensorflow WORKSPACE and copy
# over the related packages
#
http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz"],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

#
# Include Tensorflow as a git_repository. You can change this to be a local_repository or http_archive
#
git_repository(
    name = 'org_tensorflow',
    branch = 'dev/vx-delegate',
    remote = 'https://github.com/verisilicon/tensorflow',
)

#
# Give a name to the local tf_workspace
#
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow"
)

#
# Include TIM-VX as a git_repository. Change to local_repository or http_archive based on need.
#
git_repository(
    name = 'tim_vx',
    branch = 'main',
    remote = 'https://github.com/verisilicon/TIM-VX',
)

#
# external_viv_sdk defines the prebuilt-SDK package for a paticular NPU target. The name must
# be "external_viv_sdk" and can not be changed to anything else. TIM-VX build system will
# use external_viv_sdk path for linking if use_external_viv_sdk config is specified.
#
# For example: TFLite cross compilation for my_platform (.bazelrc)
#   build:my_platform --crosstool_top=@local_config_embedded_arm//:toolchain
#   build:my_platform --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
#   build:my_platform --cpu=aarch64
#   build:my_platform --define use_external_viv_sdk=true
#
http_archive(
    name = "external_viv_sdk",
    sha256 = "9c3fe033f6d012010c92ed1f173b5410019ec144ddf68cbc49eaada2b4737e7f",
    strip_prefix = "aarch64_A311D_D312513_A294074_R311680_T312233_O312045",
    urls = [
        "https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz",
    ],
)

