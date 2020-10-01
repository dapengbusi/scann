workspace(name = "scann")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//build_deps/py:python_configure.bzl", "python_configure")
#load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")

#tf_configure(
#    name = "local_config_tf",
#)

#load("//deps:repo.bzl", "tensorflow_http_archive")

http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-ee598066c4cb31ec5ed3106e61ba99ef004a4bae",
    url = "https://github.com/tensorflow/tensorflow/archive/ee598066c4cb31ec5ed3106e61ba99ef004a4bae.zip",
)
#local_repository(
#    name = "org_tensorflow",
#    path = "/home/code/scann/tensorflow",
#)

http_archive(
    name = "rules_pkg",
    sha256 = "352c090cc3d3f9a6b4e676cf42a6047c16824959b438895a76c2989c6d7c246a",
    url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz",
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load(
    "@org_tensorflow//third_party/toolchains/preconfig/generate:archives.bzl",
    "bazel_toolchains_archive",
)

bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()

# START: Upstream TensorFlow dependencies
# TensorFlow build depends on these dependencies.
# Needs to be in-sync with TensorFlow sources.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)
http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)  # https://github.com/bazelbuild/bazel-skylib/releases

# END: Upstream TensorFlow dependencies

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace()


python_configure(name = "local_config_python")

git_repository(
    name = "pybind11_bazel",
    commit = "f22df0e57ba664c2d3cf439ddfb7f8804e3f36c1",
    remote = "https://github.com/pybind/pybind11_bazel.git",
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.4.3",
    urls = [
        "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
    ],
)

http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp",
    urls = [
        "https://storage.googleapis.com/scann/abseil-cpp.tar.gz",
    ],
)

git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    tag = "v3.8.0",
)

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.8.1",
)

## rules_cc defines rules for generating C++ code from Protocol Buffers.
#http_archive(
#    name = "rules_cc",
#    sha256 = "35f2fb4ea0b3e61ad64a369de284e4fbbdcdba71836a5555abb5e194cf119509",
#    strip_prefix = "rules_cc-624b5d59dfb45672d4239422fa1e3de1822ee110",
#    urls = [
#        "https://github.com/bazelbuild/rules_cc/archive/624b5d59dfb45672d4239422fa1e3de1822ee110.tar.gz",
#    ],
#)
#
## rules_proto defines abstract rules for building Protocol Buffers.
#http_archive(
#    name = "rules_proto",
#    sha256 = "57001a3b33ec690a175cdf0698243431ef27233017b9bed23f96d44b9c98242f",
#    strip_prefix = "rules_proto-9cd4f8f1ede19d81c6d48910429fe96776e567b1",
#    urls = [
#        "https://github.com/bazelbuild/rules_proto/archive/9cd4f8f1ede19d81c6d48910429fe96776e567b1.tar.gz",
#    ],
#)

#load("@rules_cc//cc:defs.bzl", "cc_library")

#load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")
#
#rules_cc_dependencies()
#
#load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
#
#rules_proto_dependencies()
#
#rules_proto_toolchains()
