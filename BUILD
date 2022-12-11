load("@rules_python//python:defs.bzl", "py_binary")
load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")

py_binary(
    name = "first",
    srcs = glob(["src/*.py"]),
    #    data = [":transform"],  # a cc_binary which we invoke at run time
    #    deps = [
    #        ":foolib",  # a py_library
    #    ],
)

#py_binary(
#    name = "second",
#    srcs = ["src/second.py"],
#    visibility = ["//src/first.py"],
#)

#py_test(
#    name = "all tests",
#    srcs = glob(["src/*.test.py"]),
#)

#py_library(
#    name = "example lib",
#    srcs = ["deps/example_lib"]
#)

buildifier(
    name = "buildifier",
)

load("@rules_python//python/pip_install:requirements.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "--no-emit-index-url",
    ],
    env = {
        "PIP_CONFIG_FILE": "~/.pip/pip.conf",
    },
)
