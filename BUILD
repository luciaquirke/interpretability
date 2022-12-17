load("@rules_python//python:defs.bzl", "py_binary")

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

load("@my_deps//:requirements.bzl", "requirement")

py_binary(
    name = "main",
    srcs = glob(["src/*.py"]),
    #    data = [":transform"],  # a cc_binary which we invoke at run time
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        requirement("torchaudio"),
        requirement("torchtext"),
        requirement("numpy"),
        requirement("matplotlib"),
        requirement("spacy"),
        requirement("seaborn")
    ]
)

load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")

buildifier(
    name = "buildifier",
)

# Things I could add in the future
#py_binary(
#    name = "second",
#    srcs = ["src/second.py"],
#    visibility = ["//src/main.py"],
#)

#py_test(
#    name = "all tests",
#    srcs = glob(["src/*.test.py"]),
#)

#py_library(
#    name = "example lib",
#    srcs = ["deps/example_lib"]
#)

