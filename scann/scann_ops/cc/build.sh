#!/bin/bash
#CC=clang bazel build -c opt --copt=-g --copt=-mavx2 --copt=-mfma --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --cxxopt="-std=c++17" --cxxopt="-g" --copt=-fsized-deallocation --copt=-w scann_ops
#CC=clang bazel build -c opt --copt=-g --copt=-mavx2 --copt=-mfma --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --cxxopt="-std=c++17" --cxxopt="-g" --copt=-fsized-deallocation --copt=-w scann_ext
CC=clang bazel build -c opt --copt=-g --copt=-mavx2 --copt=-mfma --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --cxxopt="-std=c++17" --cxxopt="-g" --copt=-fsized-deallocation --copt=-w scann_test
