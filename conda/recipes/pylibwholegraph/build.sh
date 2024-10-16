#!/usr/bin/env bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

CMAKE_EXTRA_ARGS="--cmake-args=\"-DBUILD_OPS_WITH_TORCH_C10_API=OFF\""

./build.sh pylibwholegraph --allgpuarch -v ${CMAKE_EXTRA_ARGS}
