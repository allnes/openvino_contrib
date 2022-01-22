# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(openvino_arm_gpu_plugin)

set(ARM_GPU_PLUGIN_SOURCE_DIR ${openvino_arm_gpu_plugin_SOURCE_DIR})

find_package(OpenVINODeveloperPackage REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()

# install
ie_cpack(gpu)

