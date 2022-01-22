// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/configuration_tests.hpp"
#include <arm_gpu/arm_gpu_config.hpp>

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {{ARM_GPU_CONFIG_KEY(THROUGHPUT_STREAMS), InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}},
    {{ARM_GPU_CONFIG_KEY(THROUGHPUT_STREAMS), InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}},
    {{ARM_GPU_CONFIG_KEY(THROUGHPUT_STREAMS), "8"}},
};

const std::vector<std::map<std::string, std::string>> inconfigs = {
    {{ARM_GPU_CONFIG_KEY(THROUGHPUT_STREAMS), CONFIG_VALUE(NO)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                ::testing::ValuesIn(inconfigs)),
                        IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                ::testing::ValuesIn(inconfigs)),
                        IncorrectConfigAPITests::getTestCaseName);
} // namespace