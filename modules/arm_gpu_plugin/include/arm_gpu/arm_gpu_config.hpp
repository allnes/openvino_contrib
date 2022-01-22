// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for DLIA plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file dlia_config.hpp
 */

#pragma once

#include <string>
#include "ie_plugin_config.hpp"

namespace InferenceEngine {

namespace ArmGpuMetrics {

/**
 * @def ARM_GPU_METRIC_VALUE(name)
 * @brief Shortcut for defining ArmGpu metric values
 */
#define ARM_GPU_METRIC_VALUE(name) InferenceEngine::ArmGpuMetrics::name
#define DECLARE_ARM_GPU_METRIC_VALUE(name) static constexpr auto name = #name

// ! [public_header:metrics]
/**
 * @brief Defines whether current ArmGpu device instance supports hardware blocks for fast convolution computations.
 */
DECLARE_ARM_GPU_METRIC_VALUE(HARDWARE_CONVOLUTION);
// ! [public_header:metrics]

}  // namespace ArmGpuMetrics

namespace ArmGpuConfigParams {

/**
 * @def ARM_GPU_CONFIG_KEY(name)
 * @brief Shortcut for defining ArmGpu device configuration keys
 */
#define ARM_GPU_CONFIG_KEY(name) InferenceEngine::ArmGpuConfigParams::_CONFIG_KEY(ARM_GPU_##name)

#define DECLARE_ARM_GPU_CONFIG_KEY(name) DECLARE_CONFIG_KEY(ARM_GPU_##name)
#define DECLARE_ARM_GPU_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(ARM_GPU_##name)


/**
 * @brief Defines the number of throutput streams used by ARM_GPU plugin.
 */
DECLARE_ARM_GPU_CONFIG_KEY(THROUGHPUT_STREAMS);


}  // namespace ArmGpuConfigParams
}  // namespace InferenceEngine
