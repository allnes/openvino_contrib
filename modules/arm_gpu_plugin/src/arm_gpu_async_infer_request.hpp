// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>

#include "arm_gpu_infer_request.hpp"

namespace ArmGpuPlugin {

// ! [async_infer_request:header]
class ArmGpuAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    ArmGpuAsyncInferRequest(const ArmGpuInferRequest::Ptr& inferRequest,
                              const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                              const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                              const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~ArmGpuAsyncInferRequest();

private:
    ArmGpuInferRequest::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};
// ! [async_infer_request:header]

}  // namespace ArmGpuPlugin
