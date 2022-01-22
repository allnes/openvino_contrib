// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arm_gpu_async_infer_request.hpp"

#include "arm_gpu_itt.hpp"

using namespace ArmGpuPlugin;

// ! [async_infer_request:ctor]
ArmGpuAsyncInferRequest::ArmGpuAsyncInferRequest(const ArmGpuInferRequest::Ptr& inferRequest,
                                                     const InferenceEngine::ITaskExecutor::Ptr& cpuTaskExecutor,
                                                     const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                                                     const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, cpuTaskExecutor, callbackExecutor),
      _inferRequest(inferRequest),
      _waitExecutor(waitExecutor) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::Infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = false;

    if (remoteDevice) {
        _pipeline = {{cpuTaskExecutor,
                      [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::ArmGpuPlugin,
                                             "ArmGpuAsyncInferRequest::PreprocessingAndStartPipeline");
                          _inferRequest->inferPreprocess();
                          _inferRequest->startPipeline();
                      }},
                     {_waitExecutor,
                      [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::ArmGpuPlugin, "ArmGpuAsyncInferRequest::WaitPipeline");
                          _inferRequest->waitPipeline();
                      }},
                     {cpuTaskExecutor, [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::ArmGpuPlugin, "ArmGpuAsyncInferRequest::Postprocessing");
                          _inferRequest->inferPostprocess();
                      }}};
    }
}
// ! [async_infer_request:ctor]

// ! [async_infer_request:dtor]
ArmGpuAsyncInferRequest::~ArmGpuAsyncInferRequest() {
    InferenceEngine::AsyncInferRequestThreadSafeDefault::StopAndWait();
}
// ! [async_infer_request:dtor]
