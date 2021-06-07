// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include "cuda/event.hpp"

namespace CUDAPlugin::utils {
/**
 * @brief class PerformaceTiming measures time between two events
 * and accumulates results from sequential start/stop calls
 */
class PerformaceTiming {
public:
  PerformaceTiming() = default;
  PerformaceTiming(const CUDA::Stream& stream) : start_{CUDA::Event{stream}} {}
  void setStart(const CUDA::Stream& stream) {
    start_.emplace(CUDA::Event{stream});
  }
  void setStop(const CUDA::Stream& stream) {
    stop_.emplace(CUDA::Event{stream});
  }
  float measure() {
    if (start_.has_value() && stop_.has_value()) {
      auto elapsed = stop_->elapsedSince(*start_);
      if (elapsed != std::numeric_limits<float>::quiet_NaN()) {
        duration_ += stop_->elapsedSince(*start_);
      }
    }
    clear();
    return duration_;
  }
  float duration() const noexcept {
    return duration_;
  }
  void clear() {
    start_.reset();
    stop_.reset();
  }

private:
  std::optional<CUDA::Event> start_ {};
  std::optional<CUDA::Event> stop_ {};
  float duration_ {};
};
} // namespace CUDAPlugin::utils
