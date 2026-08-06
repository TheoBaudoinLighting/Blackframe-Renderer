#pragma once

#include <Blackframe/XPU/Shared/ConstantTextureAbi.hpp>
#include <cstddef>
#include <cstdint>

// Every pointer names storage on the active CUDA device. Input and output ranges must not overlap.
// The optional opaque handle must be a cudaStream_t created on that device; nullptr preserves
// default-stream ordering. The launch is asynchronous and every lane reports an explicit status.
extern "C" int blackframe_cuda_launch_constant_texture_evaluation(
    const std::uint8_t* scene_bytes, std::size_t scene_size,
    const blackframe::xpu::shared::ConstantTextureEvaluationRequest* requests,
    std::uint32_t request_count, blackframe::xpu::shared::ConstantTextureEvaluationResult* results,
    void* stream = nullptr) noexcept;
