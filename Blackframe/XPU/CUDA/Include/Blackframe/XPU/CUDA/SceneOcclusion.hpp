#pragma once

#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cstddef>
#include <cstdint>

// Every pointer names storage on the active CUDA device. Input and output
// ranges must not overlap. The optional opaque handle must be a cudaStream_t
// created on that device; nullptr preserves default-stream ordering. The
// launch is asynchronous, so runtime launch status, synchronization, and
// per-lane traversal status remain explicit.
extern "C" int blackframe_cuda_launch_scene_occlusion(
    const std::uint8_t* scene_bytes, std::size_t scene_size, const std::uint8_t* bvh_bytes,
    std::size_t bvh_size, const blackframe::xpu::shared::TransportRay* rays,
    std::uint32_t ray_count, blackframe::xpu::shared::SceneOcclusionResult* results,
    void* stream = nullptr) noexcept;
