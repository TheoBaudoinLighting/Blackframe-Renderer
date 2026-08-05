#pragma once

#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::shared {

enum class SceneClosestHitStatus : std::uint32_t {
    miss = 0U,
    hit = 1U,
    invalid_ray = 2U,
    invalid_scene = 3U,
    invalid_bvh = 4U,
    stack_overflow = 5U,
    invalid_topology = 6U,
    numerical_failure = 7U,
};

using SceneTraversalStatus = SceneClosestHitStatus;

enum class SceneOcclusionStatus : std::uint32_t {
    visible = 0U,
    occluded = 1U,
    invalid_ray = 2U,
    invalid_scene = 3U,
    invalid_bvh = 4U,
    stack_overflow = 5U,
    invalid_topology = 6U,
    numerical_failure = 7U,
};

// One lane of deterministic CUDA traversal output. A miss or error has a
// zero-initialized hit payload; status is therefore always authoritative.
struct alignas(16) SceneClosestHitResult final {
    std::uint32_t status{};
    std::array<std::uint32_t, 3> reserved{};
    ClosestHit hit{};
};

// One lane of deterministic CUDA any-hit output. status is authoritative and
// the tail remains canonical zero for visible, occluded, and error lanes.
struct alignas(16) SceneOcclusionResult final {
    std::uint32_t status{};
    std::array<std::uint32_t, 3> reserved{};
};

static_assert(sizeof(SceneTraversalStatus) == sizeof(std::uint32_t));
static_assert(sizeof(SceneOcclusionStatus) == sizeof(std::uint32_t));
static_assert(static_cast<std::uint32_t>(SceneOcclusionStatus::invalid_ray) ==
              static_cast<std::uint32_t>(SceneTraversalStatus::invalid_ray));
static_assert(static_cast<std::uint32_t>(SceneOcclusionStatus::invalid_scene) ==
              static_cast<std::uint32_t>(SceneTraversalStatus::invalid_scene));
static_assert(static_cast<std::uint32_t>(SceneOcclusionStatus::invalid_bvh) ==
              static_cast<std::uint32_t>(SceneTraversalStatus::invalid_bvh));
static_assert(static_cast<std::uint32_t>(SceneOcclusionStatus::stack_overflow) ==
              static_cast<std::uint32_t>(SceneTraversalStatus::stack_overflow));
static_assert(static_cast<std::uint32_t>(SceneOcclusionStatus::invalid_topology) ==
              static_cast<std::uint32_t>(SceneTraversalStatus::invalid_topology));
static_assert(static_cast<std::uint32_t>(SceneOcclusionStatus::numerical_failure) ==
              static_cast<std::uint32_t>(SceneTraversalStatus::numerical_failure));
static_assert(std::is_standard_layout_v<SceneClosestHitResult>);
static_assert(std::is_trivially_copyable_v<SceneClosestHitResult>);
static_assert(std::is_trivially_destructible_v<SceneClosestHitResult>);
static_assert(sizeof(SceneClosestHitResult) == 80U);
static_assert(alignof(SceneClosestHitResult) == 16U);
static_assert(offsetof(SceneClosestHitResult, status) == 0U);
static_assert(offsetof(SceneClosestHitResult, reserved) == 4U);
static_assert(offsetof(SceneClosestHitResult, hit) == 16U);
static_assert(std::is_standard_layout_v<SceneOcclusionResult>);
static_assert(std::is_trivially_copyable_v<SceneOcclusionResult>);
static_assert(std::is_trivially_destructible_v<SceneOcclusionResult>);
static_assert(sizeof(SceneOcclusionResult) == 16U);
static_assert(alignof(SceneOcclusionResult) == 16U);
static_assert(offsetof(SceneOcclusionResult, status) == 0U);
static_assert(offsetof(SceneOcclusionResult, reserved) == 4U);

} // namespace blackframe::xpu::shared
