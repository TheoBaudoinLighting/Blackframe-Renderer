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

// One lane of deterministic CUDA traversal output. A miss or error has a
// zero-initialized hit payload; status is therefore always authoritative.
struct alignas(16) SceneClosestHitResult final {
    std::uint32_t status{};
    std::array<std::uint32_t, 3> reserved{};
    ClosestHit hit{};
};

static_assert(sizeof(SceneClosestHitStatus) == sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<SceneClosestHitResult>);
static_assert(std::is_trivially_copyable_v<SceneClosestHitResult>);
static_assert(std::is_trivially_destructible_v<SceneClosestHitResult>);
static_assert(sizeof(SceneClosestHitResult) == 80U);
static_assert(alignof(SceneClosestHitResult) == 16U);
static_assert(offsetof(SceneClosestHitResult, status) == 0U);
static_assert(offsetof(SceneClosestHitResult, reserved) == 4U);
static_assert(offsetof(SceneClosestHitResult, hit) == 16U);

} // namespace blackframe::xpu::shared
