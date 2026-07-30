#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

// Limits are maximum accepted scattering-event counts. A zero limit disables that category
// explicitly. Transmission is orthogonal to the diffuse/glossy/specular family and is therefore
// not part of the total path depth.
struct PathDepthLimits final {
    std::uint32_t diffuse{};
    std::uint32_t glossy{};
    std::uint32_t specular{};
    std::uint32_t transmission{};
    std::uint32_t volume{};

    [[nodiscard]] constexpr bool operator==(const PathDepthLimits&) const noexcept = default;
};

struct PathDepthCounters final {
    std::uint32_t diffuse{};
    std::uint32_t glossy{};
    std::uint32_t specular{};
    std::uint32_t transmission{};
    std::uint32_t volume{};

    [[nodiscard]] constexpr bool operator==(const PathDepthCounters&) const noexcept = default;
};

// accepted() is false only for a valid event blocked by one or more exact category limits.
// blocked_limits contains only diffuse, glossy, specular, transmission, and volume bits.
struct PathDepthEventResult final {
    PathDepthCounters counters{};
    ScatteringLobe blocked_limits{ScatteringLobe::none};

    [[nodiscard]] constexpr bool accepted() const noexcept {
        return blocked_limits == ScatteringLobe::none;
    }
};

[[nodiscard]] core::Result<std::uint32_t> path_depth_total(const PathDepthCounters& counters);

[[nodiscard]] core::Status validate_path_depth_counters(const PathDepthCounters& counters);

// The expected total is PathState::depth(). Surface transmission contributes to its scattering
// family and transmission counter, but only the family contributes to the total.
[[nodiscard]] core::Status validate_path_depth_state(const PathDepthLimits& limits,
                                                     const PathDepthCounters& counters,
                                                     std::uint32_t expected_total);

// A surface event has exactly one family (diffuse/glossy/specular) and one direction
// (reflection/transmission). A volume event is exactly ScatteringLobe::volume. Invalid or
// unrepresentable transitions are errors; a reached limit returns the unchanged counters and every
// blocking category bit.
[[nodiscard]] core::Result<PathDepthEventResult>
evaluate_path_depth_event(const PathDepthLimits& limits, const PathDepthCounters& counters,
                          ScatteringLobe event_lobes);

static_assert(std::is_standard_layout_v<PathDepthLimits>);
static_assert(std::is_trivially_copyable_v<PathDepthLimits>);
static_assert(std::is_standard_layout_v<PathDepthCounters>);
static_assert(std::is_trivially_copyable_v<PathDepthCounters>);
static_assert(std::is_standard_layout_v<PathDepthEventResult>);
static_assert(std::is_trivially_copyable_v<PathDepthEventResult>);

} // namespace blackframe::renderer
