#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <limits>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Error depth_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

[[nodiscard]] constexpr ScatteringLobe combine_lobes(const ScatteringLobe left,
                                                     const ScatteringLobe right) noexcept {
    return left | right;
}

[[nodiscard]] core::Status validate_known_event(const ScatteringLobe event_lobes) {
    constexpr auto known_lobes = ScatteringLobe::diffuse | ScatteringLobe::glossy |
                                 ScatteringLobe::specular | ScatteringLobe::reflection |
                                 ScatteringLobe::transmission | ScatteringLobe::volume;
    const auto bits = static_cast<std::uint32_t>(event_lobes);
    const auto known_bits = static_cast<std::uint32_t>(known_lobes);
    if (bits == 0 || (bits & ~known_bits) != 0) {
        return std::unexpected(
            depth_error(core::StatusCode::invalid_argument,
                        "A path-depth event must contain only supported scattering-lobe bits."));
    }

    const auto is_volume = has_scattering_lobe(event_lobes, ScatteringLobe::volume);
    if (is_volume) {
        if (event_lobes != ScatteringLobe::volume) {
            return std::unexpected(depth_error(
                core::StatusCode::invalid_argument,
                "A volume path-depth event cannot contain a surface family or direction."));
        }
        return {};
    }

    const auto family_count =
        static_cast<unsigned>(has_scattering_lobe(event_lobes, ScatteringLobe::diffuse)) +
        static_cast<unsigned>(has_scattering_lobe(event_lobes, ScatteringLobe::glossy)) +
        static_cast<unsigned>(has_scattering_lobe(event_lobes, ScatteringLobe::specular));
    const auto direction_count =
        static_cast<unsigned>(has_scattering_lobe(event_lobes, ScatteringLobe::reflection)) +
        static_cast<unsigned>(has_scattering_lobe(event_lobes, ScatteringLobe::transmission));
    if (family_count != 1 || direction_count != 1) {
        return std::unexpected(depth_error(
            core::StatusCode::invalid_argument,
            "A surface path-depth event requires exactly one family and one direction."));
    }
    return {};
}

[[nodiscard]] ScatteringLobe blocked_limits(const PathDepthLimits& limits,
                                            const PathDepthCounters& counters,
                                            const ScatteringLobe event_lobes) noexcept {
    auto blocked = ScatteringLobe::none;
    if (has_scattering_lobe(event_lobes, ScatteringLobe::diffuse) &&
        counters.diffuse >= limits.diffuse) {
        blocked = combine_lobes(blocked, ScatteringLobe::diffuse);
    }
    if (has_scattering_lobe(event_lobes, ScatteringLobe::glossy) &&
        counters.glossy >= limits.glossy) {
        blocked = combine_lobes(blocked, ScatteringLobe::glossy);
    }
    if (has_scattering_lobe(event_lobes, ScatteringLobe::specular) &&
        counters.specular >= limits.specular) {
        blocked = combine_lobes(blocked, ScatteringLobe::specular);
    }
    if (has_scattering_lobe(event_lobes, ScatteringLobe::transmission) &&
        counters.transmission >= limits.transmission) {
        blocked = combine_lobes(blocked, ScatteringLobe::transmission);
    }
    if (has_scattering_lobe(event_lobes, ScatteringLobe::volume) &&
        counters.volume >= limits.volume) {
        blocked = combine_lobes(blocked, ScatteringLobe::volume);
    }
    return blocked;
}

[[nodiscard]] PathDepthCounters advanced_counters(const PathDepthCounters& counters,
                                                  const ScatteringLobe event_lobes) noexcept {
    auto advanced = counters;
    if (has_scattering_lobe(event_lobes, ScatteringLobe::diffuse)) {
        ++advanced.diffuse;
    }
    if (has_scattering_lobe(event_lobes, ScatteringLobe::glossy)) {
        ++advanced.glossy;
    }
    if (has_scattering_lobe(event_lobes, ScatteringLobe::specular)) {
        ++advanced.specular;
    }
    if (has_scattering_lobe(event_lobes, ScatteringLobe::transmission)) {
        ++advanced.transmission;
    }
    if (has_scattering_lobe(event_lobes, ScatteringLobe::volume)) {
        ++advanced.volume;
    }
    return advanced;
}

} // namespace

core::Result<std::uint32_t> path_depth_total(const PathDepthCounters& counters) {
    const auto total =
        static_cast<std::uint64_t>(counters.diffuse) + static_cast<std::uint64_t>(counters.glossy) +
        static_cast<std::uint64_t>(counters.specular) + static_cast<std::uint64_t>(counters.volume);
    if (total > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            depth_error(core::StatusCode::resource_exhausted,
                        "Path depth counters exceed the representable total path depth."));
    }
    return static_cast<std::uint32_t>(total);
}

core::Status validate_path_depth_counters(const PathDepthCounters& counters) {
    const auto total = path_depth_total(counters);
    if (!total.has_value()) {
        return std::unexpected(total.error());
    }

    const auto surface_total = static_cast<std::uint64_t>(counters.diffuse) +
                               static_cast<std::uint64_t>(counters.glossy) +
                               static_cast<std::uint64_t>(counters.specular);
    if (static_cast<std::uint64_t>(counters.transmission) > surface_total) {
        return std::unexpected(depth_error(
            core::StatusCode::invalid_argument,
            "Path transmission depth cannot exceed accepted surface scattering depth."));
    }
    return {};
}

core::Status validate_path_depth_state(const PathDepthLimits& limits,
                                       const PathDepthCounters& counters,
                                       const std::uint32_t expected_total) {
    const auto counters_status = validate_path_depth_counters(counters);
    if (!counters_status.has_value()) {
        return counters_status;
    }

    const auto total = path_depth_total(counters);
    if (!total.has_value()) {
        return std::unexpected(total.error());
    }
    if (*total != expected_total) {
        return std::unexpected(
            depth_error(core::StatusCode::invalid_argument,
                        "Path depth counters do not match the path state's total depth."));
    }
    if (counters.diffuse > limits.diffuse || counters.glossy > limits.glossy ||
        counters.specular > limits.specular || counters.transmission > limits.transmission ||
        counters.volume > limits.volume) {
        return std::unexpected(
            depth_error(core::StatusCode::invalid_argument,
                        "Path depth counters cannot exceed their configured category limits."));
    }
    return {};
}

core::Result<PathDepthEventResult> evaluate_path_depth_event(const PathDepthLimits& limits,
                                                             const PathDepthCounters& counters,
                                                             const ScatteringLobe event_lobes) {
    const auto event_status = validate_known_event(event_lobes);
    if (!event_status.has_value()) {
        return std::unexpected(event_status.error());
    }

    const auto total = path_depth_total(counters);
    if (!total.has_value()) {
        return std::unexpected(total.error());
    }
    const auto state_status = validate_path_depth_state(limits, counters, *total);
    if (!state_status.has_value()) {
        return std::unexpected(state_status.error());
    }

    const auto blocked = blocked_limits(limits, counters, event_lobes);
    if (blocked != ScatteringLobe::none) {
        return PathDepthEventResult{
            .counters = counters,
            .blocked_limits = blocked,
        };
    }
    if (*total == std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            depth_error(core::StatusCode::resource_exhausted,
                        "A path-depth event would overflow the total path depth."));
    }

    const auto advanced = advanced_counters(counters, event_lobes);
    const auto advanced_status = validate_path_depth_state(limits, advanced, *total + 1U);
    if (!advanced_status.has_value()) {
        return std::unexpected(advanced_status.error());
    }
    return PathDepthEventResult{
        .counters = advanced,
        .blocked_limits = ScatteringLobe::none,
    };
}

} // namespace blackframe::renderer
