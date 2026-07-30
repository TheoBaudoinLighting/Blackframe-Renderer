#pragma once

#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <cstdint>

namespace blackframe::renderer {

// These are data conventions shared by every transport implementation. They deliberately contain
// no CPU- or GPU-specific annotation and are never serialized by copying their native layout.
enum class ProbabilityMeasure : std::uint8_t {
    discrete,
    solid_angle,
    area,
    distance,
    volume,
    wavelength,
};

struct ProbabilityDensity {
    TransportScalar value{};
    ProbabilityMeasure measure{ProbabilityMeasure::discrete};
};

struct ReferenceProbabilityDensity {
    ReferenceScalar value{};
    ProbabilityMeasure measure{ProbabilityMeasure::discrete};
};

enum class ScatteringLobe : std::uint32_t {
    none = 0,
    diffuse = 1U << 0U,
    glossy = 1U << 1U,
    specular = 1U << 2U,
    reflection = 1U << 3U,
    transmission = 1U << 4U,
    volume = 1U << 5U,
};

[[nodiscard]] constexpr ScatteringLobe operator|(const ScatteringLobe left,
                                                 const ScatteringLobe right) noexcept {
    return static_cast<ScatteringLobe>(static_cast<std::uint32_t>(left) |
                                       static_cast<std::uint32_t>(right));
}

[[nodiscard]] constexpr bool has_scattering_lobe(const ScatteringLobe lobes,
                                                 const ScatteringLobe lobe) noexcept {
    const auto bits = static_cast<std::uint32_t>(lobes);
    const auto requested = static_cast<std::uint32_t>(lobe);
    if (requested == 0) {
        return bits == 0;
    }
    return (bits & requested) == requested;
}

enum class TransportEvent : std::uint8_t {
    none,
    surface_scattering,
    medium_scattering,
    emission,
    absorption,
    escaped,
};

struct WavelengthSample {
    TransportScalar nanometers{};
    ProbabilityDensity probability{
        .value = 0.0F,
        .measure = ProbabilityMeasure::wavelength,
    };
};

struct ReferenceWavelengthSample {
    ReferenceScalar nanometers{};
    ReferenceProbabilityDensity probability{
        .value = 0.0,
        .measure = ProbabilityMeasure::wavelength,
    };
};

} // namespace blackframe::renderer
