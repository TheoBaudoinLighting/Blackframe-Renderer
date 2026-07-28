#pragma once

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
    float value{};
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

enum class TransportEvent : std::uint8_t {
    none,
    surface_scattering,
    medium_scattering,
    emission,
    absorption,
    escaped,
};

struct WavelengthSample {
    float nanometers{};
    ProbabilityDensity probability{
        .value = 0.0F,
        .measure = ProbabilityMeasure::wavelength,
    };
};

} // namespace blackframe::renderer
