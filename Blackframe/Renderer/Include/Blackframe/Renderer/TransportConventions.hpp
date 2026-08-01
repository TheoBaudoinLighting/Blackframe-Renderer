#pragma once

#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

// These are data conventions shared by every transport implementation. They deliberately contain
// no CPU- or GPU-specific annotation and are never serialized by copying their native layout.
enum class ProbabilityMeasure : std::uint8_t {
    discrete = 0,
    solid_angle = 1,
    area = 2,
    distance = 3,
    volume = 4,
    wavelength = 5,
};

[[nodiscard]] constexpr bool
is_known_probability_measure(const ProbabilityMeasure measure) noexcept {
    switch (measure) {
    case ProbabilityMeasure::discrete:
    case ProbabilityMeasure::solid_angle:
    case ProbabilityMeasure::area:
    case ProbabilityMeasure::distance:
    case ProbabilityMeasure::volume:
    case ProbabilityMeasure::wavelength:
        return true;
    }
    return false;
}

struct ProbabilityDensity {
    TransportScalar value{};
    ProbabilityMeasure measure{ProbabilityMeasure::discrete};
};

struct ReferenceProbabilityDensity {
    ReferenceScalar value{};
    ProbabilityMeasure measure{ProbabilityMeasure::discrete};
};

enum class ScatteringLobe : std::uint32_t {
    none = 0x00000000U,
    diffuse = 0x00000001U,
    glossy = 0x00000002U,
    specular = 0x00000004U,
    reflection = 0x00000008U,
    transmission = 0x00000010U,
    volume = 0x00000020U,
};

[[nodiscard]] constexpr ScatteringLobe operator|(const ScatteringLobe left,
                                                 const ScatteringLobe right) noexcept {
    return static_cast<ScatteringLobe>(static_cast<std::uint32_t>(left) |
                                       static_cast<std::uint32_t>(right));
}

[[nodiscard]] constexpr ScatteringLobe operator&(const ScatteringLobe left,
                                                 const ScatteringLobe right) noexcept {
    return static_cast<ScatteringLobe>(static_cast<std::uint32_t>(left) &
                                       static_cast<std::uint32_t>(right));
}

inline constexpr auto ScatteringFamilyMask =
    ScatteringLobe::diffuse | ScatteringLobe::glossy | ScatteringLobe::specular;
inline constexpr auto ScatteringDirectionMask =
    ScatteringLobe::reflection | ScatteringLobe::transmission;
inline constexpr auto KnownScatteringLobeMask =
    ScatteringFamilyMask | ScatteringDirectionMask | ScatteringLobe::volume;
inline constexpr auto ContinuousBsdfProbabilityMeasure = ProbabilityMeasure::solid_angle;
inline constexpr auto DeltaBsdfProbabilityMeasure = ProbabilityMeasure::discrete;

[[nodiscard]] constexpr std::uint32_t scattering_lobe_bits(const ScatteringLobe lobes) noexcept {
    return static_cast<std::uint32_t>(lobes);
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

[[nodiscard]] constexpr bool is_known_scattering_lobe_mask(const ScatteringLobe lobes) noexcept {
    return (scattering_lobe_bits(lobes) & ~scattering_lobe_bits(KnownScatteringLobeMask)) == 0U;
}

[[nodiscard]] constexpr bool
has_exactly_one_scattering_lobe_bit(const ScatteringLobe lobes) noexcept {
    const auto bits = scattering_lobe_bits(lobes);
    return bits != 0U && (bits & (bits - 1U)) == 0U;
}

// A concrete surface event contains exactly one scattering family and exactly one direction.
// Selection masks may combine any known bits and should use is_known_scattering_lobe_mask instead.
[[nodiscard]] constexpr bool
is_valid_surface_scattering_event(const ScatteringLobe lobes) noexcept {
    if (!is_known_scattering_lobe_mask(lobes) ||
        has_scattering_lobe(lobes, ScatteringLobe::volume)) {
        return false;
    }
    return has_exactly_one_scattering_lobe_bit(lobes & ScatteringFamilyMask) &&
           has_exactly_one_scattering_lobe_bit(lobes & ScatteringDirectionMask);
}

[[nodiscard]] constexpr bool is_valid_scattering_event(const ScatteringLobe lobes) noexcept {
    return lobes == ScatteringLobe::volume || is_valid_surface_scattering_event(lobes);
}

// Specular is the surface delta family. Diffuse and glossy surface events are continuous. Volume
// events are outside the BSDF domain and are not classified by these two helpers.
[[nodiscard]] constexpr bool
is_delta_surface_scattering_event(const ScatteringLobe lobes) noexcept {
    return is_valid_surface_scattering_event(lobes) &&
           has_scattering_lobe(lobes, ScatteringLobe::specular);
}

[[nodiscard]] constexpr bool
is_continuous_surface_scattering_event(const ScatteringLobe lobes) noexcept {
    return is_valid_surface_scattering_event(lobes) &&
           !has_scattering_lobe(lobes, ScatteringLobe::specular);
}

// Radiance is the camera-to-light mode. A non-symmetric transmission closure applies the adjoint
// factor (eta_i / eta_t)^2 in radiance mode, where eta_i is on wo's side and eta_t is on wi's
// side. Importance mode applies no adjoint eta factor. Reflection is mode-invariant.
enum class TransportMode : std::uint8_t {
    radiance = 0,
    importance = 1,
};

[[nodiscard]] constexpr bool is_known_transport_mode(const TransportMode mode) noexcept {
    switch (mode) {
    case TransportMode::radiance:
    case TransportMode::importance:
        return true;
    }
    return false;
}

// BSDF wo and wi are unit directions in a caller-supplied local closure frame whose normal is +Z.
// Both point away from the surface: wo goes toward the previous path vertex and wi toward the next
// vertex. Same non-zero Z signs are reflection, opposite non-zero signs are transmission, and an
// exact tangent direction has zero BSDF support. BSDF values exclude cosine and probability terms.
// pdf(wo, wi) is the complete conditional p(wi | wo); a reverse PDF swaps the arguments and is not
// assumed equal. Continuous directional PDFs use solid angle. Ordinary directional queries exclude
// Dirac support, while a sampled delta event returns a positive discrete probability. Component
// selection is included exactly once. Throughput uses f * abs(wi.z) / p in the event's measure.

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

static_assert(sizeof(ProbabilityMeasure) == sizeof(std::uint8_t));
static_assert(sizeof(ScatteringLobe) == sizeof(std::uint32_t));
static_assert(sizeof(TransportMode) == sizeof(std::uint8_t));
static_assert(std::is_standard_layout_v<ProbabilityDensity>);
static_assert(std::is_trivially_copyable_v<ProbabilityDensity>);
static_assert(std::is_standard_layout_v<ReferenceProbabilityDensity>);
static_assert(std::is_trivially_copyable_v<ReferenceProbabilityDensity>);

} // namespace blackframe::renderer
