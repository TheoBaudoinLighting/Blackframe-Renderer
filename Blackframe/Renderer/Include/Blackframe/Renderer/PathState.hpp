#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

// The first flag describes the most recently accepted surface or medium scattering event. The
// second records whether any accepted event in the path history was non-delta. A primary path has
// neither flag because it has no completed scattering event.
enum class PathDeltaFlags : std::uint8_t {
    none = 0,
    previous_bounce_was_delta = 1U << 0U,
    any_non_delta_bounces = 1U << 1U,
};

[[nodiscard]] constexpr PathDeltaFlags operator|(const PathDeltaFlags left,
                                                 const PathDeltaFlags right) noexcept {
    return static_cast<PathDeltaFlags>(static_cast<std::uint8_t>(left) |
                                       static_cast<std::uint8_t>(right));
}

[[nodiscard]] constexpr bool has_path_delta_flag(const PathDeltaFlags flags,
                                                 const PathDeltaFlags flag) noexcept {
    const auto bits = static_cast<std::uint8_t>(flags);
    const auto requested = static_cast<std::uint8_t>(flag);
    if (requested == 0) {
        return bits == 0;
    }
    return (bits & requested) == requested;
}

namespace path_state_detail {

[[nodiscard]] inline core::Error invalid_path_state(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool
finite_spectrum(const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr bool known_delta_flags(const PathDeltaFlags flags) noexcept {
    constexpr auto known_bits =
        static_cast<std::uint8_t>(PathDeltaFlags::previous_bounce_was_delta) |
        static_cast<std::uint8_t>(PathDeltaFlags::any_non_delta_bounces);
    const auto bits = static_cast<std::uint8_t>(flags);
    return (bits & static_cast<std::uint8_t>(~known_bits)) == 0;
}

} // namespace path_state_detail

// Path depth counts accepted surface or medium scattering events and is derived from the bound
// category counters. beta is the spectral path throughput and accumulated_radiance is L; neither
// contains wavelength-PDF compensation. eta_scale is the positive transmission compensation factor
// reserved for Russian roulette, not a current index of refraction. The wavelength packet and
// medium identity are owned by value.
template <SpectrumScalar Scalar> class PathStateT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using wavelengths_type = SampledWavelengthsT<Scalar>;

    [[nodiscard]] static core::Result<PathStateT>
    create(const spectrum_type beta, const spectrum_type accumulated_radiance,
           const PathDepthCounters depth_counters, const Scalar eta_scale,
           const wavelengths_type wavelengths, const PathDeltaFlags delta_flags,
           const MediumId current_medium) {
        if (!path_state_detail::finite_spectrum(beta)) {
            return std::unexpected(path_state_detail::invalid_path_state(
                "Path beta requires every spectral lane to be finite."));
        }
        if (!path_state_detail::finite_spectrum(accumulated_radiance)) {
            return std::unexpected(path_state_detail::invalid_path_state(
                "Path accumulated radiance requires every spectral lane to be finite."));
        }
        if (!std::isfinite(eta_scale) || !(eta_scale > Scalar{0})) {
            return std::unexpected(path_state_detail::invalid_path_state(
                "Path eta scale must be finite and strictly positive."));
        }

        const auto counters_status = validate_path_depth_counters(depth_counters);
        if (!counters_status.has_value()) {
            return std::unexpected(counters_status.error());
        }
        const auto depth = path_depth_total(depth_counters);
        if (!depth.has_value()) {
            return std::unexpected(depth.error());
        }

        for (const auto& wavelength : wavelengths.samples) {
            if (!std::isfinite(wavelength.nanometers) ||
                wavelength.nanometers < static_cast<Scalar>(VisibleWavelengthMinimumNanometers) ||
                wavelength.nanometers > static_cast<Scalar>(VisibleWavelengthMaximumNanometers)) {
                return std::unexpected(path_state_detail::invalid_path_state(
                    "Path wavelengths must be finite values in [360, 830] nanometers."));
            }
            if (!std::isfinite(wavelength.probability.value) ||
                !(wavelength.probability.value > Scalar{0}) ||
                wavelength.probability.measure != ProbabilityMeasure::wavelength) {
                return std::unexpected(path_state_detail::invalid_path_state(
                    "Path wavelength PDFs must be finite, strictly positive, and use wavelength "
                    "measure."));
            }
        }

        if (!path_state_detail::known_delta_flags(delta_flags)) {
            return std::unexpected(path_state_detail::invalid_path_state(
                "Path delta flags contain unsupported bits."));
        }
        if (*depth == 0 && delta_flags != PathDeltaFlags::none) {
            return std::unexpected(path_state_detail::invalid_path_state(
                "A primary path cannot report completed-bounce delta flags."));
        }
        if (*depth == 1 &&
            has_path_delta_flag(delta_flags, PathDeltaFlags::previous_bounce_was_delta) &&
            has_path_delta_flag(delta_flags, PathDeltaFlags::any_non_delta_bounces)) {
            return std::unexpected(path_state_detail::invalid_path_state(
                "A one-bounce path cannot combine a delta previous bounce with non-delta "
                "history."));
        }
        if (*depth != 0 &&
            !has_path_delta_flag(delta_flags, PathDeltaFlags::previous_bounce_was_delta) &&
            !has_path_delta_flag(delta_flags, PathDeltaFlags::any_non_delta_bounces)) {
            return std::unexpected(path_state_detail::invalid_path_state(
                "A non-delta previous bounce must be recorded in the path history."));
        }
        const auto non_delta_depth = static_cast<std::uint64_t>(depth_counters.diffuse) +
                                     static_cast<std::uint64_t>(depth_counters.glossy) +
                                     static_cast<std::uint64_t>(depth_counters.volume);
        const auto previous_was_delta =
            has_path_delta_flag(delta_flags, PathDeltaFlags::previous_bounce_was_delta);
        const auto has_non_delta_history =
            has_path_delta_flag(delta_flags, PathDeltaFlags::any_non_delta_bounces);
        if ((non_delta_depth != 0) != has_non_delta_history ||
            (previous_was_delta && depth_counters.specular == 0) ||
            (*depth == 1 && previous_was_delta && depth_counters.specular != 1) ||
            (*depth == 1 && !previous_was_delta && depth_counters.specular != 0)) {
            return std::unexpected(path_state_detail::invalid_path_state(
                "Path depth counters are inconsistent with the delta history."));
        }

        return PathStateT{beta,      accumulated_radiance, depth_counters, *depth,
                          eta_scale, wavelengths,          delta_flags,    current_medium};
    }

    [[nodiscard]] static core::Result<PathStateT> create_initial(const wavelengths_type wavelengths,
                                                                 const MediumId current_medium) {
        auto beta = spectrum_type{};
        beta.values.fill(Scalar{1});
        return create(beta, spectrum_type{}, PathDepthCounters{}, Scalar{1}, wavelengths,
                      PathDeltaFlags::none, current_medium);
    }

    [[nodiscard]] constexpr const spectrum_type& beta() const noexcept {
        return beta_;
    }

    [[nodiscard]] constexpr const spectrum_type& accumulated_radiance() const noexcept {
        return accumulated_radiance_;
    }

    [[nodiscard]] constexpr std::uint32_t depth() const noexcept {
        return depth_;
    }

    [[nodiscard]] constexpr const PathDepthCounters& depth_counters() const noexcept {
        return depth_counters_;
    }

    [[nodiscard]] constexpr Scalar eta_scale() const noexcept {
        return eta_scale_;
    }

    [[nodiscard]] constexpr const wavelengths_type& wavelengths() const noexcept {
        return wavelengths_;
    }

    [[nodiscard]] constexpr PathDeltaFlags delta_flags() const noexcept {
        return delta_flags_;
    }

    [[nodiscard]] constexpr MediumId current_medium() const noexcept {
        return current_medium_;
    }

  private:
    constexpr PathStateT(const spectrum_type beta, const spectrum_type accumulated_radiance,
                         const PathDepthCounters depth_counters, const std::uint32_t depth,
                         const Scalar eta_scale, const wavelengths_type wavelengths,
                         const PathDeltaFlags delta_flags, const MediumId current_medium) noexcept
        : beta_{beta}, accumulated_radiance_{accumulated_radiance}, depth_counters_{depth_counters},
          depth_{depth}, eta_scale_{eta_scale}, wavelengths_{wavelengths},
          delta_flags_{delta_flags}, current_medium_{current_medium} {}

    spectrum_type beta_;
    spectrum_type accumulated_radiance_;
    PathDepthCounters depth_counters_;
    std::uint32_t depth_;
    Scalar eta_scale_;
    wavelengths_type wavelengths_;
    PathDeltaFlags delta_flags_;
    MediumId current_medium_;
};

using PathState = PathStateT<TransportScalar>;
using ReferencePathState = PathStateT<ReferenceScalar>;

static_assert(sizeof(PathDeltaFlags) == sizeof(std::uint8_t));
static_assert(!std::is_default_constructible_v<PathState>);
static_assert(!std::is_default_constructible_v<ReferencePathState>);
static_assert(std::is_standard_layout_v<PathState>);
static_assert(std::is_trivially_copyable_v<PathState>);
static_assert(std::is_standard_layout_v<ReferencePathState>);
static_assert(std::is_trivially_copyable_v<ReferencePathState>);

} // namespace blackframe::renderer
