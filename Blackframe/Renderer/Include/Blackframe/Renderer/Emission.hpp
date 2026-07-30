#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace emission_detail {

[[nodiscard]] inline core::Error invalid_emission(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool
valid_radiance(const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& radiance) noexcept {
    for (const auto value : radiance.values) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool unit_normal(const Normal3T<Scalar> normal) noexcept {
    const auto squared_length =
        std::fma(normal.x, normal.x, std::fma(normal.y, normal.y, normal.z * normal.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(normal.x) && std::isfinite(normal.y) && std::isfinite(normal.z) &&
           std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool valid_direction(const Vector3T<Scalar> direction) noexcept {
    return std::isfinite(direction.x) && std::isfinite(direction.y) && std::isfinite(direction.z) &&
           (direction.x != Scalar{0} || direction.y != Scalar{0} || direction.z != Scalar{0});
}

} // namespace emission_detail

// A one-sided surface emitter. The supplied radiance is already evaluated at the current four
// wavelengths. The geometric normal fixes the emitting side and is never face-forwarded.
template <SpectrumScalar Scalar> class OneSidedSurfaceEmissionT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

    [[nodiscard]] static core::Result<OneSidedSurfaceEmissionT>
    create(const spectrum_type radiance) {
        if (!emission_detail::valid_radiance(radiance)) {
            return std::unexpected(emission_detail::invalid_emission(
                "One-sided surface emission requires every spectral lane to be finite and "
                "non-negative."));
        }
        return OneSidedSurfaceEmissionT{radiance};
    }

    [[nodiscard]] constexpr const spectrum_type& radiance() const noexcept {
        return radiance_;
    }

    // The outgoing direction points away from the surface. A valid back-facing or tangent query
    // returns black because it lies outside this emitter's support.
    [[nodiscard]] core::Result<spectrum_type>
    eval(const Normal3T<Scalar> geometric_normal, const Vector3T<Scalar> outgoing_direction) const {
        if (!emission_detail::unit_normal(geometric_normal) ||
            !emission_detail::valid_direction(outgoing_direction)) {
            return std::unexpected(emission_detail::invalid_emission(
                "Surface emission evaluation requires a finite unit geometric normal and a "
                "finite non-zero outgoing direction."));
        }

        const auto maximum_component =
            std::max({std::abs(outgoing_direction.x), std::abs(outgoing_direction.y),
                      std::abs(outgoing_direction.z)});
        const auto scaled_outgoing = outgoing_direction / maximum_component;
        const auto alignment = std::fma(geometric_normal.x, scaled_outgoing.x,
                                        std::fma(geometric_normal.y, scaled_outgoing.y,
                                                 geometric_normal.z * scaled_outgoing.z));
        if (!std::isfinite(alignment)) {
            return std::unexpected(emission_detail::invalid_emission(
                "Surface emission orientation is not representable."));
        }

        const auto componentwise_orthogonal =
            (geometric_normal.x == Scalar{0} || outgoing_direction.x == Scalar{0}) &&
            (geometric_normal.y == Scalar{0} || outgoing_direction.y == Scalar{0}) &&
            (geometric_normal.z == Scalar{0} || outgoing_direction.z == Scalar{0});
        if (alignment == Scalar{0} && componentwise_orthogonal) {
            return spectrum_type{};
        }

        const auto absolute_sum =
            std::fma(std::abs(geometric_normal.x), std::abs(scaled_outgoing.x),
                     std::fma(std::abs(geometric_normal.y), std::abs(scaled_outgoing.y),
                              std::abs(geometric_normal.z) * std::abs(scaled_outgoing.z)));
        // Three rounded operations bound this FMA dot by less than four epsilons of its absolute
        // term sum when no underflow occurs. The denormal allowance covers products too small to
        // contribute to that sum. An ambiguous sign is an error, never an implicit back face.
        constexpr auto rounding_factor = Scalar{4} * std::numeric_limits<Scalar>::epsilon();
        constexpr auto underflow_allowance = Scalar{4} * std::numeric_limits<Scalar>::denorm_min();
        const auto orientation_uncertainty =
            std::fma(rounding_factor, absolute_sum, underflow_allowance);
        if (std::abs(alignment) <= orientation_uncertainty) {
            return std::unexpected(emission_detail::invalid_emission(
                "Surface emission orientation is not representable."));
        }
        if (!(alignment > Scalar{0})) {
            return spectrum_type{};
        }
        return radiance_;
    }

  private:
    constexpr explicit OneSidedSurfaceEmissionT(const spectrum_type radiance) noexcept
        : radiance_{radiance} {}

    spectrum_type radiance_;
};

// A direction-independent environment. The direction is still validated so an invalid escaped ray
// cannot be hidden by the constant lookup.
template <SpectrumScalar Scalar> class ConstantEnvironmentT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

    [[nodiscard]] static core::Result<ConstantEnvironmentT> create(const spectrum_type radiance) {
        if (!emission_detail::valid_radiance(radiance)) {
            return std::unexpected(emission_detail::invalid_emission(
                "Constant environment radiance requires every spectral lane to be finite and "
                "non-negative."));
        }
        return ConstantEnvironmentT{radiance};
    }

    [[nodiscard]] constexpr const spectrum_type& radiance() const noexcept {
        return radiance_;
    }

    [[nodiscard]] core::Result<spectrum_type> eval(const Vector3T<Scalar> escaped_direction) const {
        if (!emission_detail::valid_direction(escaped_direction)) {
            return std::unexpected(emission_detail::invalid_emission(
                "Constant environment evaluation requires a finite non-zero direction."));
        }
        return radiance_;
    }

  private:
    constexpr explicit ConstantEnvironmentT(const spectrum_type radiance) noexcept
        : radiance_{radiance} {}

    spectrum_type radiance_;
};

using OneSidedSurfaceEmission = OneSidedSurfaceEmissionT<TransportScalar>;
using ReferenceOneSidedSurfaceEmission = OneSidedSurfaceEmissionT<ReferenceScalar>;
using ConstantEnvironment = ConstantEnvironmentT<TransportScalar>;
using ReferenceConstantEnvironment = ConstantEnvironmentT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<OneSidedSurfaceEmission>);
static_assert(std::is_trivially_copyable_v<OneSidedSurfaceEmission>);
static_assert(std::is_standard_layout_v<ReferenceOneSidedSurfaceEmission>);
static_assert(std::is_trivially_copyable_v<ReferenceOneSidedSurfaceEmission>);
static_assert(std::is_standard_layout_v<ConstantEnvironment>);
static_assert(std::is_trivially_copyable_v<ConstantEnvironment>);
static_assert(std::is_standard_layout_v<ReferenceConstantEnvironment>);
static_assert(std::is_trivially_copyable_v<ReferenceConstantEnvironment>);

} // namespace blackframe::renderer
