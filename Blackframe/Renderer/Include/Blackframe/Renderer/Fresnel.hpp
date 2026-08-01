#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>

namespace blackframe::renderer {

template <typename Scalar>
concept FresnelScalar =
    std::same_as<Scalar, TransportScalar> || std::same_as<Scalar, ReferenceScalar>;

namespace fresnel_detail {

[[nodiscard]] inline core::Error invalid_fresnel(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <FresnelScalar Scalar>
[[nodiscard]] constexpr bool valid_incident_cosine(const Scalar cosine) noexcept {
    return std::isfinite(cosine) && cosine >= Scalar{0} && cosine <= Scalar{1};
}

template <FresnelScalar Scalar>
[[nodiscard]] constexpr bool valid_refractive_index(const Scalar eta) noexcept {
    return std::isfinite(eta) && eta > Scalar{0};
}

template <FresnelScalar Scalar>
[[nodiscard]] constexpr bool valid_extinction_coefficient(const Scalar k) noexcept {
    return std::isfinite(k) && k >= Scalar{0};
}

template <FresnelScalar Scalar>
[[nodiscard]] core::Result<Scalar>
checked_squared_magnitude_ratio(const Scalar numerator_real, const Scalar numerator_imaginary,
                                const Scalar denominator_real, const Scalar denominator_imaginary) {
    const auto numerator = std::hypot(numerator_real, numerator_imaginary);
    const auto denominator = std::hypot(denominator_real, denominator_imaginary);
    if (!std::isfinite(numerator) || !std::isfinite(denominator) || !(denominator > Scalar{0})) {
        return std::unexpected(
            invalid_fresnel("A conductor Fresnel polarization ratio is not representable."));
    }
    const auto amplitude = numerator / denominator;
    const auto reflectance = amplitude * amplitude;
    if (!std::isfinite(reflectance) || reflectance < Scalar{0} || reflectance > Scalar{1}) {
        return std::unexpected(
            invalid_fresnel("A conductor Fresnel polarization ratio is not representable."));
    }
    return reflectance;
}

template <FresnelScalar Scalar>
[[nodiscard]] core::Result<Scalar> conductor_fresnel_lane(const Scalar incident_cosine,
                                                          const Scalar relative_eta,
                                                          const Scalar relative_k) {
    if (relative_eta == Scalar{1} && relative_k == Scalar{0}) {
        return Scalar{0};
    }
    if (incident_cosine == Scalar{0}) {
        return Scalar{1};
    }

    // Normalizing the incident index 1 and both parts of eta + i*k keeps all complex products
    // bounded even when either supplied coefficient is near the finite limit of Scalar.
    const auto scale = std::max({Scalar{1}, relative_eta, relative_k});
    const auto normalized_incident_eta = Scalar{1} / scale;
    const auto normalized_eta = relative_eta / scale;
    const auto normalized_k = relative_k / scale;

    if (incident_cosine == Scalar{1}) {
        const auto reflectance =
            checked_squared_magnitude_ratio(normalized_eta - normalized_incident_eta, normalized_k,
                                            normalized_eta + normalized_incident_eta, normalized_k);
        if (!reflectance) {
            return std::unexpected(reflectance.error());
        }
        if (!(*reflectance > Scalar{0})) {
            return std::unexpected(
                invalid_fresnel("Conductor Fresnel reflectance is not representable."));
        }
        return *reflectance;
    }

    const auto incident_sine_squared =
        (Scalar{1} - incident_cosine) * (Scalar{1} + incident_cosine);
    const auto scaled_incident_sine_squared =
        normalized_incident_eta * normalized_incident_eta * incident_sine_squared;
    // (x, y) is ((eta + i*k)^2 - sin^2(theta_i)) / scale^2.
    const auto x = std::fma(normalized_eta, normalized_eta,
                            -std::fma(normalized_k, normalized_k, scaled_incident_sine_squared));
    const auto y = Scalar{2} * normalized_eta * normalized_k;
    const auto magnitude = std::hypot(x, y);

    // Principal complex square root without cancellation in magnitude +/- x. With k >= 0 the
    // imaginary component is non-negative and no sign convention is selected implicitly.
    auto root_real = Scalar{};
    auto root_imaginary = Scalar{};
    if (x >= Scalar{0}) {
        root_real = std::sqrt(Scalar{0.5} * (magnitude + x));
        root_imaginary = root_real == Scalar{0} ? Scalar{0} : y / (Scalar{2} * root_real);
    } else {
        root_imaginary = std::sqrt(Scalar{0.5} * (magnitude - x));
        root_real = root_imaginary == Scalar{0} ? Scalar{0} : y / (Scalar{2} * root_imaginary);
    }

    // Perpendicular polarization: (cos(theta_i) - q) / (cos(theta_i) + q), where q is the
    // transmitted complex normal component. Every term is divided by the common scale.
    const auto scaled_incident_cosine = normalized_incident_eta * incident_cosine;
    const auto perpendicular =
        checked_squared_magnitude_ratio(scaled_incident_cosine - root_real, -root_imaginary,
                                        scaled_incident_cosine + root_real, root_imaginary);
    if (!perpendicular) {
        return std::unexpected(perpendicular.error());
    }

    // Parallel polarization after division by scale^2:
    // ((eta + i*k)^2 cos(theta_i) - q) / ((eta + i*k)^2 cos(theta_i) + q).
    const auto squared_eta_real =
        std::fma(normalized_eta, normalized_eta, -(normalized_k * normalized_k));
    const auto squared_eta_imaginary = y;
    const auto scaled_root_real = normalized_incident_eta * root_real;
    const auto scaled_root_imaginary = normalized_incident_eta * root_imaginary;
    const auto parallel_base_real = incident_cosine * squared_eta_real;
    const auto parallel_base_imaginary = incident_cosine * squared_eta_imaginary;
    const auto parallel = checked_squared_magnitude_ratio(
        parallel_base_real - scaled_root_real, parallel_base_imaginary - scaled_root_imaginary,
        parallel_base_real + scaled_root_real, parallel_base_imaginary + scaled_root_imaginary);
    if (!parallel) {
        return std::unexpected(parallel.error());
    }

    const auto reflectance = Scalar{0.5} * (*perpendicular + *parallel);
    if (!std::isfinite(reflectance) || !(reflectance > Scalar{0}) || reflectance > Scalar{1}) {
        return std::unexpected(
            invalid_fresnel("Conductor Fresnel reflectance is not representable."));
    }
    return reflectance;
}

} // namespace fresnel_detail

// Returns the exact unpolarized power reflectance of a lossless dielectric interface. The caller
// supplies |cos(theta_i)| in [0, 1], eta_incident for the side containing the incident direction,
// and eta_transmitted for the other side. The function never clamps the cosine, takes its absolute
// value, or swaps the media implicitly. At and beyond the critical angle it returns exactly one.
template <FresnelScalar Scalar>
[[nodiscard]] core::Result<Scalar> dielectric_fresnel(const Scalar incident_cosine,
                                                      const Scalar eta_incident,
                                                      const Scalar eta_transmitted) {
    if (!fresnel_detail::valid_incident_cosine(incident_cosine)) {
        return std::unexpected(fresnel_detail::invalid_fresnel(
            "Dielectric Fresnel requires a finite incident-cosine magnitude in [0, 1]."));
    }
    if (!fresnel_detail::valid_refractive_index(eta_incident) ||
        !fresnel_detail::valid_refractive_index(eta_transmitted)) {
        return std::unexpected(fresnel_detail::invalid_fresnel(
            "Dielectric Fresnel requires finite strictly positive refractive indices."));
    }

    // With no optical interface the limiting reflectance is zero, including at grazing incidence
    // where the direct polarization formulas would otherwise contain zero divided by zero.
    if (eta_incident == eta_transmitted) {
        return Scalar{0};
    }
    if (incident_cosine == Scalar{0}) {
        return Scalar{1};
    }

    const auto incident_is_denser = eta_incident > eta_transmitted;
    const auto normalized_eta =
        incident_is_denser ? eta_transmitted / eta_incident : eta_incident / eta_transmitted;
    if (incident_cosine == Scalar{1}) {
        const auto amplitude = (Scalar{1} - normalized_eta) / (Scalar{1} + normalized_eta);
        const auto reflectance = amplitude * amplitude;
        if (!std::isfinite(reflectance) || !(reflectance > Scalar{0}) || reflectance > Scalar{1}) {
            return std::unexpected(fresnel_detail::invalid_fresnel(
                "Dielectric Fresnel reflectance is not representable."));
        }
        return reflectance;
    }

    const auto incident_sine =
        std::sqrt((Scalar{1} - incident_cosine) * (Scalar{1} + incident_cosine));
    auto transmitted_sine = Scalar{};
    if (incident_is_denser) {
        // Comparing against the inverse ratio avoids forming eta_i / eta_t, which may overflow.
        // A ratio that underflows to zero still classifies every non-normal direction correctly.
        if (normalized_eta == Scalar{0} || incident_sine >= normalized_eta) {
            return Scalar{1};
        }
        transmitted_sine = incident_sine / normalized_eta;
    } else {
        transmitted_sine = normalized_eta * incident_sine;
    }

    // Rounding immediately below the critical angle can make sin(theta_t) exactly one. Its
    // representable Fresnel limit is total reflection, never a clamped transmitted direction.
    if (!(transmitted_sine < Scalar{1})) {
        return Scalar{1};
    }
    const auto transmitted_cosine =
        std::sqrt((Scalar{1} - transmitted_sine) * (Scalar{1} + transmitted_sine));

    // Dividing both polarization formulas by the larger index keeps every product and denominator
    // representable. Which amplitude is parallel or perpendicular swaps with medium ordering, but
    // their unpolarized average is unchanged.
    const auto first_eta_cosine = normalized_eta * transmitted_cosine;
    const auto first_amplitude =
        (incident_cosine - first_eta_cosine) / (incident_cosine + first_eta_cosine);
    const auto second_eta_cosine = normalized_eta * incident_cosine;
    const auto second_amplitude =
        (second_eta_cosine - transmitted_cosine) / (second_eta_cosine + transmitted_cosine);
    const auto reflectance = Scalar{0.5} * std::fma(first_amplitude, first_amplitude,
                                                    second_amplitude * second_amplitude);
    if (!std::isfinite(reflectance) || !(reflectance > Scalar{0}) || reflectance > Scalar{1}) {
        return std::unexpected(fresnel_detail::invalid_fresnel(
            "Dielectric Fresnel reflectance is not representable."));
    }
    return reflectance;
}

// Returns exact unpolarized conductor reflectance for the four transported wavelengths. eta and k
// are the real and non-negative imaginary parts of the transmitted complex IOR divided by the real
// IOR of the non-absorbing incident medium. The cosine is a shared magnitude in [0, 1]. Validation
// is atomic across the packet: no lane is clamped, absolutized, substituted, or returned alone.
template <FresnelScalar Scalar>
[[nodiscard]] core::Result<SampledSpectrum<TransportSpectrumSampleCount, Scalar>>
conductor_fresnel(const Scalar incident_cosine,
                  const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& relative_eta,
                  const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& relative_k) {
    if (!fresnel_detail::valid_incident_cosine(incident_cosine)) {
        return std::unexpected(fresnel_detail::invalid_fresnel(
            "Conductor Fresnel requires a finite incident-cosine magnitude in [0, 1]."));
    }
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        if (!fresnel_detail::valid_refractive_index(relative_eta[lane])) {
            return std::unexpected(fresnel_detail::invalid_fresnel(
                "Conductor Fresnel requires every relative eta lane to be finite and strictly "
                "positive."));
        }
        if (!fresnel_detail::valid_extinction_coefficient(relative_k[lane])) {
            return std::unexpected(fresnel_detail::invalid_fresnel(
                "Conductor Fresnel requires every relative k lane to be finite and "
                "non-negative."));
        }
    }

    auto reflectance = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto evaluated = fresnel_detail::conductor_fresnel_lane(
            incident_cosine, relative_eta[lane], relative_k[lane]);
        if (!evaluated) {
            return std::unexpected(evaluated.error());
        }
        reflectance[lane] = *evaluated;
    }
    return reflectance;
}

} // namespace blackframe::renderer
