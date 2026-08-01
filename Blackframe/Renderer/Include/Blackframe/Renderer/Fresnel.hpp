#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
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

} // namespace blackframe::renderer
