#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

enum class MisHeuristic : std::uint8_t {
    balance,
    power,
};

template <typename Scalar>
concept MisScalar = std::same_as<Scalar, TransportScalar> || std::same_as<Scalar, ReferenceScalar>;

template <MisScalar Scalar>
using MisProbabilityDensityT = std::conditional_t<std::same_as<Scalar, TransportScalar>,
                                                  ProbabilityDensity, ReferenceProbabilityDensity>;

namespace mis_detail {

[[nodiscard]] inline core::Error invalid_mis(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

[[nodiscard]] constexpr bool supported_measure(const ProbabilityMeasure measure) noexcept {
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

template <MisScalar Scalar>
[[nodiscard]] core::Status validate_pdfs(const MisProbabilityDensityT<Scalar> sampled,
                                         const MisProbabilityDensityT<Scalar> competing) {
    if (!supported_measure(sampled.measure) || !supported_measure(competing.measure)) {
        return std::unexpected(invalid_mis("MIS PDFs must use a supported probability measure."));
    }
    if (sampled.measure != competing.measure) {
        return std::unexpected(invalid_mis("MIS PDFs must use the same probability measure."));
    }
    if (!std::isfinite(sampled.value) || !(sampled.value > Scalar{0})) {
        return std::unexpected(
            invalid_mis("The sampled-technique MIS PDF must be finite and strictly positive."));
    }
    if (!std::isfinite(competing.value) || competing.value < Scalar{0}) {
        return std::unexpected(
            invalid_mis("The competing-technique MIS PDF must be finite and non-negative."));
    }
    if (sampled.measure == ProbabilityMeasure::discrete &&
        (sampled.value > Scalar{1} || competing.value > Scalar{1})) {
        return std::unexpected(invalid_mis("Discrete MIS probabilities must not exceed one."));
    }
    return {};
}

template <MisScalar Scalar>
[[nodiscard]] Scalar ratio_at_most_one(const Scalar numerator, const Scalar denominator) noexcept {
    auto numerator_exponent = 0;
    auto denominator_exponent = 0;
    const auto numerator_significand = std::frexp(numerator, &numerator_exponent);
    const auto denominator_significand = std::frexp(denominator, &denominator_exponent);
    return std::scalbn(numerator_significand / denominator_significand,
                       numerator_exponent - denominator_exponent);
}

template <MisScalar Scalar>
[[nodiscard]] Scalar squared_ratio_at_most_one(const Scalar numerator,
                                               const Scalar denominator) noexcept {
    auto numerator_exponent = 0;
    auto denominator_exponent = 0;
    const auto numerator_significand = std::frexp(numerator, &numerator_exponent);
    const auto denominator_significand = std::frexp(denominator, &denominator_exponent);
    const auto significand_ratio = numerator_significand / denominator_significand;
    const auto exponent = 2 * (numerator_exponent - denominator_exponent);
    return std::scalbn(significand_ratio * significand_ratio, exponent);
}

template <MisScalar Scalar>
[[nodiscard]] core::Result<Scalar> checked_small_weight(const Scalar ratio) {
    if (!(ratio > Scalar{0})) {
        return std::unexpected(invalid_mis(
            "The positive MIS weight is not representable in the requested precision."));
    }
    const auto weight = ratio / (Scalar{1} + ratio);
    if (!std::isfinite(weight) || !(weight > Scalar{0})) {
        return std::unexpected(invalid_mis(
            "The positive MIS weight is not representable in the requested precision."));
    }
    return weight;
}

} // namespace mis_detail

// The arguments are the complete densities for the two techniques, including
// any discrete selection factors. They must already use the same measure.
template <MisScalar Scalar>
[[nodiscard]] core::Result<Scalar>
balance_heuristic(const MisProbabilityDensityT<Scalar> sampled,
                  const MisProbabilityDensityT<Scalar> competing) {
    const auto validation = mis_detail::validate_pdfs<Scalar>(sampled, competing);
    if (!validation) {
        return std::unexpected(validation.error());
    }
    if (competing.value == Scalar{0}) {
        return Scalar{1};
    }
    if (sampled.value < competing.value) {
        return mis_detail::checked_small_weight(
            mis_detail::ratio_at_most_one(sampled.value, competing.value));
    }
    const auto ratio = mis_detail::ratio_at_most_one(competing.value, sampled.value);
    return Scalar{1} / (Scalar{1} + ratio);
}

// The power heuristic uses exponent two. Ratio-space evaluation avoids
// squaring either input density, even when a valid PDF is near its type's max.
template <MisScalar Scalar>
[[nodiscard]] core::Result<Scalar> power_heuristic(const MisProbabilityDensityT<Scalar> sampled,
                                                   const MisProbabilityDensityT<Scalar> competing) {
    const auto validation = mis_detail::validate_pdfs<Scalar>(sampled, competing);
    if (!validation) {
        return std::unexpected(validation.error());
    }
    if (competing.value == Scalar{0}) {
        return Scalar{1};
    }
    if (sampled.value < competing.value) {
        return mis_detail::checked_small_weight(
            mis_detail::squared_ratio_at_most_one(sampled.value, competing.value));
    }
    const auto squared_ratio =
        mis_detail::squared_ratio_at_most_one(competing.value, sampled.value);
    return Scalar{1} / (Scalar{1} + squared_ratio);
}

template <MisScalar Scalar>
[[nodiscard]] core::Result<Scalar> mis_weight(const MisHeuristic heuristic,
                                              const MisProbabilityDensityT<Scalar> sampled,
                                              const MisProbabilityDensityT<Scalar> competing) {
    switch (heuristic) {
    case MisHeuristic::balance:
        return balance_heuristic<Scalar>(sampled, competing);
    case MisHeuristic::power:
        return power_heuristic<Scalar>(sampled, competing);
    }
    return std::unexpected(mis_detail::invalid_mis("The MIS heuristic is not supported."));
}

// Forms the complete solid-angle density of first selecting a light slot and
// then sampling a direction from that light. Keeping this operation explicit
// prevents the discrete selection factor from being omitted or applied twice.
template <MisScalar Scalar>
[[nodiscard]] core::Result<MisProbabilityDensityT<Scalar>>
joint_light_pdf(const MisProbabilityDensityT<Scalar> selection_probability,
                const MisProbabilityDensityT<Scalar> conditional_probability) {
    if (selection_probability.measure != ProbabilityMeasure::discrete ||
        !std::isfinite(selection_probability.value) || !(selection_probability.value > Scalar{0}) ||
        selection_probability.value > Scalar{1}) {
        return std::unexpected(mis_detail::invalid_mis(
            "A joint light PDF requires a discrete selection probability in (0, 1]."));
    }
    if (conditional_probability.measure != ProbabilityMeasure::solid_angle ||
        !std::isfinite(conditional_probability.value) ||
        conditional_probability.value < Scalar{0}) {
        return std::unexpected(mis_detail::invalid_mis(
            "A joint light PDF requires a finite non-negative conditional solid-angle PDF."));
    }
    if (conditional_probability.value == Scalar{0}) {
        return MisProbabilityDensityT<Scalar>{
            .value = Scalar{0},
            .measure = ProbabilityMeasure::solid_angle,
        };
    }

    auto selection_exponent = 0;
    auto conditional_exponent = 0;
    const auto selection_significand = std::frexp(selection_probability.value, &selection_exponent);
    const auto conditional_significand =
        std::frexp(conditional_probability.value, &conditional_exponent);
    auto normalization_exponent = 0;
    const auto normalized_significand =
        std::frexp(selection_significand * conditional_significand, &normalization_exponent);
    const auto joint = std::scalbn(
        normalized_significand, selection_exponent + conditional_exponent + normalization_exponent);
    if (!std::isfinite(joint) || !(joint > Scalar{0})) {
        return std::unexpected(mis_detail::invalid_mis(
            "The positive joint light PDF is not representable in the requested precision."));
    }
    return MisProbabilityDensityT<Scalar>{
        .value = joint,
        .measure = ProbabilityMeasure::solid_angle,
    };
}

[[nodiscard]] inline core::Result<TransportScalar>
balance_heuristic(const ProbabilityDensity sampled, const ProbabilityDensity competing) {
    return balance_heuristic<TransportScalar>(sampled, competing);
}

[[nodiscard]] inline core::Result<ReferenceScalar>
balance_heuristic(const ReferenceProbabilityDensity sampled,
                  const ReferenceProbabilityDensity competing) {
    return balance_heuristic<ReferenceScalar>(sampled, competing);
}

[[nodiscard]] inline core::Result<TransportScalar>
power_heuristic(const ProbabilityDensity sampled, const ProbabilityDensity competing) {
    return power_heuristic<TransportScalar>(sampled, competing);
}

[[nodiscard]] inline core::Result<ReferenceScalar>
power_heuristic(const ReferenceProbabilityDensity sampled,
                const ReferenceProbabilityDensity competing) {
    return power_heuristic<ReferenceScalar>(sampled, competing);
}

[[nodiscard]] inline core::Result<TransportScalar> mis_weight(const MisHeuristic heuristic,
                                                              const ProbabilityDensity sampled,
                                                              const ProbabilityDensity competing) {
    return mis_weight<TransportScalar>(heuristic, sampled, competing);
}

[[nodiscard]] inline core::Result<ReferenceScalar>
mis_weight(const MisHeuristic heuristic, const ReferenceProbabilityDensity sampled,
           const ReferenceProbabilityDensity competing) {
    return mis_weight<ReferenceScalar>(heuristic, sampled, competing);
}

[[nodiscard]] inline core::Result<ProbabilityDensity>
joint_light_pdf(const ProbabilityDensity selection_probability,
                const ProbabilityDensity conditional_probability) {
    return joint_light_pdf<TransportScalar>(selection_probability, conditional_probability);
}

[[nodiscard]] inline core::Result<ReferenceProbabilityDensity>
joint_light_pdf(const ReferenceProbabilityDensity selection_probability,
                const ReferenceProbabilityDensity conditional_probability) {
    return joint_light_pdf<ReferenceScalar>(selection_probability, conditional_probability);
}

static_assert(sizeof(MisHeuristic) == sizeof(std::uint8_t));

} // namespace blackframe::renderer
