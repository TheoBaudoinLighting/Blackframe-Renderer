#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/ClosureSet.hpp>
#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <Blackframe/Renderer/RoughConductorReflection.hpp>
#include <Blackframe/Renderer/RoughDielectric.hpp>
#include <Blackframe/Renderer/RoughDiffuseReflection.hpp>
#include <Blackframe/Renderer/SpecularDelta.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {

template <SpectrumScalar Scalar>
using ClosureProbabilityDensityT =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ProbabilityDensity,
                       ReferenceProbabilityDensity>;

template <SpectrumScalar Scalar> struct ClosureMixtureSampleT final {
    std::uint32_t selected_closure{};
    ScatteringLobe lobes{ScatteringLobe::none};
    ClosureProbabilityDensityT<Scalar> selection_probability{
        .value = Scalar{0},
        .measure = ProbabilityMeasure::discrete,
    };
    Vector3T<Scalar> incoming_local{};
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> value{};
    ClosureProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = ContinuousBsdfProbabilityMeasure,
    };
    Scalar eta_scale_multiplier{Scalar{1}};
};

using ClosureMixtureSample = ClosureMixtureSampleT<TransportScalar>;
using ReferenceClosureMixtureSample = ClosureMixtureSampleT<ReferenceScalar>;

namespace closure_mixture_detail {

[[nodiscard]] inline core::Error invalid_closure_mixture(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool unit_local_direction(const Vector3T<Scalar> direction) noexcept {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return false;
    }
    const auto squared_length = std::fma(
        direction.x, direction.x, std::fma(direction.y, direction.y, direction.z * direction.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool canonical_component_sample(const Scalar sample) noexcept {
    return std::isfinite(sample) && sample >= Scalar{0} && sample < Scalar{1};
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool canonical_direction_sample(const Point2T<Scalar> sample) noexcept {
    return std::isfinite(sample.x) && sample.x >= Scalar{0} && sample.x < Scalar{1} &&
           std::isfinite(sample.y) && sample.y >= Scalar{0} && sample.y < Scalar{1};
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool add_compensated(const Scalar value, Scalar& sum, Scalar& correction) noexcept {
    if (!std::isfinite(value) || value < Scalar{0}) {
        return false;
    }
    const auto next = sum + value;
    if (!std::isfinite(next)) {
        return false;
    }
    if (sum >= value) {
        correction += (sum - next) + value;
    } else {
        correction += (value - next) + sum;
    }
    if (!std::isfinite(correction)) {
        return false;
    }
    sum = next;
    return true;
}

template <SpectrumScalar Scalar> struct ComponentDistributionT final {
    std::array<Scalar, MaximumClosureCount> probabilities{};
    std::array<Scalar, MaximumClosureCount + 1U> cdf{};
    std::size_t count{};
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar> compensated_sum(const std::span<const Scalar> values) {
    auto sum = Scalar{0};
    auto correction = Scalar{0};
    for (const auto value : values) {
        if (!add_compensated(value, sum, correction)) {
            return std::unexpected(invalid_closure_mixture(
                "Closure-mixture probability accumulation is not representable."));
        }
    }
    const auto total = sum + correction;
    if (!std::isfinite(total) || total < Scalar{0}) {
        return std::unexpected(invalid_closure_mixture(
            "Closure-mixture probability accumulation is not representable."));
    }
    return total == Scalar{0} ? Scalar{0} : total;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ComponentDistributionT<Scalar>>
build_component_distribution(const std::span<const Scalar> probabilities,
                             const std::size_t expected_count, const bool allow_empty) {
    if (expected_count > static_cast<std::size_t>(MaximumClosureCount) ||
        probabilities.size() != expected_count) {
        return std::unexpected(invalid_closure_mixture(
            "Closure-mixture probabilities must match the active closure count."));
    }

    auto distribution = ComponentDistributionT<Scalar>{.count = expected_count};
    if (probabilities.empty()) {
        if (allow_empty) {
            return distribution;
        }
        return std::unexpected(
            invalid_closure_mixture("A probability mixture requires at least one component."));
    }

    if (probabilities.size() == 1U) {
        if (probabilities.front() != Scalar{1}) {
            return std::unexpected(
                invalid_closure_mixture("A singleton closure mixture requires probability one."));
        }
        distribution.probabilities.front() = Scalar{1};
        distribution.cdf[1] = Scalar{1};
        return distribution;
    }

    for (const auto probability : probabilities) {
        if (!std::isfinite(probability) || !(probability > Scalar{0}) ||
            !(probability < Scalar{1})) {
            return std::unexpected(invalid_closure_mixture(
                "A multi-closure mixture requires finite probabilities strictly between zero "
                "and one."));
        }
    }

    constexpr auto scalar_digits = std::numeric_limits<Scalar>::digits;
    static_assert(scalar_digits > 0 && scalar_digits < 64);
    constexpr auto probability_scale = std::uint64_t{1} << scalar_digits;
    auto masses = std::array<std::uint64_t, MaximumClosureCount>{};
    auto remainders = std::array<Scalar, MaximumClosureCount>{};
    auto base_mass = std::uint64_t{0};
    for (auto index = std::size_t{}; index < probabilities.size(); ++index) {
        const auto scaled = std::ldexp(probabilities[index], scalar_digits);
        const auto base = std::floor(scaled);
        if (!std::isfinite(scaled) || !(scaled > Scalar{0}) || base < Scalar{0} ||
            base > static_cast<Scalar>(probability_scale)) {
            return std::unexpected(invalid_closure_mixture(
                "A closure-mixture probability cannot be projected onto the sampler grid."));
        }
        masses[index] = static_cast<std::uint64_t>(base);
        remainders[index] = scaled - base;
        if (!std::isfinite(remainders[index]) || remainders[index] < Scalar{0} ||
            remainders[index] >= Scalar{1} ||
            base_mass > std::numeric_limits<std::uint64_t>::max() - masses[index]) {
            return std::unexpected(invalid_closure_mixture(
                "A closure-mixture probability cannot be projected onto the sampler grid."));
        }
        base_mass += masses[index];
    }
    if (base_mass > probability_scale) {
        return std::unexpected(invalid_closure_mixture(
            "Closure-mixture probabilities exceed one on the sampler grid."));
    }

    const auto remainder_total =
        compensated_sum(std::span<const Scalar>{remainders.data(), probabilities.size()});
    if (!remainder_total) {
        return std::unexpected(remainder_total.error());
    }
    const auto missing_mass = probability_scale - base_mass;
    if (missing_mass > probabilities.size()) {
        return std::unexpected(invalid_closure_mixture(
            "Closure-mixture probabilities do not sum to one at sampler precision."));
    }
    constexpr auto half_grid_cell = Scalar{0.5};
    constexpr auto projection_tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{16};
    if (std::abs(*remainder_total - static_cast<Scalar>(missing_mass)) >
        half_grid_cell + projection_tolerance) {
        return std::unexpected(invalid_closure_mixture(
            "Closure-mixture probabilities do not round to one at sampler precision."));
    }

    auto received_remainder = std::array<bool, MaximumClosureCount>{};
    for (auto unit = std::uint64_t{0}; unit < missing_mass; ++unit) {
        auto selected = probabilities.size();
        for (auto index = std::size_t{}; index < probabilities.size(); ++index) {
            if (received_remainder[index] || !(remainders[index] > Scalar{0})) {
                continue;
            }
            if (selected == probabilities.size() || remainders[index] > remainders[selected]) {
                selected = index;
            }
        }
        if (selected == probabilities.size()) {
            return std::unexpected(invalid_closure_mixture(
                "Closure-mixture rounding cannot preserve every positive component."));
        }
        ++masses[selected];
        received_remainder[selected] = true;
    }

    auto cumulative_mass = std::uint64_t{0};
    distribution.cdf.front() = Scalar{0};
    for (auto index = std::size_t{}; index < probabilities.size(); ++index) {
        if (masses[index] == 0U || cumulative_mass > probability_scale - masses[index]) {
            return std::unexpected(invalid_closure_mixture(
                "A positive closure-mixture probability is not representable on the sampler "
                "grid."));
        }
        cumulative_mass += masses[index];
        distribution.probabilities[index] =
            std::ldexp(static_cast<Scalar>(masses[index]), -scalar_digits);
        distribution.cdf[index + 1U] =
            std::ldexp(static_cast<Scalar>(cumulative_mass), -scalar_digits);
    }
    if (cumulative_mass != probability_scale ||
        distribution.cdf[probabilities.size()] != Scalar{1}) {
        return std::unexpected(
            invalid_closure_mixture("The closure-mixture CDF does not terminate at one."));
    }
    return distribution;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar> positive_probability_product(const Scalar selection,
                                                                const Scalar conditional) {
    if (conditional == Scalar{0}) {
        return Scalar{0};
    }
    auto selection_exponent = 0;
    auto conditional_exponent = 0;
    const auto selection_significand = std::frexp(selection, &selection_exponent);
    const auto conditional_significand = std::frexp(conditional, &conditional_exponent);
    auto normalization_exponent = 0;
    const auto normalized_significand =
        std::frexp(selection_significand * conditional_significand, &normalization_exponent);
    const auto product = std::scalbn(
        normalized_significand, selection_exponent + conditional_exponent + normalization_exponent);
    if (!std::isfinite(product) || !(product > Scalar{0})) {
        return std::unexpected(invalid_closure_mixture(
            "A positive weighted closure PDF is not representable in the requested precision."));
    }
    return product;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ClosureProbabilityDensityT<Scalar>> mix_probability_densities(
    const ComponentDistributionT<Scalar>& distribution,
    const std::span<const ClosureProbabilityDensityT<Scalar>> conditional_probabilities) {
    if (distribution.count == 0U || distribution.count != conditional_probabilities.size()) {
        return std::unexpected(invalid_closure_mixture(
            "Closure PDF components require a non-empty matching sampling distribution."));
    }
    auto maximum_density = Scalar{0};
    for (const auto probability : conditional_probabilities) {
        if (probability.measure != ContinuousBsdfProbabilityMeasure ||
            !std::isfinite(probability.value) || probability.value < Scalar{0}) {
            return std::unexpected(invalid_closure_mixture(
                "Closure PDF components require finite non-negative solid-angle densities."));
        }
        maximum_density = std::max(maximum_density, probability.value);
    }
    if (conditional_probabilities.size() == 1U) {
        return conditional_probabilities.front();
    }
    if (maximum_density == Scalar{0}) {
        return ClosureProbabilityDensityT<Scalar>{
            .value = Scalar{0},
            .measure = ContinuousBsdfProbabilityMeasure,
        };
    }

    // Scaling by the largest conditional density keeps every intermediate in [0, 1]. The dyadic
    // component probabilities sum exactly to one in the selected scalar precision.
    auto normalized_sum = Scalar{0};
    auto normalized_correction = Scalar{0};
    for (auto index = std::size_t{}; index < conditional_probabilities.size(); ++index) {
        const auto density = conditional_probabilities[index].value;
        const auto ratio = density / maximum_density;
        if (density > Scalar{0} && (!std::isfinite(ratio) || !(ratio > Scalar{0}))) {
            return std::unexpected(invalid_closure_mixture(
                "A positive normalized closure PDF is not representable in the requested "
                "precision."));
        }
        const auto contribution =
            positive_probability_product(distribution.probabilities[index], ratio);
        if (!contribution) {
            return std::unexpected(contribution.error());
        }
        if (!add_compensated(*contribution, normalized_sum, normalized_correction)) {
            return std::unexpected(invalid_closure_mixture(
                "The weighted closure PDF sum is not representable in the requested precision."));
        }
    }
    const auto normalized_density = normalized_sum + normalized_correction;
    if (!std::isfinite(normalized_density) || !(normalized_density > Scalar{0}) ||
        normalized_density > Scalar{1}) {
        return std::unexpected(invalid_closure_mixture(
            "The normalized closure PDF sum is not representable in the requested precision."));
    }
    const auto mixed = positive_probability_product(normalized_density, maximum_density);
    if (!mixed) {
        return std::unexpected(mixed.error());
    }
    return ClosureProbabilityDensityT<Scalar>{
        .value = *mixed,
        .measure = ContinuousBsdfProbabilityMeasure,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<LambertianReflectionT<Scalar>>
lambertian_from_record(const ClosureT<Scalar>& closure) {
    constexpr auto lambertian_lobes = ScatteringLobe::diffuse | ScatteringLobe::reflection;
    if (closure.kind != ClosureKind::lambertian_reflection) {
        return std::unexpected(
            invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
    }
    if (closure.lobes != lambertian_lobes) {
        return std::unexpected(
            invalid_closure_mixture("A Lambertian closure record has an incompatible lobe mask."));
    }
    for (const auto parameter : closure.parameters) {
        if (parameter != Scalar{0}) {
            return std::unexpected(invalid_closure_mixture(
                "A Lambertian closure record has a non-zero reserved payload."));
        }
    }
    return LambertianReflectionT<Scalar>::create(closure.weight);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RoughDiffuseReflectionT<Scalar>>
rough_diffuse_from_record(const ClosureT<Scalar>& closure) {
    constexpr auto rough_diffuse_lobes = ScatteringLobe::diffuse | ScatteringLobe::reflection;
    if (closure.kind != ClosureKind::rough_diffuse_reflection) {
        return std::unexpected(
            invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
    }
    if (closure.lobes != rough_diffuse_lobes) {
        return std::unexpected(invalid_closure_mixture(
            "A rough-diffuse closure record has an incompatible lobe mask."));
    }
    for (auto index = std::size_t{1}; index < closure.parameters.size(); ++index) {
        if (closure.parameters[index] != Scalar{0}) {
            return std::unexpected(invalid_closure_mixture(
                "A rough-diffuse closure record has a non-zero reserved payload."));
        }
    }
    return RoughDiffuseReflectionT<Scalar>::create(closure.weight, closure.parameters[0]);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RoughConductorReflectionT<Scalar>>
rough_conductor_from_record(const ClosureT<Scalar>& closure) {
    constexpr auto rough_conductor_lobes = ScatteringLobe::glossy | ScatteringLobe::reflection;
    if (closure.kind != ClosureKind::rough_conductor_reflection) {
        return std::unexpected(
            invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
    }
    if (closure.lobes != rough_conductor_lobes) {
        return std::unexpected(invalid_closure_mixture(
            "A rough-conductor closure record has an incompatible lobe mask."));
    }

    auto relative_eta = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    auto relative_k = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        relative_eta[lane] = closure.parameters[lane];
        relative_k[lane] = closure.parameters[TransportSpectrumSampleCount + lane];
    }
    constexpr auto alpha_x_parameter = TransportSpectrumSampleCount * 2U;
    constexpr auto alpha_y_parameter = alpha_x_parameter + 1U;
    return RoughConductorReflectionT<Scalar>::create(closure.weight, relative_eta, relative_k,
                                                     closure.parameters[alpha_x_parameter],
                                                     closure.parameters[alpha_y_parameter]);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RoughDielectricT<Scalar>>
rough_dielectric_from_record(const ClosureT<Scalar>& closure) {
    constexpr auto rough_dielectric_lobes =
        ScatteringLobe::glossy | ScatteringLobe::reflection | ScatteringLobe::transmission;
    if (closure.kind != ClosureKind::rough_dielectric) {
        return std::unexpected(
            invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
    }
    if (closure.lobes != rough_dielectric_lobes) {
        return std::unexpected(invalid_closure_mixture(
            "A rough-dielectric closure record has an incompatible lobe mask."));
    }
    for (auto index = std::size_t{4}; index < closure.parameters.size(); ++index) {
        if (closure.parameters[index] != Scalar{0}) {
            return std::unexpected(invalid_closure_mixture(
                "A rough-dielectric closure record has a non-zero reserved payload."));
        }
    }
    return RoughDielectricT<Scalar>::create(closure.weight, closure.parameters[0],
                                            closure.parameters[1], closure.parameters[2],
                                            closure.parameters[3]);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SpecularReflectionT<Scalar>>
specular_reflection_from_record(const ClosureT<Scalar>& closure) {
    constexpr auto specular_reflection_lobes =
        ScatteringLobe::specular | ScatteringLobe::reflection;
    if (closure.kind != ClosureKind::specular_reflection) {
        return std::unexpected(
            invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
    }
    if (closure.lobes != specular_reflection_lobes) {
        return std::unexpected(invalid_closure_mixture(
            "A specular-reflection closure record has an incompatible lobe mask."));
    }
    for (const auto parameter : closure.parameters) {
        if (parameter != Scalar{0}) {
            return std::unexpected(invalid_closure_mixture(
                "A specular-reflection closure record has a non-zero reserved payload."));
        }
    }
    return SpecularReflectionT<Scalar>::create(closure.weight);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SpecularTransmissionT<Scalar>>
specular_transmission_from_record(const ClosureT<Scalar>& closure) {
    constexpr auto specular_transmission_lobes =
        ScatteringLobe::specular | ScatteringLobe::transmission;
    if (closure.kind != ClosureKind::specular_transmission) {
        return std::unexpected(
            invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
    }
    if (closure.lobes != specular_transmission_lobes) {
        return std::unexpected(invalid_closure_mixture(
            "A specular-transmission closure record has an incompatible lobe mask."));
    }
    for (auto index = std::size_t{2}; index < closure.parameters.size(); ++index) {
        if (closure.parameters[index] != Scalar{0}) {
            return std::unexpected(invalid_closure_mixture(
                "A specular-transmission closure record has a non-zero reserved payload."));
        }
    }
    return SpecularTransmissionT<Scalar>::create(closure.weight, closure.parameters[0],
                                                 closure.parameters[1]);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_closure_record(const ClosureT<Scalar>& closure) {
    switch (closure.kind) {
    case ClosureKind::lambertian_reflection: {
        const auto model = lambertian_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return {};
    }
    case ClosureKind::rough_diffuse_reflection: {
        const auto model = rough_diffuse_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return {};
    }
    case ClosureKind::rough_conductor_reflection: {
        const auto model = rough_conductor_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return {};
    }
    case ClosureKind::rough_dielectric: {
        const auto model = rough_dielectric_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return {};
    }
    case ClosureKind::specular_reflection: {
        const auto model = specular_reflection_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return {};
    }
    case ClosureKind::specular_transmission: {
        const auto model = specular_transmission_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return {};
    }
    case ClosureKind::none:
        break;
    }
    return std::unexpected(
        invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SampledSpectrum<TransportSpectrumSampleCount, Scalar>>
eval_closure_record(const ClosureT<Scalar>& closure, const Vector3T<Scalar> outgoing_local,
                    const Vector3T<Scalar> incoming_local, const TransportMode mode) {
    switch (closure.kind) {
    case ClosureKind::lambertian_reflection: {
        const auto model = lambertian_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->eval(outgoing_local, incoming_local);
    }
    case ClosureKind::rough_diffuse_reflection: {
        const auto model = rough_diffuse_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->eval(outgoing_local, incoming_local);
    }
    case ClosureKind::rough_conductor_reflection: {
        const auto model = rough_conductor_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->eval(outgoing_local, incoming_local);
    }
    case ClosureKind::rough_dielectric: {
        const auto model = rough_dielectric_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->eval(outgoing_local, incoming_local, mode);
    }
    case ClosureKind::specular_reflection: {
        const auto model = specular_reflection_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->eval(outgoing_local, incoming_local);
    }
    case ClosureKind::specular_transmission: {
        const auto model = specular_transmission_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->eval(outgoing_local, incoming_local, mode);
    }
    case ClosureKind::none:
        break;
    }
    return std::unexpected(
        invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ClosureProbabilityDensityT<Scalar>>
pdf_closure_record(const ClosureT<Scalar>& closure, const Vector3T<Scalar> outgoing_local,
                   const Vector3T<Scalar> incoming_local, const TransportMode mode) {
    switch (closure.kind) {
    case ClosureKind::lambertian_reflection: {
        const auto model = lambertian_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->pdf(outgoing_local, incoming_local);
    }
    case ClosureKind::rough_diffuse_reflection: {
        const auto model = rough_diffuse_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->pdf(outgoing_local, incoming_local);
    }
    case ClosureKind::rough_conductor_reflection: {
        const auto model = rough_conductor_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->pdf(outgoing_local, incoming_local);
    }
    case ClosureKind::rough_dielectric: {
        const auto model = rough_dielectric_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->pdf(outgoing_local, incoming_local, mode);
    }
    case ClosureKind::specular_reflection: {
        const auto model = specular_reflection_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->pdf(outgoing_local, incoming_local);
    }
    case ClosureKind::specular_transmission: {
        const auto model = specular_transmission_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        return model->pdf(outgoing_local, incoming_local, mode);
    }
    case ClosureKind::none:
        break;
    }
    return std::unexpected(
        invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
}

template <SpectrumScalar Scalar> struct ClosureDirectionSampleT final {
    Vector3T<Scalar> incoming_local{};
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> value{};
    ClosureProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = ContinuousBsdfProbabilityMeasure,
    };
    ScatteringLobe lobes{ScatteringLobe::none};
    Scalar eta_scale_multiplier{Scalar{1}};
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::optional<ClosureDirectionSampleT<Scalar>>>
sample_closure_record(const ClosureT<Scalar>& closure, const Vector3T<Scalar> outgoing_local,
                      const Scalar event_sample, const Point2T<Scalar> direction_sample,
                      const TransportMode mode) {
    const auto convert =
        [](const auto& sampled,
           const ScatteringLobe lobes) -> std::optional<ClosureDirectionSampleT<Scalar>> {
        if (!sampled.has_value()) {
            return {};
        }
        return ClosureDirectionSampleT<Scalar>{
            .incoming_local = sampled->incoming_local,
            .value = sampled->value,
            .probability = sampled->probability,
            .lobes = lobes,
            .eta_scale_multiplier = Scalar{1},
        };
    };

    switch (closure.kind) {
    case ClosureKind::lambertian_reflection: {
        const auto model = lambertian_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        const auto sampled = model->sample(outgoing_local, direction_sample);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        return convert(*sampled, closure.lobes);
    }
    case ClosureKind::rough_diffuse_reflection: {
        const auto model = rough_diffuse_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        const auto sampled = model->sample(outgoing_local, direction_sample);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        return convert(*sampled, closure.lobes);
    }
    case ClosureKind::rough_conductor_reflection: {
        const auto model = rough_conductor_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        const auto sampled = model->sample(outgoing_local, direction_sample);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        return convert(*sampled, closure.lobes);
    }
    case ClosureKind::rough_dielectric: {
        const auto model = rough_dielectric_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        const auto sampled = model->sample(outgoing_local, event_sample, direction_sample, mode);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        if (!sampled->has_value()) {
            return std::optional<ClosureDirectionSampleT<Scalar>>{};
        }
        return std::optional<ClosureDirectionSampleT<Scalar>>{ClosureDirectionSampleT<Scalar>{
            .incoming_local = (**sampled).incoming_local,
            .value = (**sampled).value,
            .probability = (**sampled).probability,
            .lobes = (**sampled).lobes,
            .eta_scale_multiplier = (**sampled).eta_scale_multiplier,
        }};
    }
    case ClosureKind::specular_reflection: {
        const auto model = specular_reflection_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        const auto sampled = model->sample(outgoing_local);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        if (!sampled->has_value()) {
            return std::optional<ClosureDirectionSampleT<Scalar>>{};
        }
        return std::optional<ClosureDirectionSampleT<Scalar>>{ClosureDirectionSampleT<Scalar>{
            .incoming_local = (**sampled).incoming_local,
            .value = (**sampled).value,
            .probability = (**sampled).probability,
            .lobes = (**sampled).lobes,
            .eta_scale_multiplier = (**sampled).eta_scale_multiplier,
        }};
    }
    case ClosureKind::specular_transmission: {
        const auto model = specular_transmission_from_record(closure);
        if (!model) {
            return std::unexpected(model.error());
        }
        const auto sampled = model->sample(outgoing_local, mode);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        if (!sampled->has_value()) {
            return std::optional<ClosureDirectionSampleT<Scalar>>{};
        }
        return std::optional<ClosureDirectionSampleT<Scalar>>{ClosureDirectionSampleT<Scalar>{
            .incoming_local = (**sampled).incoming_local,
            .value = (**sampled).value,
            .probability = (**sampled).probability,
            .lobes = (**sampled).lobes,
            .eta_scale_multiplier = (**sampled).eta_scale_multiplier,
        }};
    }
    case ClosureKind::none:
        break;
    }
    return std::unexpected(
        invalid_closure_mixture("A closure mixture contains an unsupported closure record."));
}

[[nodiscard]] constexpr bool is_delta_closure_kind(const ClosureKind kind) noexcept {
    return kind == ClosureKind::specular_reflection || kind == ClosureKind::specular_transmission;
}

template <SpectrumScalar Scalar> struct DeltaAtomT final {
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> value{};
    ClosureProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = DeltaBsdfProbabilityMeasure,
    };
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<DeltaAtomT<Scalar>>
aggregate_delta_atom(const std::span<const ClosureT<Scalar>> closures,
                     const std::span<const Scalar> component_probabilities,
                     const Vector3T<Scalar> outgoing_local, const Vector3T<Scalar> incoming_local,
                     const TransportMode mode) {
    if (closures.empty() || closures.size() != component_probabilities.size()) {
        return std::unexpected(invalid_closure_mixture(
            "A delta atom requires a non-empty matching sampling distribution."));
    }

    auto value_sums = std::array<Scalar, TransportSpectrumSampleCount>{};
    auto value_corrections = std::array<Scalar, TransportSpectrumSampleCount>{};
    auto probability_sum = Scalar{0};
    auto probability_correction = Scalar{0};
    auto matching_count = std::size_t{};
    for (auto index = std::size_t{}; index < closures.size(); ++index) {
        if (!is_delta_closure_kind(closures[index].kind)) {
            continue;
        }

        // Delta closures are deterministic. Exact comparison deliberately merges only identical
        // representable atoms; nearby directions remain distinct discrete outcomes.
        const auto candidate = sample_closure_record(closures[index], outgoing_local, Scalar{0},
                                                     Point2T<Scalar>{}, mode);
        if (!candidate) {
            return std::unexpected(candidate.error());
        }
        if (!candidate->has_value() || (**candidate).incoming_local.x != incoming_local.x ||
            (**candidate).incoming_local.y != incoming_local.y ||
            (**candidate).incoming_local.z != incoming_local.z) {
            continue;
        }
        if ((**candidate).probability.measure != DeltaBsdfProbabilityMeasure ||
            !std::isfinite((**candidate).probability.value) ||
            !((**candidate).probability.value > Scalar{0}) ||
            (**candidate).probability.value > Scalar{1}) {
            return std::unexpected(invalid_closure_mixture(
                "A sampled delta closure requires a finite positive discrete probability."));
        }

        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            if (!add_compensated((**candidate).value[lane], value_sums[lane],
                                 value_corrections[lane])) {
                return std::unexpected(invalid_closure_mixture(
                    "A coincident delta value is not representable in the requested precision."));
            }
        }
        const auto weighted_probability = positive_probability_product(
            component_probabilities[index], (**candidate).probability.value);
        if (!weighted_probability) {
            return std::unexpected(weighted_probability.error());
        }
        if (!add_compensated(*weighted_probability, probability_sum, probability_correction)) {
            return std::unexpected(invalid_closure_mixture(
                "A coincident delta probability is not representable in the requested precision."));
        }
        ++matching_count;
    }

    const auto probability = probability_sum + probability_correction;
    if (matching_count == 0U || !std::isfinite(probability) || !(probability > Scalar{0}) ||
        probability > Scalar{1}) {
        return std::unexpected(
            invalid_closure_mixture("The sampled delta atom is absent from the closure mixture."));
    }

    auto value = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        value[lane] = value_sums[lane] + value_corrections[lane];
        if (!std::isfinite(value[lane]) || value[lane] < Scalar{0}) {
            return std::unexpected(invalid_closure_mixture(
                "A coincident delta value is not representable in the requested precision."));
        }
    }
    return DeltaAtomT<Scalar>{
        .value = value,
        .probability =
            {
                .value = probability,
                .measure = DeltaBsdfProbabilityMeasure,
            },
    };
}

} // namespace closure_mixture_detail

// The discrete component probabilities are supplied explicitly and are not inferred from spectral
// closure weights. Inputs that round to unit mass are projected deterministically onto the scalar
// sampler grid; the resulting dyadic q_i values and their authoritative CDF are observable. The
// result is p_mix(wi | wo) = sum_i q_i p_i(wi | wo) in solid angle; measures are never converted or
// combined implicitly.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ClosureProbabilityDensityT<Scalar>> mix_closure_probability_densities(
    const std::span<const Scalar> component_probabilities,
    const std::span<const ClosureProbabilityDensityT<Scalar>> conditional_probabilities) {
    const auto distribution = closure_mixture_detail::build_component_distribution(
        component_probabilities, conditional_probabilities.size(), false);
    if (!distribution) {
        return std::unexpected(distribution.error());
    }
    return closure_mixture_detail::mix_probability_densities(*distribution,
                                                             conditional_probabilities);
}

// A ClosureMixture owns both its bounded ClosureSet and its fixed-capacity sampling distribution.
// Closure coefficients already carry physical weights, so eval adds them without q_i. Only the PDF
// is weighted by q_i, exactly once. Construction only performs the documented sampler-grid
// projection; invalid or unrepresentable distributions are rejected and no closure is inferred,
// clamped, dropped, or reordered.
template <SpectrumScalar Scalar> class ClosureMixtureT final {
  public:
    using closure_set_type = ClosureSetT<Scalar>;
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using probability_density_type = ClosureProbabilityDensityT<Scalar>;
    using sample_type = ClosureMixtureSampleT<Scalar>;

    [[nodiscard]] static core::Result<ClosureMixtureT>
    create(closure_set_type closures, const std::span<const Scalar> component_probabilities) {
        if (closures.size() > closure_set_type::capacity()) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "A closure mixture contains an invalid active closure count."));
        }
        const auto distribution = closure_mixture_detail::build_component_distribution(
            component_probabilities, static_cast<std::size_t>(closures.size()), true);
        if (!distribution) {
            return std::unexpected(distribution.error());
        }
        for (const auto& closure : closures.closures()) {
            const auto status = closure_mixture_detail::validate_closure_record(closure);
            if (!status) {
                return std::unexpected(status.error());
            }
        }
        return ClosureMixtureT{std::move(closures), distribution->probabilities, distribution->cdf};
    }

    [[nodiscard]] constexpr const closure_set_type& closure_set() const noexcept {
        return closures_;
    }

    [[nodiscard]] constexpr std::span<const Scalar> component_probabilities() const noexcept {
        return {probabilities_.data(), static_cast<std::size_t>(closures_.size())};
    }

    [[nodiscard]] constexpr std::span<const Scalar> component_cdf() const noexcept {
        const auto boundary_count =
            closures_.empty() ? std::size_t{0} : static_cast<std::size_t>(closures_.size()) + 1U;
        return {cdf_.data(), boundary_count};
    }

    [[nodiscard]] core::Result<spectrum_type> eval(const Vector3T<Scalar> outgoing_local,
                                                   const Vector3T<Scalar> incoming_local,
                                                   const TransportMode mode) const {
        const auto query_status = validate_query(outgoing_local, incoming_local, mode);
        if (!query_status) {
            return std::unexpected(query_status.error());
        }
        if (closures_.empty()) {
            return spectrum_type{};
        }
        if (closures_.size() == 1U) {
            return closure_mixture_detail::eval_closure_record(
                closures_.closures().front(), outgoing_local, incoming_local, mode);
        }

        auto result = spectrum_type{};
        for (const auto& closure : closures_.closures()) {
            const auto component = closure_mixture_detail::eval_closure_record(
                closure, outgoing_local, incoming_local, mode);
            if (!component) {
                return std::unexpected(component.error());
            }
            for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
                const auto sum = result[lane] + (*component)[lane];
                if (!std::isfinite(sum)) {
                    return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                        "The closure-mixture value is not representable in the requested "
                        "precision."));
                }
                result[lane] = sum;
            }
        }
        return result;
    }

    [[nodiscard]] core::Result<probability_density_type> pdf(const Vector3T<Scalar> outgoing_local,
                                                             const Vector3T<Scalar> incoming_local,
                                                             const TransportMode mode) const {
        const auto query_status = validate_query(outgoing_local, incoming_local, mode);
        if (!query_status) {
            return std::unexpected(query_status.error());
        }
        if (closures_.empty()) {
            return probability_density_type{
                .value = Scalar{0},
                .measure = ContinuousBsdfProbabilityMeasure,
            };
        }
        if (closures_.size() == 1U) {
            return closure_mixture_detail::pdf_closure_record(closures_.closures().front(),
                                                              outgoing_local, incoming_local, mode);
        }

        auto conditional_probabilities =
            std::array<probability_density_type, MaximumClosureCount>{};
        for (auto index = std::size_t{}; index < closures_.closures().size(); ++index) {
            const auto component = closure_mixture_detail::pdf_closure_record(
                closures_.closures()[index], outgoing_local, incoming_local, mode);
            if (!component) {
                return std::unexpected(component.error());
            }
            conditional_probabilities[index] = *component;
        }
        const auto distribution = closure_mixture_detail::ComponentDistributionT<Scalar>{
            .probabilities = probabilities_,
            .cdf = cdf_,
            .count = static_cast<std::size_t>(closures_.size()),
        };
        return closure_mixture_detail::mix_probability_densities(
            distribution, std::span<const probability_density_type>{
                              conditional_probabilities.data(), closures_.closures().size()});
    }

    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample(const Vector3T<Scalar> outgoing_local, const Scalar component_sample,
           const Point2T<Scalar> direction_sample, const TransportMode mode) const {
        if (!is_known_transport_mode(mode)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Closure-mixture sampling requires a supported transport mode."));
        }
        if (!closure_mixture_detail::unit_local_direction(outgoing_local)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Closure-mixture directions must be finite unit vectors."));
        }
        if (!closure_mixture_detail::canonical_component_sample(component_sample)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Closure component selection requires a finite sample in [0, 1)."));
        }
        if (!closure_mixture_detail::canonical_direction_sample(direction_sample)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Closure direction sampling requires finite coordinates in [0, 1)."));
        }
        if (closures_.empty()) {
            return std::optional<sample_type>{};
        }

        const auto selected = select_component(component_sample);
        if (!selected) {
            return std::unexpected(selected.error());
        }
        const auto& closure = closures_.closures()[*selected];
        const auto event_sample = remap_component_sample(*selected, component_sample);
        if (!event_sample) {
            return std::unexpected(event_sample.error());
        }
        const auto sampled = closure_mixture_detail::sample_closure_record(
            closure, outgoing_local, *event_sample, direction_sample, mode);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        if (!sampled->has_value()) {
            return std::optional<sample_type>{};
        }

        if (closures_.size() == 1U) {
            return std::optional<sample_type>{sample_type{
                .selected_closure = 0U,
                .lobes = (**sampled).lobes,
                .selection_probability =
                    {
                        .value = Scalar{1},
                        .measure = ProbabilityMeasure::discrete,
                    },
                .incoming_local = (**sampled).incoming_local,
                .value = (**sampled).value,
                .probability = (**sampled).probability,
                .eta_scale_multiplier = (**sampled).eta_scale_multiplier,
            }};
        }

        auto value = spectrum_type{};
        auto probability = probability_density_type{};
        if ((**sampled).probability.measure == ContinuousBsdfProbabilityMeasure) {
            const auto mixed_value = eval(outgoing_local, (**sampled).incoming_local, mode);
            if (!mixed_value) {
                return std::unexpected(mixed_value.error());
            }
            const auto mixed_probability = pdf(outgoing_local, (**sampled).incoming_local, mode);
            if (!mixed_probability) {
                return std::unexpected(mixed_probability.error());
            }
            value = *mixed_value;
            probability = *mixed_probability;
        } else if ((**sampled).probability.measure == DeltaBsdfProbabilityMeasure) {
            const auto atom = closure_mixture_detail::aggregate_delta_atom(
                closures_.closures(), component_probabilities(), outgoing_local,
                (**sampled).incoming_local, mode);
            if (!atom) {
                return std::unexpected(atom.error());
            }
            value = atom->value;
            probability = atom->probability;
        } else {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "A sampled closure returned an unsupported probability measure."));
        }
        return std::optional<sample_type>{sample_type{
            .selected_closure = static_cast<std::uint32_t>(*selected),
            .lobes = (**sampled).lobes,
            .selection_probability =
                {
                    .value = probabilities_[*selected],
                    .measure = ProbabilityMeasure::discrete,
                },
            .incoming_local = (**sampled).incoming_local,
            .value = value,
            .probability = probability,
            .eta_scale_multiplier = (**sampled).eta_scale_multiplier,
        }};
    }

  private:
    constexpr ClosureMixtureT(closure_set_type closures,
                              std::array<Scalar, MaximumClosureCount> probabilities,
                              std::array<Scalar, MaximumClosureCount + 1U> cdf) noexcept
        : closures_{std::move(closures)}, probabilities_{probabilities}, cdf_{cdf} {}

    [[nodiscard]] static core::Status validate_query(const Vector3T<Scalar> outgoing_local,
                                                     const Vector3T<Scalar> incoming_local,
                                                     const TransportMode mode) {
        if (!is_known_transport_mode(mode)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Closure-mixture queries require a supported transport mode."));
        }
        if (!closure_mixture_detail::unit_local_direction(outgoing_local) ||
            !closure_mixture_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Closure-mixture directions must be finite unit vectors."));
        }
        return {};
    }

    [[nodiscard]] core::Result<Scalar> remap_component_sample(const std::size_t selected,
                                                              const Scalar sample) const {
        const auto lower = cdf_[selected];
        const auto probability = probabilities_[selected];
        const auto local = (sample - lower) / probability;
        if (!std::isfinite(local) || local < Scalar{0} || !(local < Scalar{1})) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "The selected closure sample cannot be remapped to [0, 1)."));
        }
        return local;
    }

    [[nodiscard]] core::Result<std::size_t> select_component(const Scalar sample) const {
        for (auto index = std::size_t{}; index < component_probabilities().size(); ++index) {
            if (sample < cdf_[index + 1U]) {
                return index;
            }
        }
        return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
            "The closure component CDF did not select a component."));
    }

    closure_set_type closures_;
    std::array<Scalar, MaximumClosureCount> probabilities_{};
    std::array<Scalar, MaximumClosureCount + 1U> cdf_{};
};

using ClosureMixture = ClosureMixtureT<TransportScalar>;
using ReferenceClosureMixture = ClosureMixtureT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<ClosureMixtureSample>);
static_assert(std::is_trivially_copyable_v<ClosureMixtureSample>);
static_assert(std::is_standard_layout_v<ReferenceClosureMixtureSample>);
static_assert(std::is_trivially_copyable_v<ReferenceClosureMixtureSample>);
static_assert(std::is_standard_layout_v<ClosureMixture>);
static_assert(std::is_trivially_copyable_v<ClosureMixture>);
static_assert(std::is_trivially_destructible_v<ClosureMixture>);
static_assert(std::is_standard_layout_v<ReferenceClosureMixture>);
static_assert(std::is_trivially_copyable_v<ReferenceClosureMixture>);
static_assert(std::is_trivially_destructible_v<ReferenceClosureMixture>);

} // namespace blackframe::renderer
