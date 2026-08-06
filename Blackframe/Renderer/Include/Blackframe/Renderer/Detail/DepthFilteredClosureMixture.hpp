#pragma once

#include <Blackframe/Renderer/ClosureMixture.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>

namespace blackframe::renderer::detail {

[[nodiscard]] constexpr bool
valid_rough_dielectric_direction_mask(const ScatteringLobe directions) noexcept {
    return directions == ScatteringLobe::reflection || directions == ScatteringLobe::transmission ||
           directions == ScatteringDirectionMask;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SampledSpectrum<TransportSpectrumSampleCount, Scalar>>
rough_dielectric_eval_with_direction_mask(const RoughDielectricT<Scalar>& dielectric,
                                          const Vector3T<Scalar> outgoing_local,
                                          const Vector3T<Scalar> incoming_local,
                                          const TransportMode mode,
                                          const ScatteringLobe allowed_directions) {
    if (!valid_rough_dielectric_direction_mask(allowed_directions)) {
        return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
            "A filtered rough dielectric requires reflection, transmission, or both."));
    }
    const auto evaluated = dielectric.eval(outgoing_local, incoming_local, mode);
    if (!evaluated) {
        return std::unexpected(evaluated.error());
    }
    if (outgoing_local.z == Scalar{0} || incoming_local.z == Scalar{0}) {
        return *evaluated;
    }
    const auto direction = (outgoing_local.z > Scalar{0}) == (incoming_local.z > Scalar{0})
                               ? ScatteringLobe::reflection
                               : ScatteringLobe::transmission;
    return has_scattering_lobe(allowed_directions, direction)
               ? *evaluated
               : SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RoughDielectricProbabilityDensityT<Scalar>>
rough_dielectric_pdf_with_direction_mask(const RoughDielectricT<Scalar>& dielectric,
                                         const Vector3T<Scalar> outgoing_local,
                                         const Vector3T<Scalar> incoming_local,
                                         const TransportMode mode,
                                         const ScatteringLobe allowed_directions) {
    if (!valid_rough_dielectric_direction_mask(allowed_directions)) {
        return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
            "A filtered rough dielectric requires reflection, transmission, or both."));
    }
    const auto base = dielectric.pdf(outgoing_local, incoming_local, mode);
    if (!base) {
        return std::unexpected(base.error());
    }
    if (outgoing_local.z == Scalar{0} || incoming_local.z == Scalar{0} ||
        base->value == Scalar{0}) {
        return *base;
    }

    const auto reflection = (outgoing_local.z > Scalar{0}) == (incoming_local.z > Scalar{0});
    const auto direction = reflection ? ScatteringLobe::reflection : ScatteringLobe::transmission;
    if (!has_scattering_lobe(allowed_directions, direction)) {
        return RoughDielectricProbabilityDensityT<Scalar>{
            .value = Scalar{0},
            .measure = ContinuousBsdfProbabilityMeasure,
        };
    }
    if (allowed_directions == ScatteringDirectionMask) {
        return *base;
    }

    const auto interface = rough_dielectric_detail::interface_for(
        outgoing_local, dielectric.exterior_eta(), dielectric.interior_eta());
    const auto outgoing_face =
        rough_dielectric_detail::scaled_direction(outgoing_local, interface.face_sign);
    const auto incoming_face =
        rough_dielectric_detail::scaled_direction(incoming_local, interface.face_sign);
    auto branch_probability = Scalar{0};
    if (reflection) {
        const auto geometry =
            rough_dielectric_detail::reflection_geometry(outgoing_face, incoming_face);
        if (!geometry) {
            return std::unexpected(geometry.error());
        }
        const auto fresnel = dielectric_fresnel(geometry->outgoing_dot_microfacet,
                                                interface.incident_eta, interface.transmitted_eta);
        if (!fresnel) {
            return std::unexpected(fresnel.error());
        }
        branch_probability = *fresnel;
    } else {
        const auto geometry =
            rough_dielectric_detail::transmission_geometry(outgoing_face, incoming_face, interface);
        if (!geometry) {
            return std::unexpected(geometry.error());
        }
        if (!geometry->has_value()) {
            return RoughDielectricProbabilityDensityT<Scalar>{
                .value = Scalar{0},
                .measure = ContinuousBsdfProbabilityMeasure,
            };
        }
        const auto fresnel = dielectric_fresnel((**geometry).outgoing_dot_microfacet,
                                                interface.incident_eta, interface.transmitted_eta);
        if (!fresnel) {
            return std::unexpected(fresnel.error());
        }
        branch_probability = Scalar{1} - *fresnel;
    }
    if (!std::isfinite(branch_probability) || !(branch_probability > Scalar{0}) ||
        branch_probability > Scalar{1}) {
        return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
            "A filtered rough-dielectric branch probability is not representable."));
    }
    const auto conditional = rough_dielectric_detail::checked_product_ratio(
        std::array{base->value}, std::array{branch_probability},
        "A conditional rough-dielectric direction PDF is not representable.");
    if (!conditional) {
        return std::unexpected(conditional.error());
    }
    return RoughDielectricProbabilityDensityT<Scalar>{
        .value = *conditional,
        .measure = ContinuousBsdfProbabilityMeasure,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::optional<RoughDielectricSampleT<Scalar>>>
rough_dielectric_sample_with_direction_mask(const RoughDielectricT<Scalar>& dielectric,
                                            const Vector3T<Scalar> outgoing_local,
                                            const Scalar event_sample,
                                            const Point2T<Scalar> visible_normal_sample,
                                            const TransportMode mode,
                                            const ScatteringLobe allowed_directions) {
    if (!valid_rough_dielectric_direction_mask(allowed_directions)) {
        return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
            "A filtered rough dielectric requires reflection, transmission, or both."));
    }
    if (!rough_dielectric_detail::canonical_event_sample(event_sample)) {
        return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
            "Filtered rough-dielectric event selection requires a finite sample in [0, 1)."));
    }

    auto selected_event = event_sample;
    if (allowed_directions == ScatteringLobe::reflection) {
        selected_event = Scalar{0};
    } else if (allowed_directions == ScatteringLobe::transmission) {
        selected_event = std::nextafter(Scalar{1}, Scalar{0});
    }
    const auto sampled =
        dielectric.sample(outgoing_local, selected_event, visible_normal_sample, mode);
    if (!sampled) {
        return std::unexpected(sampled.error());
    }
    if (!sampled->has_value()) {
        return std::optional<RoughDielectricSampleT<Scalar>>{};
    }
    const auto sampled_direction = (**sampled).lobes & ScatteringDirectionMask;
    if (!has_scattering_lobe(allowed_directions, sampled_direction)) {
        // A forced transmission can encounter a total-internal-reflection microfacet. There is no
        // transmitted direction for that sample, and reflection is not a permitted fallback.
        return std::optional<RoughDielectricSampleT<Scalar>>{};
    }
    const auto probability = rough_dielectric_pdf_with_direction_mask(
        dielectric, outgoing_local, (**sampled).incoming_local, mode, allowed_directions);
    if (!probability) {
        return std::unexpected(probability.error());
    }
    auto result = **sampled;
    result.probability = *probability;
    return std::optional<RoughDielectricSampleT<Scalar>>{result};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SampledSpectrum<TransportSpectrumSampleCount, Scalar>>
eval_closure_record_with_direction_mask(const ClosureT<Scalar>& closure,
                                        const ScatteringLobe allowed_directions,
                                        const Vector3T<Scalar> outgoing_local,
                                        const Vector3T<Scalar> incoming_local,
                                        const TransportMode mode) {
    if (closure.kind != ClosureKind::rough_dielectric) {
        return closure_mixture_detail::eval_closure_record(closure, outgoing_local, incoming_local,
                                                           mode);
    }
    const auto model = closure_mixture_detail::rough_dielectric_from_record(closure);
    if (!model) {
        return std::unexpected(model.error());
    }
    return rough_dielectric_eval_with_direction_mask(*model, outgoing_local, incoming_local, mode,
                                                     allowed_directions);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ClosureProbabilityDensityT<Scalar>>
pdf_closure_record_with_direction_mask(const ClosureT<Scalar>& closure,
                                       const ScatteringLobe allowed_directions,
                                       const Vector3T<Scalar> outgoing_local,
                                       const Vector3T<Scalar> incoming_local,
                                       const TransportMode mode) {
    if (closure.kind != ClosureKind::rough_dielectric) {
        return closure_mixture_detail::pdf_closure_record(closure, outgoing_local, incoming_local,
                                                          mode);
    }
    const auto model = closure_mixture_detail::rough_dielectric_from_record(closure);
    if (!model) {
        return std::unexpected(model.error());
    }
    return rough_dielectric_pdf_with_direction_mask(*model, outgoing_local, incoming_local, mode,
                                                    allowed_directions);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::optional<closure_mixture_detail::ClosureDirectionSampleT<Scalar>>>
sample_closure_record_with_direction_mask(const ClosureT<Scalar>& closure,
                                          const ScatteringLobe allowed_directions,
                                          const Vector3T<Scalar> outgoing_local,
                                          const Scalar event_sample,
                                          const Point2T<Scalar> direction_sample,
                                          const TransportMode mode) {
    if (closure.kind != ClosureKind::rough_dielectric) {
        return closure_mixture_detail::sample_closure_record(closure, outgoing_local, event_sample,
                                                             direction_sample, mode);
    }
    const auto model = closure_mixture_detail::rough_dielectric_from_record(closure);
    if (!model) {
        return std::unexpected(model.error());
    }
    const auto sampled = rough_dielectric_sample_with_direction_mask(
        *model, outgoing_local, event_sample, direction_sample, mode, allowed_directions);
    if (!sampled) {
        return std::unexpected(sampled.error());
    }
    if (!sampled->has_value()) {
        return std::optional<closure_mixture_detail::ClosureDirectionSampleT<Scalar>>{};
    }
    return std::optional<closure_mixture_detail::ClosureDirectionSampleT<Scalar>>{
        closure_mixture_detail::ClosureDirectionSampleT<Scalar>{
            .incoming_local = (**sampled).incoming_local,
            .value = (**sampled).value,
            .probability = (**sampled).probability,
            .lobes = (**sampled).lobes,
            .eta_scale_multiplier = (**sampled).eta_scale_multiplier,
        }};
}

template <SpectrumScalar Scalar> class DepthFilteredClosureMixtureT final {
  public:
    using source_type = ClosureMixtureT<Scalar>;
    using closure_type = ClosureT<Scalar>;
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using probability_density_type = ClosureProbabilityDensityT<Scalar>;
    using sample_type = ClosureMixtureSampleT<Scalar>;

    [[nodiscard]] static core::Result<DepthFilteredClosureMixtureT>
    create(const source_type& source, const PathDepthLimits& limits,
           const PathDepthCounters& counters) {
        const auto depth = path_depth_total(counters);
        if (!depth) {
            return std::unexpected(depth.error());
        }
        if (const auto status = validate_path_depth_state(limits, counters, *depth); !status) {
            return std::unexpected(status.error());
        }

        auto result = DepthFilteredClosureMixtureT{source};
        const auto source_records = source.closure_set().closures();
        const auto source_probabilities = source.component_probabilities();
        for (auto source_index = std::size_t{}; source_index < source_records.size();
             ++source_index) {
            const auto& record = source_records[source_index];
            auto allowed_directions = ScatteringLobe::none;
            const auto consider = [&](const ScatteringLobe event,
                                      const ScatteringLobe direction) -> core::Status {
                const auto evaluated = evaluate_path_depth_event(limits, counters, event);
                if (!evaluated) {
                    return std::unexpected(evaluated.error());
                }
                if (evaluated->accepted()) {
                    allowed_directions = allowed_directions | direction;
                } else {
                    result.blocked_lobes_ = result.blocked_lobes_ | evaluated->blocked_limits;
                }
                return {};
            };

            if (record.kind == ClosureKind::rough_dielectric) {
                if (const auto status =
                        consider(ScatteringLobe::glossy | ScatteringLobe::reflection,
                                 ScatteringLobe::reflection);
                    !status) {
                    return std::unexpected(status.error());
                }
                if (const auto status =
                        consider(ScatteringLobe::glossy | ScatteringLobe::transmission,
                                 ScatteringLobe::transmission);
                    !status) {
                    return std::unexpected(status.error());
                }
            } else {
                if (!is_valid_surface_scattering_event(record.lobes)) {
                    return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                        "A depth-filtered closure record requires a concrete surface event."));
                }
                const auto direction = record.lobes & ScatteringDirectionMask;
                if (const auto status = consider(record.lobes, direction); !status) {
                    return std::unexpected(status.error());
                }
            }
            if (allowed_directions == ScatteringLobe::none) {
                continue;
            }
            const auto active_index = result.count_++;
            result.source_indices_[active_index] = static_cast<std::uint32_t>(source_index);
            result.allowed_directions_[active_index] = allowed_directions;
            result.probabilities_[active_index] = source_probabilities[source_index];
        }

        if (result.count_ == 0U) {
            return result;
        }
        if (result.count_ == source_records.size()) {
            std::copy(source.component_probabilities().begin(),
                      source.component_probabilities().end(), result.probabilities_.begin());
            std::copy(source.component_cdf().begin(), source.component_cdf().end(),
                      result.cdf_.begin());
            return result;
        }
        if (result.count_ == 1U) {
            result.probabilities_[0] = Scalar{1};
            result.cdf_[0] = Scalar{0};
            result.cdf_[1] = Scalar{1};
            return result;
        }

        const auto total = closure_mixture_detail::compensated_sum(
            std::span<const Scalar>{result.probabilities_.data(), result.count_});
        if (!total) {
            return std::unexpected(total.error());
        }
        if (!std::isfinite(*total) || !(*total > Scalar{0})) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Active depth-filtered closure probabilities have no representable mass."));
        }
        auto normalized = std::array<Scalar, MaximumClosureCount>{};
        for (auto index = std::size_t{}; index < result.count_; ++index) {
            normalized[index] = result.probabilities_[index] / *total;
        }
        const auto distribution = closure_mixture_detail::build_component_distribution(
            std::span<const Scalar>{normalized.data(), result.count_}, result.count_, false);
        if (!distribution) {
            return std::unexpected(distribution.error());
        }
        result.probabilities_ = distribution->probabilities;
        result.cdf_ = distribution->cdf;
        return result;
    }

    [[nodiscard]] constexpr bool source_empty() const noexcept {
        return source_->closure_set().empty();
    }
    [[nodiscard]] constexpr bool empty() const noexcept {
        return count_ == 0U;
    }
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return count_;
    }
    [[nodiscard]] constexpr ScatteringLobe blocked_lobes() const noexcept {
        return blocked_lobes_;
    }
    [[nodiscard]] constexpr const closure_type& active_closure(const std::size_t index) const {
        return source_->closure_set().closures()[source_indices_[index]];
    }
    [[nodiscard]] constexpr std::uint32_t source_closure_index(const std::size_t index) const {
        return source_indices_[index];
    }
    [[nodiscard]] constexpr ScatteringLobe allowed_directions(const std::size_t index) const {
        return allowed_directions_[index];
    }
    [[nodiscard]] constexpr std::span<const Scalar> component_probabilities() const noexcept {
        return {probabilities_.data(), count_};
    }
    [[nodiscard]] constexpr std::span<const Scalar> component_cdf() const noexcept {
        return empty() ? std::span<const Scalar>{}
                       : std::span<const Scalar>{cdf_.data(), count_ + 1U};
    }

    [[nodiscard]] core::Result<spectrum_type> eval(const Vector3T<Scalar> outgoing_local,
                                                   const Vector3T<Scalar> incoming_local,
                                                   const TransportMode mode) const {
        if (!is_known_transport_mode(mode) ||
            !closure_mixture_detail::unit_local_direction(outgoing_local) ||
            !closure_mixture_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Depth-filtered closure queries require finite unit directions and a supported "
                "transport mode."));
        }
        auto result = spectrum_type{};
        for (auto index = std::size_t{}; index < count_; ++index) {
            const auto component = eval_closure_record_with_direction_mask(
                active_closure(index), allowed_directions_[index], outgoing_local, incoming_local,
                mode);
            if (!component) {
                return std::unexpected(component.error());
            }
            for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
                const auto sum = result[lane] + (*component)[lane];
                if (!std::isfinite(sum)) {
                    return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                        "A depth-filtered closure value is not representable."));
                }
                result[lane] = sum;
            }
        }
        return result;
    }

    [[nodiscard]] core::Result<probability_density_type> pdf(const Vector3T<Scalar> outgoing_local,
                                                             const Vector3T<Scalar> incoming_local,
                                                             const TransportMode mode) const {
        if (!is_known_transport_mode(mode) ||
            !closure_mixture_detail::unit_local_direction(outgoing_local) ||
            !closure_mixture_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Depth-filtered closure queries require finite unit directions and a supported "
                "transport mode."));
        }
        if (empty()) {
            return probability_density_type{
                .value = Scalar{0},
                .measure = ContinuousBsdfProbabilityMeasure,
            };
        }
        auto conditional = std::array<probability_density_type, MaximumClosureCount>{};
        for (auto index = std::size_t{}; index < count_; ++index) {
            const auto component = pdf_closure_record_with_direction_mask(
                active_closure(index), allowed_directions_[index], outgoing_local, incoming_local,
                mode);
            if (!component) {
                return std::unexpected(component.error());
            }
            conditional[index] = *component;
        }
        const auto distribution = closure_mixture_detail::ComponentDistributionT<Scalar>{
            .probabilities = probabilities_,
            .cdf = cdf_,
            .count = count_,
        };
        return closure_mixture_detail::mix_probability_densities(
            distribution, std::span<const probability_density_type>{conditional.data(), count_});
    }

    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample(const Vector3T<Scalar> outgoing_local, const Scalar component_sample,
           const Point2T<Scalar> direction_sample, const TransportMode mode) const {
        if (!is_known_transport_mode(mode) ||
            !closure_mixture_detail::unit_local_direction(outgoing_local) ||
            !closure_mixture_detail::canonical_component_sample(component_sample) ||
            !closure_mixture_detail::canonical_direction_sample(direction_sample)) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "Depth-filtered closure sampling requires canonical samples, a finite unit "
                "direction, and a supported transport mode."));
        }
        if (empty()) {
            return std::optional<sample_type>{};
        }
        auto selected = count_;
        for (auto index = std::size_t{}; index < count_; ++index) {
            if (component_sample < cdf_[index + 1U]) {
                selected = index;
                break;
            }
        }
        if (selected == count_) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "The depth-filtered closure CDF did not select a component."));
        }
        const auto event_sample = (component_sample - cdf_[selected]) / probabilities_[selected];
        if (!std::isfinite(event_sample) || event_sample < Scalar{0} ||
            !(event_sample < Scalar{1})) {
            return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                "The depth-filtered closure sample cannot be remapped to [0, 1)."));
        }
        const auto sampled = sample_closure_record_with_direction_mask(
            active_closure(selected), allowed_directions_[selected], outgoing_local, event_sample,
            direction_sample, mode);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        if (!sampled->has_value()) {
            return std::optional<sample_type>{};
        }

        auto value = (**sampled).value;
        auto probability = (**sampled).probability;
        if (count_ > 1U) {
            if (probability.measure == ContinuousBsdfProbabilityMeasure) {
                const auto mixed_value = eval(outgoing_local, (**sampled).incoming_local, mode);
                if (!mixed_value) {
                    return std::unexpected(mixed_value.error());
                }
                const auto mixed_probability =
                    pdf(outgoing_local, (**sampled).incoming_local, mode);
                if (!mixed_probability) {
                    return std::unexpected(mixed_probability.error());
                }
                value = *mixed_value;
                probability = *mixed_probability;
            } else if (probability.measure == DeltaBsdfProbabilityMeasure) {
                auto closures = std::array<closure_type, MaximumClosureCount>{};
                for (auto index = std::size_t{}; index < count_; ++index) {
                    closures[index] = active_closure(index);
                }
                const auto atom = closure_mixture_detail::aggregate_delta_atom(
                    std::span<const closure_type>{closures.data(), count_},
                    component_probabilities(), outgoing_local, (**sampled).incoming_local, mode);
                if (!atom) {
                    return std::unexpected(atom.error());
                }
                value = atom->value;
                probability = atom->probability;
            } else {
                return std::unexpected(closure_mixture_detail::invalid_closure_mixture(
                    "A depth-filtered closure returned an unsupported probability measure."));
            }
        }
        return std::optional<sample_type>{sample_type{
            .selected_closure = source_indices_[selected],
            .lobes = (**sampled).lobes,
            .selection_probability =
                {
                    .value = probabilities_[selected],
                    .measure = ProbabilityMeasure::discrete,
                },
            .incoming_local = (**sampled).incoming_local,
            .value = value,
            .probability = probability,
            .eta_scale_multiplier = (**sampled).eta_scale_multiplier,
        }};
    }

  private:
    explicit constexpr DepthFilteredClosureMixtureT(const source_type& source) noexcept
        : source_{&source} {}

    const source_type* source_{};
    std::array<std::uint32_t, MaximumClosureCount> source_indices_{};
    std::array<ScatteringLobe, MaximumClosureCount> allowed_directions_{};
    std::array<Scalar, MaximumClosureCount> probabilities_{};
    std::array<Scalar, MaximumClosureCount + 1U> cdf_{};
    std::size_t count_{};
    ScatteringLobe blocked_lobes_{ScatteringLobe::none};
};

using DepthFilteredClosureMixture = DepthFilteredClosureMixtureT<TransportScalar>;
using ReferenceDepthFilteredClosureMixture = DepthFilteredClosureMixtureT<ReferenceScalar>;

static_assert(std::is_trivially_copyable_v<DepthFilteredClosureMixture>);
static_assert(std::is_trivially_copyable_v<ReferenceDepthFilteredClosureMixture>);

} // namespace blackframe::renderer::detail
