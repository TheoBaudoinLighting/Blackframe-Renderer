#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Error roulette_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_policy(const RussianRoulettePolicyT<Scalar>& policy) {
    switch (policy.mode()) {
    case RussianRouletteMode::disabled:
        if (policy.first_eligible_depth() != 0 ||
            policy.minimum_survival_probability() != Scalar{0} ||
            policy.maximum_survival_probability() != Scalar{0}) {
            return std::unexpected(roulette_error(
                core::StatusCode::invalid_argument,
                "A disabled Russian roulette policy must use its canonical zero state."));
        }
        return {};
    case RussianRouletteMode::enabled:
        if (policy.first_eligible_depth() == 0 ||
            !std::isfinite(policy.minimum_survival_probability()) ||
            !std::isfinite(policy.maximum_survival_probability()) ||
            !(policy.minimum_survival_probability() > Scalar{0}) ||
            !(policy.minimum_survival_probability() < Scalar{1}) ||
            policy.maximum_survival_probability() < policy.minimum_survival_probability() ||
            policy.maximum_survival_probability() > Scalar{1}) {
            return std::unexpected(roulette_error(
                core::StatusCode::invalid_argument,
                "An enabled Russian roulette policy has invalid depth or survival bounds."));
        }
        return {};
    }
    return std::unexpected(
        roulette_error(core::StatusCode::invalid_argument,
                       "Russian roulette policy mode contains an unsupported value."));
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RussianRouletteResultT<Scalar>>
evaluate(const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& throughput,
         const Scalar eta_scale, const std::uint32_t completed_depth, const Scalar canonical_sample,
         const RussianRoulettePolicyT<Scalar>& policy) {
    const auto policy_status = validate_policy(policy);
    if (!policy_status.has_value()) {
        return std::unexpected(policy_status.error());
    }
    if (!std::isfinite(eta_scale) || !(eta_scale > Scalar{0})) {
        return std::unexpected(
            roulette_error(core::StatusCode::invalid_argument,
                           "Russian roulette eta scale must be finite and strictly positive."));
    }
    if (!std::isfinite(canonical_sample) || canonical_sample < Scalar{0} ||
        !(canonical_sample < Scalar{1})) {
        return std::unexpected(
            roulette_error(core::StatusCode::invalid_argument,
                           "Russian roulette canonical sample must be finite and in [0, 1)."));
    }

    auto maximum_throughput = Scalar{0};
    for (const auto lane : throughput.values) {
        if (!std::isfinite(lane) || lane < Scalar{0}) {
            return std::unexpected(
                roulette_error(core::StatusCode::invalid_argument,
                               "Russian roulette throughput must be finite and non-negative."));
        }
        maximum_throughput = std::max(maximum_throughput, lane);
    }

    const auto not_evaluated = [&] {
        return RussianRouletteResultT<Scalar>{
            .throughput = throughput,
            .survival_probability =
                {
                    .value = Scalar{1},
                    .measure = ProbabilityMeasure::discrete,
                },
            .outcome = RussianRouletteOutcome::not_evaluated,
        };
    };
    if (!policy.is_enabled() || completed_depth < policy.first_eligible_depth()) {
        return not_evaluated();
    }
    if (maximum_throughput == Scalar{0}) {
        return std::unexpected(roulette_error(
            core::StatusCode::invalid_argument,
            "Russian roulette cannot evaluate an all-zero continuation throughput."));
    }

    const auto minimum = policy.minimum_survival_probability();
    const auto maximum = policy.maximum_survival_probability();
    auto survival_probability = Scalar{0};
    // These comparisons evaluate clamp(max(beta) * etaScale, minimum, maximum) without requiring
    // an overflowing or underflowing intermediate product.
    if (maximum_throughput >= maximum / eta_scale) {
        survival_probability = maximum;
    } else if (maximum_throughput <= minimum / eta_scale) {
        survival_probability = minimum;
    } else {
        survival_probability = std::clamp(maximum_throughput * eta_scale, minimum, maximum);
    }
    if (!std::isfinite(survival_probability) || !(survival_probability > Scalar{0}) ||
        survival_probability > Scalar{1}) {
        return std::unexpected(
            roulette_error(core::StatusCode::resource_exhausted,
                           "Russian roulette survival probability is not representable."));
    }

    const auto probability = RussianRouletteProbabilityDensityT<Scalar>{
        .value = survival_probability,
        .measure = ProbabilityMeasure::discrete,
    };
    if (!(canonical_sample < survival_probability)) {
        return RussianRouletteResultT<Scalar>{
            .throughput = throughput,
            .survival_probability = probability,
            .outcome = RussianRouletteOutcome::terminated,
        };
    }

    auto compensated = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        compensated[lane] = throughput[lane] / survival_probability;
        if (!std::isfinite(compensated[lane]) ||
            (throughput[lane] != Scalar{0} && compensated[lane] == Scalar{0})) {
            return std::unexpected(
                roulette_error(core::StatusCode::resource_exhausted,
                               "Russian roulette throughput compensation is not representable."));
        }
    }
    return RussianRouletteResultT<Scalar>{
        .throughput = compensated,
        .survival_probability = probability,
        .outcome = RussianRouletteOutcome::survived,
    };
}

} // namespace

core::Status validate_russian_roulette_policy(const RussianRoulettePolicy& policy) {
    return validate_policy(policy);
}

core::Status validate_russian_roulette_policy(const ReferenceRussianRoulettePolicy& policy) {
    return validate_policy(policy);
}

core::Result<RussianRouletteResult>
evaluate_russian_roulette(const TransportSpectrum& throughput, const TransportScalar eta_scale,
                          const std::uint32_t completed_depth,
                          const TransportScalar canonical_sample,
                          const RussianRoulettePolicy& policy) {
    return evaluate(throughput, eta_scale, completed_depth, canonical_sample, policy);
}

core::Result<ReferenceRussianRouletteResult>
evaluate_russian_roulette(const ReferenceSpectrum& throughput, const ReferenceScalar eta_scale,
                          const std::uint32_t completed_depth,
                          const ReferenceScalar canonical_sample,
                          const ReferenceRussianRoulettePolicy& policy) {
    return evaluate(throughput, eta_scale, completed_depth, canonical_sample, policy);
}

} // namespace blackframe::renderer
