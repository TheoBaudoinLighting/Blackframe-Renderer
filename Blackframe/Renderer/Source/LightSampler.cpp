#include <Blackframe/Renderer/LightSampler.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Error invalid_light_sampler(const char* const message) {
    return {
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

[[nodiscard]] core::Error exhausted_light_sampler(const char* const message) {
    return {
        .code = core::StatusCode::resource_exhausted,
        .message = message,
    };
}

[[nodiscard]] core::Error inconsistent_light_sampler(const char* const message) {
    return {
        .code = core::StatusCode::internal_error,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::uint32_t> checked_light_count(const std::size_t light_count) {
    if (light_count == 0U) {
        return std::unexpected(
            invalid_light_sampler("Light sampling requires at least one registry slot."));
    }
    if (light_count > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(exhausted_light_sampler(
            "Light sampling exceeds the stable 32-bit registry-slot domain."));
    }
    if (light_count >= std::numeric_limits<std::size_t>::max()) {
        return std::unexpected(exhausted_light_sampler(
            "Light sampling cannot represent the terminal CDF boundary on this host."));
    }
    return static_cast<std::uint32_t>(light_count);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::vector<Scalar>> allocate_values(const std::size_t value_count) {
    auto values = std::vector<Scalar>{};
    if (value_count > values.max_size()) {
        return std::unexpected(
            exhausted_light_sampler("Light sampling exceeds host container limits."));
    }
    try {
        values.resize(value_count);
    } catch (const std::bad_alloc&) {
        return std::unexpected(exhausted_light_sampler("Light sampling exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            exhausted_light_sampler("Light sampling exceeds host container limits."));
    }
    return values;
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

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar> compensated_sum(const std::span<const Scalar> values) {
    auto sum = Scalar{0};
    auto correction = Scalar{0};
    for (const auto value : values) {
        if (!add_compensated(value, sum, correction)) {
            return std::unexpected(exhausted_light_sampler(
                "Light-selection power accumulation is not representable."));
        }
    }
    const auto total = sum + correction;
    if (!std::isfinite(total) || total < Scalar{0}) {
        return std::unexpected(
            exhausted_light_sampler("Light-selection power accumulation is not representable."));
    }
    return total == Scalar{0} ? Scalar{0} : total;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status build_power_cdf(const std::span<const Scalar> weights,
                                           const std::span<Scalar> boundaries) {
    if (weights.empty() || boundaries.size() != weights.size() + 1U) {
        return std::unexpected(
            inconsistent_light_sampler("Light-selection CDF storage is inconsistent."));
    }

    const auto total = compensated_sum(weights);
    if (!total) {
        return std::unexpected(total.error());
    }
    if (*total == Scalar{0}) {
        return std::unexpected(invalid_light_sampler(
            "Power-weighted light sampling requires positive spectral power."));
    }

    auto last_positive = weights.size();
    for (auto index = weights.size(); index > 0U; --index) {
        if (weights[index - 1U] > Scalar{0}) {
            last_positive = index - 1U;
            break;
        }
    }
    if (last_positive == weights.size()) {
        return std::unexpected(inconsistent_light_sampler(
            "Positive light-selection power has no supported registry slot."));
    }

    boundaries.front() = Scalar{0};
    auto prefix_sum = Scalar{0};
    auto prefix_correction = Scalar{0};
    auto previous = Scalar{0};
    for (auto index = std::size_t{0}; index < weights.size(); ++index) {
        const auto weight = weights[index];
        if (weight == Scalar{0}) {
            boundaries[index + 1U] = previous;
            continue;
        }
        if (!add_compensated(weight, prefix_sum, prefix_correction)) {
            return std::unexpected(
                exhausted_light_sampler("Light-selection power prefix is not representable."));
        }
        const auto prefix = prefix_sum + prefix_correction;
        if (!std::isfinite(prefix) || !(prefix > Scalar{0})) {
            return std::unexpected(
                exhausted_light_sampler("Light-selection power prefix is not representable."));
        }

        auto boundary = Scalar{1};
        if (index != last_positive) {
            boundary = prefix / *total;
            if (!std::isfinite(boundary) || !(boundary > previous) || !(boundary < Scalar{1})) {
                return std::unexpected(exhausted_light_sampler(
                    "A positive light-selection probability is not representable."));
            }
        } else {
            const auto nominal_probability = weight / *total;
            const auto candidate = previous + nominal_probability;
            if (!std::isfinite(nominal_probability) || !(nominal_probability > Scalar{0}) ||
                !std::isfinite(candidate) || !(candidate > previous)) {
                return std::unexpected(exhausted_light_sampler(
                    "The terminal positive light-selection probability is not representable."));
            }
        }
        boundaries[index + 1U] = boundary;
        previous = boundary;
    }

    if (boundaries.back() != Scalar{1}) {
        return std::unexpected(
            inconsistent_light_sampler("Light-selection CDF does not terminate at one."));
    }
    for (auto index = std::size_t{0}; index < weights.size(); ++index) {
        const auto probability = boundaries[index + 1U] - boundaries[index];
        if (!std::isfinite(probability) || probability < Scalar{0} ||
            ((weights[index] > Scalar{0}) != (probability > Scalar{0}))) {
            return std::unexpected(exhausted_light_sampler(
                "Light-selection CDF lost or created probability support."));
        }
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::vector<Scalar>>
power_weights(const std::span<const LightSpectrumT<Scalar>> spectral_powers) {
    auto global_peak = Scalar{0};
    for (const auto& power : spectral_powers) {
        for (const auto lane : power.values) {
            if (!std::isfinite(lane) || lane < Scalar{0}) {
                return std::unexpected(invalid_light_sampler(
                    "Power-weighted light sampling requires finite non-negative spectral lanes."));
            }
            global_peak = std::max(global_peak, lane);
        }
    }
    if (global_peak == Scalar{0}) {
        return std::unexpected(invalid_light_sampler(
            "Power-weighted light sampling requires positive spectral power."));
    }

    auto weights = allocate_values<Scalar>(spectral_powers.size());
    if (!weights) {
        return std::unexpected(weights.error());
    }
    for (auto index = std::size_t{0}; index < spectral_powers.size(); ++index) {
        auto lane_sum = Scalar{0};
        auto lane_correction = Scalar{0};
        auto has_positive_lane = false;
        for (const auto lane : spectral_powers[index].values) {
            const auto ratio = lane / global_peak;
            if (lane > Scalar{0}) {
                has_positive_lane = true;
                if (!(ratio > Scalar{0})) {
                    return std::unexpected(exhausted_light_sampler(
                        "A positive spectral power lane is not representable after scaling."));
                }
            }
            if (!add_compensated(ratio, lane_sum, lane_correction)) {
                return std::unexpected(exhausted_light_sampler(
                    "A spectral light-selection weight is not representable."));
            }
        }
        const auto weight = lane_sum + lane_correction;
        if (!std::isfinite(weight) || weight < Scalar{0} ||
            (has_positive_lane && !(weight > Scalar{0}))) {
            return std::unexpected(
                exhausted_light_sampler("A spectral light-selection weight is not representable."));
        }
        (*weights)[index] = weight == Scalar{0} ? Scalar{0} : weight;
    }
    return weights;
}

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr std::uint64_t maximum_uniform_light_count() noexcept {
    return std::uint64_t{1} << std::numeric_limits<Scalar>::digits;
}

} // namespace

template <SpectrumScalar Scalar>
LightSamplerT<Scalar>::LightSamplerT(const LightSamplingStrategy strategy,
                                     std::vector<Scalar> cdf_boundaries) noexcept
    : strategy_{strategy}, cdf_boundaries_{std::move(cdf_boundaries)} {}

template <SpectrumScalar Scalar>
LightSamplerT<Scalar>::LightSamplerT(LightSamplerT&& other) noexcept
    : strategy_{other.strategy_}, cdf_boundaries_{std::move(other.cdf_boundaries_)} {
    other.cdf_boundaries_.clear();
}

template <SpectrumScalar Scalar>
core::Result<LightSamplerT<Scalar>>
LightSamplerT<Scalar>::create_uniform(const std::size_t light_count) {
    const auto checked_count = checked_light_count<Scalar>(light_count);
    if (!checked_count) {
        return std::unexpected(checked_count.error());
    }
    if (static_cast<std::uint64_t>(light_count) > maximum_uniform_light_count<Scalar>()) {
        return std::unexpected(
            exhausted_light_sampler("Uniform light sampling exceeds the scalar precision limit."));
    }

    auto boundaries = allocate_values<Scalar>(light_count + 1U);
    if (!boundaries) {
        return std::unexpected(boundaries.error());
    }
    boundaries->front() = Scalar{0};
    const auto scalar_count = static_cast<Scalar>(light_count);
    auto previous = Scalar{0};
    for (auto index = std::size_t{1}; index < light_count; ++index) {
        const auto boundary = static_cast<Scalar>(index) / scalar_count;
        if (!std::isfinite(boundary) || !(boundary > previous) || !(boundary < Scalar{1})) {
            return std::unexpected(exhausted_light_sampler(
                "A uniform light-selection interval is not representable."));
        }
        (*boundaries)[index] = boundary;
        previous = boundary;
    }
    boundaries->back() = Scalar{1};
    return LightSamplerT{LightSamplingStrategy::uniform, std::move(*boundaries)};
}

template <SpectrumScalar Scalar>
core::Result<LightSamplerT<Scalar>> LightSamplerT<Scalar>::create_power_weighted(
    const std::span<const LightSpectrumT<Scalar>> spectral_powers) {
    const auto checked_count = checked_light_count<Scalar>(spectral_powers.size());
    if (!checked_count) {
        return std::unexpected(checked_count.error());
    }
    const auto weights = power_weights<Scalar>(spectral_powers);
    if (!weights) {
        return std::unexpected(weights.error());
    }
    auto boundaries = allocate_values<Scalar>(spectral_powers.size() + 1U);
    if (!boundaries) {
        return std::unexpected(boundaries.error());
    }
    const auto cdf_status = build_power_cdf<Scalar>(*weights, *boundaries);
    if (!cdf_status) {
        return std::unexpected(cdf_status.error());
    }
    return LightSamplerT{LightSamplingStrategy::power_weighted, std::move(*boundaries)};
}

template <SpectrumScalar Scalar>
core::Result<LightSelectionT<Scalar>>
LightSamplerT<Scalar>::sample(const Scalar canonical_sample) const {
    if (cdf_boundaries_.size() < 2U) {
        return std::unexpected(inconsistent_light_sampler(
            "A moved-from light sampler cannot classify canonical samples."));
    }
    if (!std::isfinite(canonical_sample) || canonical_sample < Scalar{0} ||
        !(canonical_sample < Scalar{1})) {
        return std::unexpected(
            invalid_light_sampler("Light sampling requires a finite canonical sample in [0, 1)."));
    }

    const auto boundary =
        std::upper_bound(cdf_boundaries_.begin() + 1, cdf_boundaries_.end(), canonical_sample);
    if (boundary == cdf_boundaries_.end()) {
        return std::unexpected(inconsistent_light_sampler(
            "Light-selection CDF could not classify a canonical sample."));
    }
    const auto offset = static_cast<std::size_t>(boundary - cdf_boundaries_.begin());
    const auto index = offset - 1U;
    const auto selected_probability = cdf_boundaries_[index + 1U] - cdf_boundaries_[index];
    if (!std::isfinite(selected_probability) || !(selected_probability > Scalar{0}) ||
        selected_probability > Scalar{1}) {
        return std::unexpected(inconsistent_light_sampler(
            "Light-selection CDF selected a slot without positive probability."));
    }
    return LightSelectionT<Scalar>{
        static_cast<std::uint32_t>(index),
        LightSelectionProbabilityT<Scalar>{selected_probability},
    };
}

template <SpectrumScalar Scalar>
core::Result<LightSelectionProbabilityT<Scalar>>
LightSamplerT<Scalar>::probability(const std::uint32_t light_index) const {
    if (light_index >= light_count()) {
        return std::unexpected(
            invalid_light_sampler("Light-selection probability index is out of range."));
    }
    const auto index = static_cast<std::size_t>(light_index);
    const auto value = cdf_boundaries_[index + 1U] - cdf_boundaries_[index];
    if (!std::isfinite(value) || value < Scalar{0} || value > Scalar{1}) {
        return std::unexpected(inconsistent_light_sampler(
            "Light-selection CDF contains an invalid discrete probability."));
    }
    return LightSelectionProbabilityT<Scalar>{value == Scalar{0} ? Scalar{0} : value};
}

template <SpectrumScalar Scalar> std::uint32_t LightSamplerT<Scalar>::light_count() const noexcept {
    if (cdf_boundaries_.size() < 2U) {
        return 0U;
    }
    return static_cast<std::uint32_t>(cdf_boundaries_.size() - 1U);
}

template class LightSamplerT<TransportScalar>;
template class LightSamplerT<ReferenceScalar>;

} // namespace blackframe::renderer
