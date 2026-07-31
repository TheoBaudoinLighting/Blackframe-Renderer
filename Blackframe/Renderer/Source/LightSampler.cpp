#include <Blackframe/Renderer/LightSampler.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <numeric>
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

[[nodiscard]] core::Error unavailable_light_sampler(const char* const message) {
    return {
        .code = core::StatusCode::unavailable,
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

template <SpectrumScalar Scalar>
[[nodiscard]] bool finite_bounds(const Bounds3T<Scalar>& bounds) noexcept {
    const auto& minimum = bounds.minimum();
    const auto& maximum = bounds.maximum();
    return !bounds.is_empty() && std::isfinite(minimum.x) && std::isfinite(minimum.y) &&
           std::isfinite(minimum.z) && std::isfinite(maximum.x) && std::isfinite(maximum.y) &&
           std::isfinite(maximum.z);
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool canonical_unbounded_bounds(const Bounds3T<Scalar>& bounds) noexcept {
    constexpr auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto& minimum = bounds.minimum();
    const auto& maximum = bounds.maximum();
    return !bounds.is_empty() && minimum.x == -infinity && minimum.y == -infinity &&
           minimum.z == -infinity && maximum.x == infinity && maximum.y == infinity &&
           maximum.z == infinity;
}

template <SpectrumScalar Scalar>
[[nodiscard]] Point3T<Scalar> bounds_centroid(const Bounds3T<Scalar>& bounds) noexcept {
    return {
        .x = std::midpoint(bounds.minimum().x, bounds.maximum().x),
        .y = std::midpoint(bounds.minimum().y, bounds.maximum().y),
        .z = std::midpoint(bounds.minimum().z, bounds.maximum().z),
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Bounds3T<Scalar>> merged_bounds(const Bounds3T<Scalar>& left,
                                                           const Bounds3T<Scalar>& right) {
    if (canonical_unbounded_bounds(left) || canonical_unbounded_bounds(right)) {
        return Bounds3T<Scalar>::unbounded();
    }
    return Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{
            .x = std::min(left.minimum().x, right.minimum().x),
            .y = std::min(left.minimum().y, right.minimum().y),
            .z = std::min(left.minimum().z, right.minimum().z),
        },
        Point3T<Scalar>{
            .x = std::max(left.maximum().x, right.maximum().x),
            .y = std::max(left.maximum().y, right.maximum().y),
            .z = std::max(left.maximum().z, right.maximum().z),
        });
}

template <SpectrumScalar Scalar> struct ScaledPositive final {
    Scalar mantissa{};
    int exponent{};
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ScaledPositive<Scalar>> scaled_positive_difference(const Scalar upper,
                                                                              const Scalar lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower) || upper < lower) {
        return std::unexpected(inconsistent_light_sampler(
            "Light-tree scaled difference requires finite ordered values."));
    }
    auto upper_exponent = 0;
    auto lower_exponent = 0;
    static_cast<void>(std::frexp(std::abs(upper), &upper_exponent));
    static_cast<void>(std::frexp(std::abs(lower), &lower_exponent));
    const auto scale_exponent = std::max(upper_exponent, lower_exponent);
    const auto scaled_upper = std::scalbn(upper, -scale_exponent);
    const auto scaled_lower = std::scalbn(lower, -scale_exponent);
    const auto scaled_difference = scaled_upper - scaled_lower;
    if (!std::isfinite(scaled_difference) || scaled_difference < Scalar{0}) {
        return std::unexpected(exhausted_light_sampler(
            "Light-tree coordinate difference is not representable after scaling."));
    }
    if (scaled_difference == Scalar{0}) {
        return ScaledPositive<Scalar>{};
    }
    auto normalized_exponent = 0;
    const auto mantissa = std::frexp(scaled_difference, &normalized_exponent);
    if (!std::isfinite(mantissa) || !(mantissa > Scalar{0})) {
        return std::unexpected(exhausted_light_sampler(
            "Light-tree coordinate difference is not representable after scaling."));
    }
    return ScaledPositive<Scalar>{
        .mantissa = mantissa,
        .exponent = scale_exponent + normalized_exponent,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool scaled_less(const ScaledPositive<Scalar> left,
                               const ScaledPositive<Scalar> right) noexcept {
    if (left.mantissa == Scalar{0}) {
        return right.mantissa > Scalar{0};
    }
    if (right.mantissa == Scalar{0}) {
        return false;
    }
    return left.exponent != right.exponent ? left.exponent < right.exponent
                                           : left.mantissa < right.mantissa;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ScaledPositive<Scalar>>
scaled_distance(const Bounds3T<Scalar>& bounds, const Point3T<Scalar> reference_position) {
    const auto axis_distance = [](const Scalar value, const Scalar minimum,
                                  const Scalar maximum) -> core::Result<ScaledPositive<Scalar>> {
        if (value < minimum) {
            return scaled_positive_difference(minimum, value);
        }
        if (value > maximum) {
            return scaled_positive_difference(value, maximum);
        }
        return ScaledPositive<Scalar>{};
    };
    const auto distance_x =
        axis_distance(reference_position.x, bounds.minimum().x, bounds.maximum().x);
    const auto distance_y =
        axis_distance(reference_position.y, bounds.minimum().y, bounds.maximum().y);
    const auto distance_z =
        axis_distance(reference_position.z, bounds.minimum().z, bounds.maximum().z);
    if (!distance_x) {
        return std::unexpected(distance_x.error());
    }
    if (!distance_y) {
        return std::unexpected(distance_y.error());
    }
    if (!distance_z) {
        return std::unexpected(distance_z.error());
    }
    auto maximum = *distance_x;
    if (scaled_less(maximum, *distance_y)) {
        maximum = *distance_y;
    }
    if (scaled_less(maximum, *distance_z)) {
        maximum = *distance_z;
    }
    if (maximum.mantissa == Scalar{0}) {
        return maximum;
    }

    const auto scaled_component = [maximum](const ScaledPositive<Scalar> value) noexcept {
        if (value.mantissa == Scalar{0}) {
            return Scalar{0};
        }
        return std::scalbn(value.mantissa, value.exponent - maximum.exponent);
    };
    const auto hypotenuse = std::hypot(scaled_component(*distance_x), scaled_component(*distance_y),
                                       scaled_component(*distance_z));
    if (!std::isfinite(hypotenuse) || !(hypotenuse > Scalar{0})) {
        return std::unexpected(exhausted_light_sampler(
            "Light-tree query distance is not representable after scaling."));
    }
    auto normalized_exponent = 0;
    const auto mantissa = std::frexp(hypotenuse, &normalized_exponent);
    return ScaledPositive<Scalar>{
        .mantissa = mantissa,
        .exponent = maximum.exponent + normalized_exponent,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] Scalar scaled_logarithm(const ScaledPositive<Scalar> value) noexcept {
    return std::log(value.mantissa) + static_cast<Scalar>(value.exponent) * std::log(Scalar{2});
}

template <SpectrumScalar Scalar> struct RootExtent final {
    Scalar logarithm{};
    bool positive{};
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RootExtent<Scalar>> root_extent(const Bounds3T<Scalar>& bounds) {
    const auto extent_x = scaled_positive_difference(bounds.maximum().x, bounds.minimum().x);
    const auto extent_y = scaled_positive_difference(bounds.maximum().y, bounds.minimum().y);
    const auto extent_z = scaled_positive_difference(bounds.maximum().z, bounds.minimum().z);
    if (!extent_x) {
        return std::unexpected(extent_x.error());
    }
    if (!extent_y) {
        return std::unexpected(extent_y.error());
    }
    if (!extent_z) {
        return std::unexpected(extent_z.error());
    }
    auto maximum = *extent_x;
    if (scaled_less(maximum, *extent_y)) {
        maximum = *extent_y;
    }
    if (scaled_less(maximum, *extent_z)) {
        maximum = *extent_z;
    }
    if (maximum.mantissa == Scalar{0}) {
        return RootExtent<Scalar>{};
    }
    const auto logarithm = scaled_logarithm(maximum);
    if (!std::isfinite(logarithm)) {
        return std::unexpected(
            exhausted_light_sampler("Light-tree root extent logarithm is not representable."));
    }
    return RootExtent<Scalar>{.logarithm = logarithm, .positive = true};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar>
log_distance_attenuation(const Bounds3T<Scalar>& bounds, const Point3T<Scalar> reference_position,
                         const Scalar root_log_extent, const bool root_has_extent) {
    if (canonical_unbounded_bounds(bounds) || !root_has_extent) {
        return Scalar{0};
    }
    if (!finite_bounds(bounds) || !std::isfinite(root_log_extent)) {
        return std::unexpected(inconsistent_light_sampler(
            "Light-tree spatial attenuation received inconsistent finite bounds."));
    }
    const auto distance = scaled_distance(bounds, reference_position);
    if (!distance) {
        return std::unexpected(distance.error());
    }
    if (distance->mantissa == Scalar{0}) {
        return Scalar{0};
    }
    const auto log_normalized_distance = scaled_logarithm(*distance) - root_log_extent;
    if (!std::isfinite(log_normalized_distance)) {
        return std::unexpected(
            exhausted_light_sampler("Light-tree normalized log-distance is not representable."));
    }

    auto log_attenuation = Scalar{0};
    if (log_normalized_distance <= Scalar{0}) {
        const auto normalized_distance = std::exp(log_normalized_distance);
        log_attenuation = -std::log1p(normalized_distance * normalized_distance);
    } else {
        const auto inverse_squared = std::exp(Scalar{-2} * log_normalized_distance);
        log_attenuation = Scalar{-2} * log_normalized_distance - std::log1p(inverse_squared);
    }
    if (!std::isfinite(log_attenuation) || log_attenuation > Scalar{0}) {
        return std::unexpected(
            exhausted_light_sampler("Light-tree logarithmic attenuation is not representable."));
    }
    return log_attenuation;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar> multiply_probability(const Scalar accumulated,
                                                        const Scalar branch_probability) {
    const auto product = accumulated * branch_probability;
    if (!std::isfinite(product) || !(product > Scalar{0}) || product > Scalar{1}) {
        return std::unexpected(
            exhausted_light_sampler("Light-tree traversal lost positive selection probability."));
    }
    return product;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar> remap_canonical_sample(const Scalar canonical_sample,
                                                          const Scalar interval_begin,
                                                          const Scalar interval_probability) {
    const auto interval_end = interval_begin + interval_probability;
    const auto distance_from_begin = canonical_sample - interval_begin;
    const auto distance_to_end = interval_end - canonical_sample;
    if (!std::isfinite(interval_end) || distance_from_begin < Scalar{0} ||
        !(distance_to_end > Scalar{0})) {
        return std::unexpected(inconsistent_light_sampler(
            "Light-tree traversal could not remap its canonical sample."));
    }
    auto remapped = Scalar{0};
    if (distance_from_begin <= distance_to_end) {
        remapped = distance_from_begin / interval_probability;
    } else {
        remapped = Scalar{1} - distance_to_end / interval_probability;
    }
    if (!std::isfinite(remapped) || remapped < Scalar{0} || !(remapped < Scalar{1})) {
        return std::unexpected(inconsistent_light_sampler(
            "Light-tree canonical remap is not representable without clamping."));
    }
    return remapped;
}

} // namespace

template <SpectrumScalar Scalar>
LightSamplerT<Scalar>::LightSamplerT(const LightSamplingStrategy strategy,
                                     std::vector<Scalar> cdf_boundaries) noexcept
    : strategy_{strategy}, cdf_boundaries_{std::move(cdf_boundaries)} {}

template <SpectrumScalar Scalar>
LightSamplerT<Scalar>::LightSamplerT(
    std::vector<Scalar> cdf_boundaries, std::vector<TreeNode> tree_nodes,
    std::vector<std::uint32_t> light_to_node, const Scalar finite_root_log_extent,
    const bool finite_root_has_extent, const std::uint32_t tree_root,
    const std::uint32_t maximum_tree_depth, const std::uint32_t finite_light_count,
    const std::uint32_t unbounded_light_count) noexcept
    : strategy_{LightSamplingStrategy::spatial_tree}, cdf_boundaries_{std::move(cdf_boundaries)},
      tree_nodes_{std::move(tree_nodes)}, light_to_node_{std::move(light_to_node)},
      finite_root_log_extent_{finite_root_log_extent},
      finite_root_has_extent_{finite_root_has_extent}, tree_root_{tree_root},
      maximum_tree_depth_{maximum_tree_depth}, finite_light_count_{finite_light_count},
      unbounded_light_count_{unbounded_light_count} {}

template <SpectrumScalar Scalar>
LightSamplerT<Scalar>::LightSamplerT(LightSamplerT&& other) noexcept
    : strategy_{other.strategy_}, cdf_boundaries_{std::move(other.cdf_boundaries_)},
      tree_nodes_{std::move(other.tree_nodes_)}, light_to_node_{std::move(other.light_to_node_)},
      finite_root_log_extent_{other.finite_root_log_extent_},
      finite_root_has_extent_{other.finite_root_has_extent_}, tree_root_{other.tree_root_},
      maximum_tree_depth_{other.maximum_tree_depth_},
      finite_light_count_{other.finite_light_count_},
      unbounded_light_count_{other.unbounded_light_count_} {
    other.cdf_boundaries_.clear();
    other.tree_nodes_.clear();
    other.light_to_node_.clear();
    other.finite_root_log_extent_ = Scalar{0};
    other.finite_root_has_extent_ = false;
    other.tree_root_ = InvalidTreeIndex;
    other.maximum_tree_depth_ = 0U;
    other.finite_light_count_ = 0U;
    other.unbounded_light_count_ = 0U;
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
core::Result<LightSamplerT<Scalar>>
LightSamplerT<Scalar>::create_spatial_tree(const std::span<const LightTreeInputT<Scalar>> lights) {
    const auto checked_count = checked_light_count<Scalar>(lights.size());
    if (!checked_count) {
        return std::unexpected(checked_count.error());
    }

    struct BuildEntry final {
        Bounds3T<Scalar> bounds;
        Point3T<Scalar> centroid;
        Scalar power_weight;
        std::uint32_t light_index;
    };

    try {
        auto spectral_powers = std::vector<LightSpectrumT<Scalar>>{};
        auto unbounded_flags = std::vector<bool>{};
        if (lights.size() > spectral_powers.max_size() ||
            lights.size() > unbounded_flags.max_size()) {
            return std::unexpected(
                exhausted_light_sampler("Light-tree input exceeds host container limits."));
        }
        spectral_powers.reserve(lights.size());
        unbounded_flags.reserve(lights.size());

        auto finite_light_count = std::uint32_t{0};
        auto unbounded_light_count = std::uint32_t{0};
        for (const auto& light : lights) {
            const auto is_finite = finite_bounds(light.bounds);
            const auto is_unbounded = canonical_unbounded_bounds(light.bounds);
            if (!is_finite && !is_unbounded) {
                return std::unexpected(invalid_light_sampler(
                    "Light-tree bounds must be finite and non-empty or canonically unbounded."));
            }
            spectral_powers.push_back(light.spectral_power);
            unbounded_flags.push_back(is_unbounded);
            if (is_unbounded) {
                ++unbounded_light_count;
            } else {
                ++finite_light_count;
            }
        }

        const auto weights = power_weights<Scalar>(spectral_powers);
        if (!weights) {
            return std::unexpected(weights.error());
        }
        auto boundaries = allocate_values<Scalar>(lights.size() + 1U);
        if (!boundaries) {
            return std::unexpected(boundaries.error());
        }
        const auto cdf_status = build_power_cdf<Scalar>(*weights, *boundaries);
        if (!cdf_status) {
            return std::unexpected(cdf_status.error());
        }

        auto supported_light_count = std::size_t{0};
        for (const auto weight : *weights) {
            supported_light_count += weight > Scalar{0} ? 1U : 0U;
        }
        constexpr auto maximum_supported_lights =
            (static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1U) / 2U;
        if (supported_light_count == 0U) {
            return std::unexpected(inconsistent_light_sampler(
                "Positive light-tree power has no supported registry slot."));
        }
        if (static_cast<std::uint64_t>(supported_light_count) > maximum_supported_lights) {
            return std::unexpected(exhausted_light_sampler(
                "Light-tree node indices exceed the stable 32-bit domain."));
        }

        auto finite_entries = std::vector<BuildEntry>{};
        auto unbounded_entries = std::vector<BuildEntry>{};
        finite_entries.reserve(supported_light_count);
        unbounded_entries.reserve(supported_light_count);
        for (auto index = std::size_t{0}; index < lights.size(); ++index) {
            if ((*weights)[index] == Scalar{0}) {
                continue;
            }
            const auto light_index = static_cast<std::uint32_t>(index);
            if (unbounded_flags[index]) {
                unbounded_entries.push_back(BuildEntry{
                    .bounds = Bounds3T<Scalar>::unbounded(),
                    .centroid = Point3T<Scalar>{},
                    .power_weight = (*weights)[index],
                    .light_index = light_index,
                });
            } else {
                finite_entries.push_back(BuildEntry{
                    .bounds = lights[index].bounds,
                    .centroid = bounds_centroid(lights[index].bounds),
                    .power_weight = (*weights)[index],
                    .light_index = light_index,
                });
            }
        }
        std::sort(unbounded_entries.begin(), unbounded_entries.end(),
                  [](const BuildEntry& left, const BuildEntry& right) noexcept {
                      return left.light_index < right.light_index;
                  });

        const auto tree_node_capacity = supported_light_count * 2U - 1U;
        auto tree_nodes = std::vector<TreeNode>{};
        auto light_to_node = std::vector<std::uint32_t>{};
        if (tree_node_capacity > tree_nodes.max_size() ||
            lights.size() > light_to_node.max_size()) {
            return std::unexpected(
                exhausted_light_sampler("Light-tree storage exceeds host container limits."));
        }
        tree_nodes.reserve(tree_node_capacity);
        light_to_node.assign(lights.size(), InvalidTreeIndex);
        auto maximum_tree_depth = std::uint32_t{0};

        const auto build_subtree = [&tree_nodes, &light_to_node, &maximum_tree_depth](
                                       auto&& self, std::vector<BuildEntry>& entries,
                                       const std::size_t begin, const std::size_t end,
                                       const bool spatial, const std::uint32_t parent,
                                       const std::uint32_t depth) -> core::Result<std::uint32_t> {
            if (begin >= end) {
                return std::unexpected(inconsistent_light_sampler(
                    "Light-tree construction received an empty supported range."));
            }
            if (tree_nodes.size() >= std::numeric_limits<std::uint32_t>::max()) {
                return std::unexpected(exhausted_light_sampler(
                    "Light-tree node indices exceed the stable 32-bit domain."));
            }
            const auto node_index = static_cast<std::uint32_t>(tree_nodes.size());
            if (end - begin == 1U) {
                const auto& entry = entries[begin];
                tree_nodes.push_back(TreeNode{
                    .bounds = entry.bounds,
                    .power_weight = entry.power_weight,
                    .parent = parent,
                    .left_child = InvalidTreeIndex,
                    .right_child = InvalidTreeIndex,
                    .light_index = entry.light_index,
                    .kind = spatial ? TreeNodeKind::finite_leaf : TreeNodeKind::unbounded_leaf,
                });
                light_to_node[entry.light_index] = node_index;
                maximum_tree_depth = std::max(maximum_tree_depth, depth);
                return node_index;
            }

            if (spatial) {
                auto minimum = std::array{entries[begin].centroid.x, entries[begin].centroid.y,
                                          entries[begin].centroid.z};
                auto maximum = minimum;
                for (auto index = begin + 1U; index < end; ++index) {
                    const auto coordinates =
                        std::array{entries[index].centroid.x, entries[index].centroid.y,
                                   entries[index].centroid.z};
                    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
                        minimum[axis] = std::min(minimum[axis], coordinates[axis]);
                        maximum[axis] = std::max(maximum[axis], coordinates[axis]);
                    }
                }
                auto extents = std::array<ScaledPositive<Scalar>, 3>{};
                for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
                    const auto extent = scaled_positive_difference(maximum[axis], minimum[axis]);
                    if (!extent) {
                        return std::unexpected(extent.error());
                    }
                    extents[axis] = *extent;
                }
                auto split_axis = std::size_t{0};
                if (scaled_less(extents[split_axis], extents[1])) {
                    split_axis = 1U;
                }
                if (scaled_less(extents[split_axis], extents[2])) {
                    split_axis = 2U;
                }
                const auto coordinate = [split_axis](const BuildEntry& entry) noexcept {
                    if (split_axis == 0U) {
                        return entry.centroid.x;
                    }
                    if (split_axis == 1U) {
                        return entry.centroid.y;
                    }
                    return entry.centroid.z;
                };
                std::sort(entries.begin() + static_cast<std::ptrdiff_t>(begin),
                          entries.begin() + static_cast<std::ptrdiff_t>(end),
                          [&coordinate](const BuildEntry& left, const BuildEntry& right) noexcept {
                              const auto left_coordinate = coordinate(left);
                              const auto right_coordinate = coordinate(right);
                              if (left_coordinate != right_coordinate) {
                                  return left_coordinate < right_coordinate;
                              }
                              return left.light_index < right.light_index;
                          });
            }

            tree_nodes.push_back(TreeNode{
                .bounds = Bounds3T<Scalar>::empty(),
                .power_weight = Scalar{0},
                .parent = parent,
                .left_child = InvalidTreeIndex,
                .right_child = InvalidTreeIndex,
                .light_index = InvalidTreeIndex,
                .kind = TreeNodeKind::internal,
            });
            const auto middle = begin + (end - begin) / 2U;
            const auto left_child =
                self(self, entries, begin, middle, spatial, node_index, depth + 1U);
            if (!left_child) {
                return std::unexpected(left_child.error());
            }
            const auto right_child =
                self(self, entries, middle, end, spatial, node_index, depth + 1U);
            if (!right_child) {
                return std::unexpected(right_child.error());
            }
            const auto bounds =
                merged_bounds(tree_nodes[*left_child].bounds, tree_nodes[*right_child].bounds);
            if (!bounds) {
                return std::unexpected(bounds.error());
            }
            const auto power_weight =
                tree_nodes[*left_child].power_weight + tree_nodes[*right_child].power_weight;
            if (!std::isfinite(power_weight) ||
                !(power_weight > tree_nodes[*left_child].power_weight) ||
                !(power_weight > tree_nodes[*right_child].power_weight)) {
                return std::unexpected(
                    exhausted_light_sampler("Light-tree aggregation lost positive subtree power."));
            }
            tree_nodes[node_index] = TreeNode{
                .bounds = *bounds,
                .power_weight = power_weight,
                .parent = parent,
                .left_child = *left_child,
                .right_child = *right_child,
                .light_index = InvalidTreeIndex,
                .kind = TreeNodeKind::internal,
            };
            return node_index;
        };

        auto tree_root = InvalidTreeIndex;
        auto finite_root = InvalidTreeIndex;
        auto unbounded_root = InvalidTreeIndex;
        if (!finite_entries.empty() && !unbounded_entries.empty()) {
            tree_root = 0U;
            tree_nodes.push_back(TreeNode{
                .bounds = Bounds3T<Scalar>::empty(),
                .power_weight = Scalar{0},
                .parent = InvalidTreeIndex,
                .left_child = InvalidTreeIndex,
                .right_child = InvalidTreeIndex,
                .light_index = InvalidTreeIndex,
                .kind = TreeNodeKind::internal,
            });
            const auto built_finite_root = build_subtree(
                build_subtree, finite_entries, 0U, finite_entries.size(), true, tree_root, 2U);
            if (!built_finite_root) {
                return std::unexpected(built_finite_root.error());
            }
            const auto built_unbounded_root =
                build_subtree(build_subtree, unbounded_entries, 0U, unbounded_entries.size(), false,
                              tree_root, 2U);
            if (!built_unbounded_root) {
                return std::unexpected(built_unbounded_root.error());
            }
            finite_root = *built_finite_root;
            unbounded_root = *built_unbounded_root;
            const auto power_weight =
                tree_nodes[finite_root].power_weight + tree_nodes[unbounded_root].power_weight;
            if (!std::isfinite(power_weight) ||
                !(power_weight > tree_nodes[finite_root].power_weight) ||
                !(power_weight > tree_nodes[unbounded_root].power_weight)) {
                return std::unexpected(exhausted_light_sampler(
                    "Light-tree aggregation lost positive finite or unbounded power."));
            }
            tree_nodes[tree_root] = TreeNode{
                .bounds = Bounds3T<Scalar>::unbounded(),
                .power_weight = power_weight,
                .parent = InvalidTreeIndex,
                .left_child = finite_root,
                .right_child = unbounded_root,
                .light_index = InvalidTreeIndex,
                .kind = TreeNodeKind::internal,
            };
        } else if (!finite_entries.empty()) {
            const auto root = build_subtree(build_subtree, finite_entries, 0U,
                                            finite_entries.size(), true, InvalidTreeIndex, 1U);
            if (!root) {
                return std::unexpected(root.error());
            }
            tree_root = *root;
            finite_root = *root;
        } else {
            const auto root = build_subtree(build_subtree, unbounded_entries, 0U,
                                            unbounded_entries.size(), false, InvalidTreeIndex, 1U);
            if (!root) {
                return std::unexpected(root.error());
            }
            tree_root = *root;
            unbounded_root = *root;
        }

        auto finite_root_extent = RootExtent<Scalar>{};
        if (finite_root != InvalidTreeIndex) {
            const auto extent = root_extent(tree_nodes[finite_root].bounds);
            if (!extent) {
                return std::unexpected(extent.error());
            }
            finite_root_extent = *extent;
        }
        if (tree_nodes.size() != tree_node_capacity || tree_root == InvalidTreeIndex ||
            maximum_tree_depth == 0U) {
            return std::unexpected(
                inconsistent_light_sampler("Light-tree construction produced invalid storage."));
        }

        return LightSamplerT{
            std::move(*boundaries),       std::move(tree_nodes),       std::move(light_to_node),
            finite_root_extent.logarithm, finite_root_extent.positive, tree_root,
            maximum_tree_depth,           finite_light_count,          unbounded_light_count};
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            exhausted_light_sampler("Light-tree construction exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            exhausted_light_sampler("Light-tree construction exceeds host container limits."));
    }
}

template <SpectrumScalar Scalar>
core::Result<Scalar>
LightSamplerT<Scalar>::tree_left_probability(const Point3T<Scalar>& reference_position,
                                             const std::uint32_t node_index) const {
    if (node_index >= tree_nodes_.size()) {
        return std::unexpected(
            inconsistent_light_sampler("Light-tree traversal reached an invalid node index."));
    }
    const auto& node = tree_nodes_[node_index];
    if (node.kind != TreeNodeKind::internal || node.left_child >= tree_nodes_.size() ||
        node.right_child >= tree_nodes_.size()) {
        return std::unexpected(
            inconsistent_light_sampler("Light-tree traversal reached an invalid branch."));
    }

    const auto score = [this, reference_position](const TreeNode& child) -> core::Result<Scalar> {
        if (!std::isfinite(child.power_weight) || !(child.power_weight > Scalar{0})) {
            return std::unexpected(
                inconsistent_light_sampler("Light-tree branch contains invalid aggregate power."));
        }
        const auto log_attenuation = log_distance_attenuation(
            child.bounds, reference_position, finite_root_log_extent_, finite_root_has_extent_);
        if (!log_attenuation) {
            return std::unexpected(log_attenuation.error());
        }
        const auto log_score = std::log(child.power_weight) + *log_attenuation;
        if (!std::isfinite(log_score)) {
            return std::unexpected(
                exhausted_light_sampler("Light-tree log-importance is not representable."));
        }
        return log_score;
    };
    const auto left_score = score(tree_nodes_[node.left_child]);
    if (!left_score) {
        return std::unexpected(left_score.error());
    }
    const auto right_score = score(tree_nodes_[node.right_child]);
    if (!right_score) {
        return std::unexpected(right_score.error());
    }

    auto left_probability = Scalar{0};
    if (*left_score <= *right_score) {
        const auto ratio = std::exp(*left_score - *right_score);
        if (!std::isfinite(ratio) || !(ratio > Scalar{0})) {
            return std::unexpected(
                exhausted_light_sampler("Light-tree branch normalization lost positive support."));
        }
        left_probability = ratio / (Scalar{1} + ratio);
    } else {
        const auto ratio = std::exp(*right_score - *left_score);
        if (!std::isfinite(ratio) || !(ratio > Scalar{0})) {
            return std::unexpected(
                exhausted_light_sampler("Light-tree branch normalization lost positive support."));
        }
        const auto right_probability = ratio / (Scalar{1} + ratio);
        left_probability = Scalar{1} - right_probability;
    }
    const auto right_probability = Scalar{1} - left_probability;
    if (!std::isfinite(left_probability) || !(left_probability > Scalar{0}) ||
        !(left_probability < Scalar{1}) || !std::isfinite(right_probability) ||
        !(right_probability > Scalar{0}) || !(right_probability < Scalar{1})) {
        return std::unexpected(
            exhausted_light_sampler("Light-tree branch probability is not representable."));
    }
    return left_probability;
}

template <SpectrumScalar Scalar>
core::Result<LightSelectionT<Scalar>>
LightSamplerT<Scalar>::sample(const Scalar canonical_sample) const {
    if (cdf_boundaries_.size() < 2U) {
        return std::unexpected(inconsistent_light_sampler(
            "A moved-from light sampler cannot classify canonical samples."));
    }
    if (strategy_ == LightSamplingStrategy::spatial_tree) {
        return std::unexpected(unavailable_light_sampler(
            "Spatial light-tree sampling requires an explicit light sample context."));
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
core::Result<LightSelectionT<Scalar>>
LightSamplerT<Scalar>::sample(const LightSampleContextT<Scalar>& context,
                              const Scalar canonical_sample) const {
    if (strategy_ != LightSamplingStrategy::spatial_tree) {
        return sample(canonical_sample);
    }
    if (cdf_boundaries_.size() < 2U || tree_nodes_.empty() ||
        light_to_node_.size() + 1U != cdf_boundaries_.size() || tree_root_ >= tree_nodes_.size()) {
        return std::unexpected(inconsistent_light_sampler(
            "A moved-from or inconsistent light tree cannot classify canonical samples."));
    }
    if (!std::isfinite(canonical_sample) || canonical_sample < Scalar{0} ||
        !(canonical_sample < Scalar{1})) {
        return std::unexpected(invalid_light_sampler(
            "Light-tree sampling requires a finite canonical sample in [0, 1)."));
    }

    auto node_index = tree_root_;
    auto remapped_sample = canonical_sample;
    auto selection_probability = Scalar{1};
    while (tree_nodes_[node_index].kind == TreeNodeKind::internal) {
        const auto left_probability = tree_left_probability(context.position(), node_index);
        if (!left_probability) {
            return std::unexpected(left_probability.error());
        }
        const auto& node = tree_nodes_[node_index];
        if (remapped_sample < *left_probability) {
            const auto product = multiply_probability(selection_probability, *left_probability);
            if (!product) {
                return std::unexpected(product.error());
            }
            const auto remapped =
                remap_canonical_sample(remapped_sample, Scalar{0}, *left_probability);
            if (!remapped) {
                return std::unexpected(remapped.error());
            }
            selection_probability = *product;
            remapped_sample = *remapped;
            node_index = node.left_child;
        } else {
            const auto right_probability = Scalar{1} - *left_probability;
            const auto product = multiply_probability(selection_probability, right_probability);
            if (!product) {
                return std::unexpected(product.error());
            }
            const auto remapped =
                remap_canonical_sample(remapped_sample, *left_probability, right_probability);
            if (!remapped) {
                return std::unexpected(remapped.error());
            }
            selection_probability = *product;
            remapped_sample = *remapped;
            node_index = node.right_child;
        }
        if (node_index >= tree_nodes_.size()) {
            return std::unexpected(
                inconsistent_light_sampler("Light-tree traversal reached an invalid child index."));
        }
    }

    const auto light_index = tree_nodes_[node_index].light_index;
    if (light_index >= light_count() || !(selection_probability > Scalar{0}) ||
        selection_probability > Scalar{1}) {
        return std::unexpected(
            inconsistent_light_sampler("Light-tree traversal reached an invalid supported leaf."));
    }
    return LightSelectionT<Scalar>{
        light_index,
        LightSelectionProbabilityT<Scalar>{selection_probability},
    };
}

template <SpectrumScalar Scalar>
core::Result<LightSelectionProbabilityT<Scalar>>
LightSamplerT<Scalar>::probability(const std::uint32_t light_index) const {
    if (light_index >= light_count()) {
        return std::unexpected(
            invalid_light_sampler("Light-selection probability index is out of range."));
    }
    if (strategy_ == LightSamplingStrategy::spatial_tree) {
        return std::unexpected(unavailable_light_sampler(
            "Spatial light-tree probability requires an explicit light sample context."));
    }
    const auto index = static_cast<std::size_t>(light_index);
    const auto value = cdf_boundaries_[index + 1U] - cdf_boundaries_[index];
    if (!std::isfinite(value) || value < Scalar{0} || value > Scalar{1}) {
        return std::unexpected(inconsistent_light_sampler(
            "Light-selection CDF contains an invalid discrete probability."));
    }
    return LightSelectionProbabilityT<Scalar>{value == Scalar{0} ? Scalar{0} : value};
}

template <SpectrumScalar Scalar>
core::Result<LightSelectionProbabilityT<Scalar>>
LightSamplerT<Scalar>::probability(const LightSampleContextT<Scalar>& context,
                                   const std::uint32_t light_index) const {
    if (strategy_ != LightSamplingStrategy::spatial_tree) {
        return probability(light_index);
    }
    if (light_index >= light_count()) {
        return std::unexpected(
            invalid_light_sampler("Light-tree probability index is out of range."));
    }
    if (tree_nodes_.empty() || light_to_node_.size() != light_count() ||
        tree_root_ >= tree_nodes_.size()) {
        return std::unexpected(inconsistent_light_sampler(
            "A moved-from or inconsistent light tree has no queryable probability."));
    }

    const auto slot = static_cast<std::size_t>(light_index);
    const auto baseline_probability = cdf_boundaries_[slot + 1U] - cdf_boundaries_[slot];
    if (!std::isfinite(baseline_probability) || baseline_probability < Scalar{0}) {
        return std::unexpected(
            inconsistent_light_sampler("Light-tree support CDF contains an invalid probability."));
    }
    if (baseline_probability == Scalar{0}) {
        return LightSelectionProbabilityT<Scalar>{Scalar{0}};
    }

    struct PathStep final {
        std::uint32_t parent;
        bool chose_left;
    };
    constexpr auto maximum_path_length = std::size_t{64};
    auto reverse_path = std::array<PathStep, maximum_path_length>{};
    auto path_length = std::size_t{0};
    auto node_index = light_to_node_[slot];
    if (node_index >= tree_nodes_.size() || tree_nodes_[node_index].light_index != light_index) {
        return std::unexpected(
            inconsistent_light_sampler("Light-tree slot does not map to its supported leaf."));
    }
    while (node_index != tree_root_) {
        if (path_length == reverse_path.size()) {
            return std::unexpected(inconsistent_light_sampler(
                "Light-tree path exceeds its validated balanced depth."));
        }
        const auto parent = tree_nodes_[node_index].parent;
        if (parent >= tree_nodes_.size() || tree_nodes_[parent].kind != TreeNodeKind::internal) {
            return std::unexpected(
                inconsistent_light_sampler("Light-tree leaf has an invalid parent path."));
        }
        const auto chose_left = tree_nodes_[parent].left_child == node_index;
        if (!chose_left && tree_nodes_[parent].right_child != node_index) {
            return std::unexpected(
                inconsistent_light_sampler("Light-tree parent does not reference its child."));
        }
        reverse_path[path_length++] = PathStep{.parent = parent, .chose_left = chose_left};
        node_index = parent;
    }

    auto selection_probability = Scalar{1};
    for (auto offset = path_length; offset > 0U; --offset) {
        const auto& step = reverse_path[offset - 1U];
        const auto left_probability = tree_left_probability(context.position(), step.parent);
        if (!left_probability) {
            return std::unexpected(left_probability.error());
        }
        const auto branch_probability =
            step.chose_left ? *left_probability : Scalar{1} - *left_probability;
        const auto product = multiply_probability(selection_probability, branch_probability);
        if (!product) {
            return std::unexpected(product.error());
        }
        selection_probability = *product;
    }
    return LightSelectionProbabilityT<Scalar>{selection_probability};
}

template <SpectrumScalar Scalar> std::uint32_t LightSamplerT<Scalar>::light_count() const noexcept {
    if (cdf_boundaries_.size() < 2U) {
        return 0U;
    }
    return static_cast<std::uint32_t>(cdf_boundaries_.size() - 1U);
}

template <SpectrumScalar Scalar>
std::uint32_t LightSamplerT<Scalar>::tree_node_count() const noexcept {
    return static_cast<std::uint32_t>(tree_nodes_.size());
}

template class LightSamplerT<TransportScalar>;
template class LightSamplerT<ReferenceScalar>;

} // namespace blackframe::renderer
