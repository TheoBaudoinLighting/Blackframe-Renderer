#pragma once

#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

#if !defined(__CUDACC__)
#error "The CUDA transport helpers require the CUDA compiler."
#endif

namespace blackframe::xpu::cuda::transport_device {

inline constexpr auto SpectrumLaneCount = shared::HostDeviceSpectrumLaneCount;
inline constexpr auto MaximumUniformSelectionCount = std::uint32_t{1U} << 24U;
inline constexpr auto FloatEpsilon = 0x1p-23F;
inline constexpr auto InversePi = 0.31830988618379067154F;
inline constexpr auto QuarterPi = 0.78539816339744830962F;

enum class Status : std::uint8_t {
    success = 0U,
    invalid_argument = 1U,
    not_representable = 2U,
    outside_support = 3U,
};

enum class ProbabilityMeasure : std::uint8_t {
    discrete = 0U,
    solid_angle = 1U,
    area = 2U,
    distance = 3U,
    volume = 4U,
    wavelength = 5U,
};

enum class MisHeuristic : std::uint8_t {
    balance = 0U,
    power = 1U,
};

enum class RussianRouletteMode : std::uint8_t {
    disabled = 0U,
    enabled = 1U,
};

enum class RussianRouletteOutcome : std::uint8_t {
    not_evaluated = 0U,
    survived = 1U,
    terminated = 2U,
};

struct Vector3 final {
    float x;
    float y;
    float z;
};

struct ProbabilityDensity final {
    float value;
    ProbabilityMeasure measure;
    std::uint8_t reserved[3U];
};

struct ScalarResult final {
    float value;
    Status status;
    std::uint8_t reserved[3U];
};

struct SpectrumResult final {
    shared::TransportSpectrum value;
    Status status;
    std::uint8_t reserved[15U];
};

struct ProbabilityResult final {
    ProbabilityDensity value;
    Status status;
    std::uint8_t reserved[3U];
};

struct VectorResult final {
    Vector3 value;
    Status status;
    std::uint8_t reserved[3U];
};

struct LambertSampleResult final {
    shared::TransportSpectrum value;
    Vector3 incoming_local;
    ProbabilityDensity probability;
    Status status;
    std::uint8_t reserved[11U];
};

struct UniformSelectionResult final {
    std::uint32_t index;
    float lower_boundary;
    float upper_boundary;
    float interval_probability;
    Status status;
    std::uint8_t reserved[3U];
};

struct RussianRoulettePolicy final {
    RussianRouletteMode mode;
    std::uint8_t reserved[3U];
    std::uint32_t first_eligible_depth;
    float minimum_survival_probability;
    float maximum_survival_probability;
};

struct RussianRouletteResult final {
    shared::TransportSpectrum throughput;
    ProbabilityDensity survival_probability;
    RussianRouletteOutcome outcome;
    Status status;
    std::uint8_t reserved[6U];
};

[[nodiscard]] __host__ __device__ inline bool succeeded(const Status status) noexcept {
    return status == Status::success;
}

[[nodiscard]] __host__ __device__ inline bool
known_measure(const ProbabilityMeasure measure) noexcept {
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

[[nodiscard]] __host__ __device__ inline ProbabilityDensity
probability_density(const float value, const ProbabilityMeasure measure) noexcept {
    return ProbabilityDensity{
        .value = value,
        .measure = measure,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline bool
canonical_probability_density(const ProbabilityDensity probability) noexcept {
    return probability.reserved[0U] == 0U && probability.reserved[1U] == 0U &&
           probability.reserved[2U] == 0U;
}

[[nodiscard]] __host__ __device__ inline bool
spectrum_is_finite(const shared::TransportSpectrum& spectrum) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        if (!isfinite(spectrum.values[lane])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __host__ __device__ inline bool
spectrum_is_finite_nonnegative(const shared::TransportSpectrum& spectrum) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        if (!isfinite(spectrum.values[lane]) || spectrum.values[lane] < 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __host__ __device__ inline bool
spectrum_is_zero(const shared::TransportSpectrum& spectrum) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        if (spectrum.values[lane] != 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __host__ __device__ inline bool
spectrum_is_reflectance(const shared::TransportSpectrum& spectrum) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        if (!isfinite(spectrum.values[lane]) || spectrum.values[lane] < 0.0F ||
            spectrum.values[lane] > 1.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __host__ __device__ inline ScalarResult checked_product(const float left,
                                                                      const float right) noexcept {
    if (!isfinite(left) || left < 0.0F || !isfinite(right) || right < 0.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    const auto product = left * right;
    if (!isfinite(product) || product < 0.0F ||
        (left != 0.0F && right != 0.0F && product == 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = product, .status = Status::success, .reserved = {}};
}

template <std::size_t NumeratorCount, std::size_t DenominatorCount>
[[nodiscard]] __host__ __device__ inline ScalarResult
checked_product_quotient(const float (&numerators)[NumeratorCount],
                         const float (&denominators)[DenominatorCount]) noexcept {
    static_assert(NumeratorCount > 0U);
    static_assert(DenominatorCount > 0U);
    static_assert(NumeratorCount + DenominatorCount <= 16U);

    auto has_zero_numerator = false;
    for (const auto denominator : denominators) {
        if (!isfinite(denominator) || !(denominator > 0.0F)) {
            return ScalarResult{
                .value = 0.0F,
                .status = Status::invalid_argument,
                .reserved = {},
            };
        }
    }
    for (const auto numerator : numerators) {
        if (!isfinite(numerator) || numerator < 0.0F) {
            return ScalarResult{
                .value = 0.0F,
                .status = Status::invalid_argument,
                .reserved = {},
            };
        }
        has_zero_numerator = has_zero_numerator || numerator == 0.0F;
    }
    if (has_zero_numerator) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }

    auto significand = 1.0F;
    auto exponent = 0;
    for (const auto numerator : numerators) {
        auto factor_exponent = 0;
        significand *= frexpf(numerator, &factor_exponent);
        exponent += factor_exponent;
    }
    for (const auto denominator : denominators) {
        auto factor_exponent = 0;
        significand /= frexpf(denominator, &factor_exponent);
        exponent -= factor_exponent;
    }
    if (!isfinite(significand) || !(significand > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    auto normalization_exponent = 0;
    significand = frexpf(significand, &normalization_exponent);
    exponent += normalization_exponent;
    const auto result = scalbnf(significand, exponent);
    if (!isfinite(result) || !(result > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
checked_spectrum_product(const shared::TransportSpectrum& left,
                         const shared::TransportSpectrum& right) noexcept {
    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        const auto product = checked_product(left.values[lane], right.values[lane]);
        if (!succeeded(product.status)) {
            return SpectrumResult{.value = {}, .status = product.status, .reserved = {}};
        }
        result.values[lane] = product.value;
    }
    return SpectrumResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
checked_spectrum_scale(const shared::TransportSpectrum& spectrum, const float scale) noexcept {
    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        const auto product = checked_product(spectrum.values[lane], scale);
        if (!succeeded(product.status)) {
            return SpectrumResult{.value = {}, .status = product.status, .reserved = {}};
        }
        result.values[lane] = product.value;
    }
    return SpectrumResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
checked_accumulate(const shared::TransportSpectrum& accumulated,
                   const shared::TransportSpectrum& contribution) noexcept {
    if (!spectrum_is_finite_nonnegative(accumulated) ||
        !spectrum_is_finite_nonnegative(contribution)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    auto result = accumulated;
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        result.values[lane] += contribution.values[lane];
        if (!isfinite(result.values[lane]) || result.values[lane] < 0.0F) {
            return SpectrumResult{
                .value = {},
                .status = Status::not_representable,
                .reserved = {},
            };
        }
    }
    return SpectrumResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
checked_accumulate_product(const shared::TransportSpectrum& accumulated,
                           const shared::TransportSpectrum& first,
                           const shared::TransportSpectrum& second) noexcept {
    const auto contribution = checked_spectrum_product(first, second);
    if (!succeeded(contribution.status)) {
        return SpectrumResult{.value = {}, .status = contribution.status, .reserved = {}};
    }
    return checked_accumulate(accumulated, contribution.value);
}

[[nodiscard]] __host__ __device__ inline bool unit_vector(const Vector3 direction) noexcept {
    if (!isfinite(direction.x) || !isfinite(direction.y) || !isfinite(direction.z)) {
        return false;
    }
    const auto squared_length =
        fmaf(direction.x, direction.x, fmaf(direction.y, direction.y, direction.z * direction.z));
    constexpr auto tolerance = 128.0F * FloatEpsilon;
    return isfinite(squared_length) && fabsf(squared_length - 1.0F) <= tolerance;
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
lambert_eval(const shared::TransportSpectrum& reflectance, const Vector3 outgoing_local,
             const Vector3 incoming_local) noexcept {
    if (!spectrum_is_reflectance(reflectance) || !unit_vector(outgoing_local) ||
        !unit_vector(incoming_local)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    if (!(outgoing_local.z > 0.0F) || !(incoming_local.z > 0.0F)) {
        return SpectrumResult{.value = {}, .status = Status::success, .reserved = {}};
    }
    auto value = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        value.values[lane] = reflectance.values[lane] * InversePi;
        if (!isfinite(value.values[lane]) || value.values[lane] < 0.0F) {
            return SpectrumResult{
                .value = {},
                .status = Status::not_representable,
                .reserved = {},
            };
        }
    }
    return SpectrumResult{.value = value, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
lambert_pdf(const Vector3 outgoing_local, const Vector3 incoming_local) noexcept {
    if (!unit_vector(outgoing_local) || !unit_vector(incoming_local)) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    if (!(outgoing_local.z > 0.0F) || !(incoming_local.z > 0.0F)) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::success,
            .reserved = {},
        };
    }
    const auto density = checked_product(incoming_local.z, InversePi);
    if (!succeeded(density.status) || !(density.value > 0.0F)) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = succeeded(density.status) ? Status::not_representable : density.status,
            .reserved = {},
        };
    }
    return ProbabilityResult{
        .value = probability_density(density.value, ProbabilityMeasure::solid_angle),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline VectorResult
sample_cosine_hemisphere(const float canonical_u, const float canonical_v) noexcept {
    if (!isfinite(canonical_u) || canonical_u < 0.0F || !(canonical_u < 1.0F) ||
        !isfinite(canonical_v) || canonical_v < 0.0F || !(canonical_v < 1.0F)) {
        return VectorResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }

    const auto offset_x = 2.0F * canonical_u - 1.0F;
    const auto offset_y = 2.0F * canonical_v - 1.0F;
    if (offset_x == 0.0F && offset_y == 0.0F) {
        return VectorResult{
            .value = Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            .status = Status::success,
            .reserved = {},
        };
    }

    auto signed_radius = 0.0F;
    auto azimuth = 0.0F;
    if (fabsf(offset_x) > fabsf(offset_y)) {
        signed_radius = offset_x;
        azimuth = QuarterPi * (offset_y / offset_x);
    } else {
        signed_radius = offset_y;
        azimuth = 2.0F * QuarterPi - QuarterPi * (offset_x / offset_y);
    }
    const auto radius = fabsf(signed_radius);
    const auto direction = Vector3{
        .x = signed_radius * cosf(azimuth),
        .y = signed_radius * sinf(azimuth),
        .z = sqrtf((1.0F - radius) * (1.0F + radius)),
    };
    if (!unit_vector(direction)) {
        return VectorResult{.value = {}, .status = Status::not_representable, .reserved = {}};
    }
    return VectorResult{.value = direction, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline LambertSampleResult
sample_lambert(const shared::TransportSpectrum& reflectance, const Vector3 outgoing_local,
               const float canonical_u, const float canonical_v) noexcept {
    if (!spectrum_is_reflectance(reflectance) || !unit_vector(outgoing_local)) {
        return LambertSampleResult{
            .value = {},
            .incoming_local = {},
            .probability = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    const auto incoming = sample_cosine_hemisphere(canonical_u, canonical_v);
    if (!succeeded(incoming.status)) {
        return LambertSampleResult{
            .value = {},
            .incoming_local = {},
            .probability = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = incoming.status,
            .reserved = {},
        };
    }
    if (!(outgoing_local.z > 0.0F) || !(incoming.value.z > 0.0F)) {
        return LambertSampleResult{
            .value = {},
            .incoming_local = incoming.value,
            .probability = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::outside_support,
            .reserved = {},
        };
    }
    const auto value = lambert_eval(reflectance, outgoing_local, incoming.value);
    const auto probability = lambert_pdf(outgoing_local, incoming.value);
    if (!succeeded(value.status) || !succeeded(probability.status)) {
        return LambertSampleResult{
            .value = {},
            .incoming_local = incoming.value,
            .probability = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = !succeeded(value.status) ? value.status : probability.status,
            .reserved = {},
        };
    }
    return LambertSampleResult{
        .value = value.value,
        .incoming_local = incoming.value,
        .probability = probability.value,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline UniformSelectionResult
uniform_light_selection(const std::uint32_t count, const float canonical_sample) noexcept {
    if (count == 0U || !isfinite(canonical_sample) || canonical_sample < 0.0F ||
        !(canonical_sample < 1.0F)) {
        return UniformSelectionResult{
            .index = 0U,
            .lower_boundary = 0.0F,
            .upper_boundary = 0.0F,
            .interval_probability = 0.0F,
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    if (count > MaximumUniformSelectionCount) {
        return UniformSelectionResult{
            .index = 0U,
            .lower_boundary = 0.0F,
            .upper_boundary = 0.0F,
            .interval_probability = 0.0F,
            .status = Status::not_representable,
            .reserved = {},
        };
    }

    const auto scalar_count = static_cast<float>(count);
    auto first = std::uint32_t{1U};
    auto last = count + 1U;
    while (first < last) {
        const auto middle = first + (last - first) / 2U;
        const auto boundary = middle == count ? 1.0F : static_cast<float>(middle) / scalar_count;
        if (!(canonical_sample < boundary)) {
            first = middle + 1U;
        } else {
            last = middle;
        }
    }
    if (first == 0U || first > count) {
        return UniformSelectionResult{
            .index = 0U,
            .lower_boundary = 0.0F,
            .upper_boundary = 0.0F,
            .interval_probability = 0.0F,
            .status = Status::not_representable,
            .reserved = {},
        };
    }

    const auto index = first - 1U;
    const auto lower = index == 0U ? 0.0F : static_cast<float>(index) / scalar_count;
    const auto upper = first == count ? 1.0F : static_cast<float>(first) / scalar_count;
    const auto interval = upper - lower;
    if (!isfinite(lower) || !isfinite(upper) || !isfinite(interval) || lower < 0.0F ||
        !(lower <= canonical_sample) || !(canonical_sample < upper) || !(interval > 0.0F) ||
        interval > 1.0F) {
        return UniformSelectionResult{
            .index = 0U,
            .lower_boundary = 0.0F,
            .upper_boundary = 0.0F,
            .interval_probability = 0.0F,
            .status = Status::not_representable,
            .reserved = {},
        };
    }
    return UniformSelectionResult{
        .index = index,
        .lower_boundary = lower,
        .upper_boundary = upper,
        .interval_probability = interval,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
joint_light_pdf(const ProbabilityDensity selection_probability,
                const ProbabilityDensity conditional_probability) noexcept {
    if (!canonical_probability_density(selection_probability) ||
        !canonical_probability_density(conditional_probability) ||
        selection_probability.measure != ProbabilityMeasure::discrete ||
        !isfinite(selection_probability.value) || !(selection_probability.value > 0.0F) ||
        selection_probability.value > 1.0F ||
        conditional_probability.measure != ProbabilityMeasure::solid_angle ||
        !isfinite(conditional_probability.value) || conditional_probability.value < 0.0F) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    if (conditional_probability.value == 0.0F) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::success,
            .reserved = {},
        };
    }
    const float numerators[]{selection_probability.value, conditional_probability.value};
    const float denominators[]{1.0F};
    const auto product = checked_product_quotient(numerators, denominators);
    if (!succeeded(product.status)) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = product.status,
            .reserved = {},
        };
    }
    return ProbabilityResult{
        .value = probability_density(product.value, ProbabilityMeasure::solid_angle),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline ScalarResult
ratio_at_most_one(const float numerator, const float denominator) noexcept {
    if (!isfinite(numerator) || !(numerator > 0.0F) || !isfinite(denominator) ||
        !(denominator > 0.0F) || numerator > denominator) {
        return ScalarResult{
            .value = 0.0F,
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    auto numerator_exponent = 0;
    auto denominator_exponent = 0;
    const auto numerator_significand = frexpf(numerator, &numerator_exponent);
    const auto denominator_significand = frexpf(denominator, &denominator_exponent);
    const auto ratio = scalbnf(numerator_significand / denominator_significand,
                               numerator_exponent - denominator_exponent);
    if (!isfinite(ratio) || ratio < 0.0F || ratio > 1.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = ratio, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ScalarResult
squared_ratio_at_most_one(const float numerator, const float denominator) noexcept {
    if (!isfinite(numerator) || !(numerator > 0.0F) || !isfinite(denominator) ||
        !(denominator > 0.0F) || numerator > denominator) {
        return ScalarResult{
            .value = 0.0F,
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    auto numerator_exponent = 0;
    auto denominator_exponent = 0;
    const auto numerator_significand = frexpf(numerator, &numerator_exponent);
    const auto denominator_significand = frexpf(denominator, &denominator_exponent);
    const auto significand_ratio = numerator_significand / denominator_significand;
    const auto ratio = scalbnf(significand_ratio * significand_ratio,
                               2 * (numerator_exponent - denominator_exponent));
    if (!isfinite(ratio) || ratio < 0.0F || ratio > 1.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = ratio, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ScalarResult
mis_weight(const MisHeuristic heuristic, const ProbabilityDensity sampled,
           const ProbabilityDensity competing) noexcept {
    if (!canonical_probability_density(sampled) || !canonical_probability_density(competing) ||
        !known_measure(sampled.measure) || sampled.measure != competing.measure ||
        !isfinite(sampled.value) || !(sampled.value > 0.0F) || !isfinite(competing.value) ||
        competing.value < 0.0F ||
        (sampled.measure == ProbabilityMeasure::discrete &&
         (sampled.value > 1.0F || competing.value > 1.0F))) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    if (heuristic != MisHeuristic::balance && heuristic != MisHeuristic::power) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    if (competing.value == 0.0F) {
        return ScalarResult{.value = 1.0F, .status = Status::success, .reserved = {}};
    }

    if (sampled.value < competing.value) {
        const auto ratio = heuristic == MisHeuristic::balance
                               ? ratio_at_most_one(sampled.value, competing.value)
                               : squared_ratio_at_most_one(sampled.value, competing.value);
        if (!succeeded(ratio.status) || !(ratio.value > 0.0F)) {
            return ScalarResult{
                .value = 0.0F,
                .status = succeeded(ratio.status) ? Status::not_representable : ratio.status,
                .reserved = {},
            };
        }
        const auto weight = ratio.value / (1.0F + ratio.value);
        if (!isfinite(weight) || !(weight > 0.0F) || weight > 1.0F) {
            return ScalarResult{
                .value = 0.0F,
                .status = Status::not_representable,
                .reserved = {},
            };
        }
        return ScalarResult{.value = weight, .status = Status::success, .reserved = {}};
    }

    const auto ratio = heuristic == MisHeuristic::balance
                           ? ratio_at_most_one(competing.value, sampled.value)
                           : squared_ratio_at_most_one(competing.value, sampled.value);
    if (!succeeded(ratio.status)) {
        return ScalarResult{.value = 0.0F, .status = ratio.status, .reserved = {}};
    }
    const auto weight = 1.0F / (1.0F + ratio.value);
    if (!isfinite(weight) || !(weight > 0.0F) || weight > 1.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = weight, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline bool
valid_russian_roulette_policy(const RussianRoulettePolicy policy) noexcept {
    if (policy.reserved[0U] != 0U || policy.reserved[1U] != 0U || policy.reserved[2U] != 0U) {
        return false;
    }
    switch (policy.mode) {
    case RussianRouletteMode::disabled:
        return policy.first_eligible_depth == 0U && policy.minimum_survival_probability == 0.0F &&
               policy.maximum_survival_probability == 0.0F;
    case RussianRouletteMode::enabled:
        return policy.first_eligible_depth != 0U && isfinite(policy.minimum_survival_probability) &&
               isfinite(policy.maximum_survival_probability) &&
               policy.minimum_survival_probability > 0.0F &&
               policy.minimum_survival_probability < 1.0F &&
               policy.maximum_survival_probability >= policy.minimum_survival_probability &&
               policy.maximum_survival_probability <= 1.0F;
    }
    return false;
}

[[nodiscard]] __host__ __device__ inline RussianRouletteResult
evaluate_russian_roulette(const shared::TransportSpectrum& throughput, const float eta_scale,
                          const std::uint32_t completed_depth, const float canonical_sample,
                          const RussianRoulettePolicy policy) noexcept {
    const auto invalid_result = RussianRouletteResult{
        .throughput = {},
        .survival_probability = probability_density(0.0F, ProbabilityMeasure::discrete),
        .outcome = RussianRouletteOutcome::not_evaluated,
        .status = Status::invalid_argument,
        .reserved = {},
    };
    if (!valid_russian_roulette_policy(policy) || !spectrum_is_finite_nonnegative(throughput) ||
        !isfinite(eta_scale) || !(eta_scale > 0.0F) || !isfinite(canonical_sample) ||
        canonical_sample < 0.0F || !(canonical_sample < 1.0F)) {
        return invalid_result;
    }

    if (policy.mode == RussianRouletteMode::disabled ||
        completed_depth < policy.first_eligible_depth) {
        return RussianRouletteResult{
            .throughput = throughput,
            .survival_probability = probability_density(1.0F, ProbabilityMeasure::discrete),
            .outcome = RussianRouletteOutcome::not_evaluated,
            .status = Status::success,
            .reserved = {},
        };
    }

    auto maximum_throughput = 0.0F;
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        maximum_throughput = fmaxf(maximum_throughput, throughput.values[lane]);
    }
    if (maximum_throughput == 0.0F) {
        return invalid_result;
    }

    const auto minimum = policy.minimum_survival_probability;
    const auto maximum = policy.maximum_survival_probability;
    auto survival_probability = 0.0F;
    if (maximum_throughput >= maximum / eta_scale) {
        survival_probability = maximum;
    } else if (maximum_throughput <= minimum / eta_scale) {
        survival_probability = minimum;
    } else {
        const auto candidate = maximum_throughput * eta_scale;
        survival_probability =
            candidate < minimum ? minimum : (maximum < candidate ? maximum : candidate);
    }
    if (!isfinite(survival_probability) || !(survival_probability > 0.0F) ||
        survival_probability > 1.0F) {
        auto result = invalid_result;
        result.status = Status::not_representable;
        return result;
    }

    const auto probability =
        probability_density(survival_probability, ProbabilityMeasure::discrete);
    if (!(canonical_sample < survival_probability)) {
        return RussianRouletteResult{
            .throughput = throughput,
            .survival_probability = probability,
            .outcome = RussianRouletteOutcome::terminated,
            .status = Status::success,
            .reserved = {},
        };
    }

    auto compensated = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{0U}; lane < SpectrumLaneCount; ++lane) {
        compensated.values[lane] = throughput.values[lane] / survival_probability;
        if (!isfinite(compensated.values[lane]) ||
            (throughput.values[lane] != 0.0F && compensated.values[lane] == 0.0F)) {
            return RussianRouletteResult{
                .throughput = {},
                .survival_probability = probability,
                .outcome = RussianRouletteOutcome::not_evaluated,
                .status = Status::not_representable,
                .reserved = {},
            };
        }
    }
    return RussianRouletteResult{
        .throughput = compensated,
        .survival_probability = probability,
        .outcome = RussianRouletteOutcome::survived,
        .status = Status::success,
        .reserved = {},
    };
}

static_assert(SpectrumLaneCount == 4U);
static_assert(MaximumUniformSelectionCount == 16'777'216U);
static_assert(sizeof(Status) == 1U);
static_assert(sizeof(ProbabilityMeasure) == 1U);
static_assert(sizeof(MisHeuristic) == 1U);
static_assert(sizeof(RussianRouletteMode) == 1U);
static_assert(sizeof(RussianRouletteOutcome) == 1U);

#define BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(record)                                            \
    static_assert(std::is_standard_layout_v<record>);                                              \
    static_assert(std::is_trivially_copyable_v<record>);                                           \
    static_assert(std::is_trivially_destructible_v<record>)

BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(Vector3);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(ProbabilityDensity);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(ScalarResult);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(SpectrumResult);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(ProbabilityResult);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(VectorResult);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(LambertSampleResult);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(UniformSelectionResult);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(RussianRoulettePolicy);
BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD(RussianRouletteResult);

#undef BLACKFRAME_ASSERT_CUDA_TRANSPORT_RECORD

static_assert(sizeof(Vector3) == 12U);
static_assert(sizeof(ProbabilityDensity) == 8U);
static_assert(sizeof(ScalarResult) == 8U);
static_assert(sizeof(SpectrumResult) == 32U);
static_assert(sizeof(ProbabilityResult) == 12U);
static_assert(sizeof(VectorResult) == 16U);
static_assert(sizeof(LambertSampleResult) == 48U);
static_assert(sizeof(UniformSelectionResult) == 20U);
static_assert(sizeof(RussianRoulettePolicy) == 16U);
static_assert(sizeof(RussianRouletteResult) == 32U);

} // namespace blackframe::xpu::cuda::transport_device
