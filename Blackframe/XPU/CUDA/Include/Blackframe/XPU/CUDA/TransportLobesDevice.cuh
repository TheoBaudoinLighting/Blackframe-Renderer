#pragma once

#include <Blackframe/XPU/CUDA/TransportDevice.cuh>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if !defined(__CUDACC__)
#error "The CUDA transport-lobe helpers require the CUDA compiler."
#endif

namespace blackframe::xpu::cuda::transport_device {

inline constexpr auto MaximumClosureCount = std::uint32_t{8U};
inline constexpr auto ClosureParameterScalarCount = std::uint32_t{10U};
inline constexpr auto Pi = 3.14159265358979323846F;
inline constexpr auto TwoPi = 6.28318530717958647692F;

enum class TransportMode : std::uint8_t {
    radiance = 0U,
    importance = 1U,
};

enum class ScatteringLobe : std::uint32_t {
    none = 0x00000000U,
    diffuse = 0x00000001U,
    glossy = 0x00000002U,
    specular = 0x00000004U,
    reflection = 0x00000008U,
    transmission = 0x00000010U,
    volume = 0x00000020U,
};

enum class ClosureKind : std::uint32_t {
    none = 0U,
    lambertian_reflection = 1U,
    rough_diffuse_reflection = 2U,
    rough_conductor_reflection = 3U,
    rough_dielectric = 4U,
    specular_reflection = 5U,
    specular_transmission = 6U,
};

[[nodiscard]] __host__ __device__ constexpr ScatteringLobe
operator|(const ScatteringLobe left, const ScatteringLobe right) noexcept {
    return static_cast<ScatteringLobe>(static_cast<std::uint32_t>(left) |
                                       static_cast<std::uint32_t>(right));
}

inline constexpr auto ScatteringDirectionMask =
    ScatteringLobe::reflection | ScatteringLobe::transmission;

[[nodiscard]] __host__ __device__ constexpr bool
has_scattering_lobe(const ScatteringLobe mask, const ScatteringLobe lobe) noexcept {
    return (static_cast<std::uint32_t>(mask) & static_cast<std::uint32_t>(lobe)) ==
           static_cast<std::uint32_t>(lobe);
}

[[nodiscard]] __host__ __device__ constexpr bool
valid_rough_dielectric_direction_mask(const ScatteringLobe directions) noexcept {
    return directions == ScatteringLobe::reflection || directions == ScatteringLobe::transmission ||
           directions == ScatteringDirectionMask;
}

struct alignas(16) ClosureRecord final {
    ClosureKind kind;
    ScatteringLobe lobes;
    float weight[SpectrumLaneCount];
    float parameters[ClosureParameterScalarCount];
};

// Probabilities and CDF boundaries are authoritative, host-projected sampler-grid values. They are
// never inferred from spectral closure weights on the device.
struct alignas(16) ClosureMixtureRecord final {
    std::uint32_t active_count;
    std::uint32_t reserved_header[3U];
    ClosureRecord closures[MaximumClosureCount];
    float probabilities[MaximumClosureCount];
    float cdf[MaximumClosureCount + 1U];
    std::uint32_t reserved_tail[3U];
};

struct LobeSampleResult final {
    shared::TransportSpectrum value;
    Vector3 incoming_local;
    ProbabilityDensity probability;
    ScatteringLobe lobes;
    float eta_scale_multiplier;
    Status status;
    std::uint8_t reserved[3U];
};

struct VisibleNormalSampleResult final {
    Vector3 microfacet_normal;
    ProbabilityDensity probability;
    Status status;
    std::uint8_t reserved[3U];
};

struct ClosureMixtureSampleResult final {
    shared::TransportSpectrum value;
    Vector3 incoming_local;
    ProbabilityDensity probability;
    ProbabilityDensity selection_probability;
    std::uint32_t selected_closure;
    ScatteringLobe lobes;
    float eta_scale_multiplier;
    Status status;
    std::uint8_t reserved[3U];
};

struct PathDepthLimitsRecord final {
    std::uint32_t diffuse;
    std::uint32_t glossy;
    std::uint32_t specular;
    std::uint32_t transmission;
    std::uint32_t volume;
};

struct PathDepthCountersRecord final {
    std::uint32_t diffuse;
    std::uint32_t glossy;
    std::uint32_t specular;
    std::uint32_t transmission;
    std::uint32_t volume;
};

// The source mixture is compacted in place. The sidecar preserves the source indices and the
// per-record direction mask needed when only one rough-dielectric branch remains available.
struct DepthFilteredClosureMixtureState final {
    std::uint32_t source_indices[MaximumClosureCount];
    ScatteringLobe allowed_directions[MaximumClosureCount];
    std::uint32_t source_count;
    ScatteringLobe blocked_lobes;
    Status status;
    std::uint8_t reserved[3U];
};

[[nodiscard]] __host__ __device__ constexpr bool
known_transport_mode(const TransportMode mode) noexcept {
    return mode == TransportMode::radiance || mode == TransportMode::importance;
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult
empty_lobe_sample(const Status status,
                  const ProbabilityMeasure measure = ProbabilityMeasure::solid_angle) noexcept {
    return LobeSampleResult{
        .value = {},
        .incoming_local = {},
        .probability = probability_density(0.0F, measure),
        .lobes = ScatteringLobe::none,
        .eta_scale_multiplier = 1.0F,
        .status = status,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline ClosureMixtureSampleResult
empty_mixture_sample(const Status status) noexcept {
    return ClosureMixtureSampleResult{
        .value = {},
        .incoming_local = {},
        .probability = probability_density(0.0F, ProbabilityMeasure::solid_angle),
        .selection_probability = probability_density(0.0F, ProbabilityMeasure::discrete),
        .selected_closure = 0U,
        .lobes = ScatteringLobe::none,
        .eta_scale_multiplier = 1.0F,
        .status = status,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline float dot(const Vector3 left,
                                                   const Vector3 right) noexcept {
    return fmaf(left.x, right.x, fmaf(left.y, right.y, left.z * right.z));
}

[[nodiscard]] __host__ __device__ inline Vector3 cross(const Vector3 left,
                                                       const Vector3 right) noexcept {
    return Vector3{
        .x = fmaf(left.y, right.z, -left.z * right.y),
        .y = fmaf(left.z, right.x, -left.x * right.z),
        .z = fmaf(left.x, right.y, -left.y * right.x),
    };
}

[[nodiscard]] __host__ __device__ inline float length(const Vector3 value) noexcept {
    return hypotf(hypotf(value.x, value.y), value.z);
}

[[nodiscard]] __host__ __device__ inline VectorResult
normalized_vector(const Vector3 value) noexcept {
    const auto magnitude = length(value);
    if (!isfinite(magnitude) || !(magnitude > 0.0F)) {
        return VectorResult{.value = {}, .status = Status::not_representable, .reserved = {}};
    }
    const auto result = Vector3{
        .x = value.x / magnitude,
        .y = value.y / magnitude,
        .z = value.z / magnitude,
    };
    if (!unit_vector(result)) {
        return VectorResult{.value = {}, .status = Status::not_representable, .reserved = {}};
    }
    return VectorResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline shared::TransportSpectrum
record_weight(const ClosureRecord& record) noexcept {
    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        result.values[lane] = record.weight[lane];
    }
    return result;
}

// Energy-preserving Oren--Nayar (EON), including the exact FON directional albedo and reciprocal
// analytical multiple-scattering compensation used by Renderer.
inline constexpr auto FonCoefficient = 0.5F - 2.0F * InversePi / 3.0F;
inline constexpr auto FonAverageCoefficient = 2.0F / 3.0F - 28.0F * InversePi / 15.0F;
inline constexpr auto FonAverageLossCoefficient = FonCoefficient - FonAverageCoefficient;

[[nodiscard]] __host__ __device__ inline ScalarResult
rough_diffuse_directional_loss_shape(const Vector3 direction) noexcept {
    const auto sine = hypotf(direction.x, direction.y);
    const auto cosine = direction.z;
    const auto angle = atan2f(sine, cosine);
    const auto stable_ratio = sine * cosine * (1.0F + sine + sine * sine) / (1.0F + sine);
    const auto g = sine * (angle - sine * cosine) + (2.0F / 3.0F) * (stable_ratio - sine);
    const auto loss_shape = FonCoefficient - g * InversePi;
    if (!isfinite(loss_shape) || loss_shape < 0.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = loss_shape, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
rough_diffuse_eval(const shared::TransportSpectrum& reflectance, const float roughness,
                   const Vector3 outgoing_local, const Vector3 incoming_local) noexcept {
    if (!spectrum_is_reflectance(reflectance) || !isfinite(roughness) || roughness < 0.0F ||
        roughness > 1.0F || !unit_vector(outgoing_local) || !unit_vector(incoming_local)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    if (!(outgoing_local.z > 0.0F) || !(incoming_local.z > 0.0F) || spectrum_is_zero(reflectance)) {
        return SpectrumResult{.value = {}, .status = Status::success, .reserved = {}};
    }
    if (roughness == 0.0F) {
        return lambert_eval(reflectance, outgoing_local, incoming_local);
    }

    const auto tangent_dot =
        fmaf(outgoing_local.x, incoming_local.x, outgoing_local.y * incoming_local.y);
    const auto s_over_t =
        tangent_dot > 0.0F ? tangent_dot / fmaxf(outgoing_local.z, incoming_local.z) : tangent_dot;
    const auto a = 1.0F / (1.0F + FonCoefficient * roughness);
    const auto single_scatter = InversePi * a * (1.0F + roughness * s_over_t);
    if (!isfinite(single_scatter) || single_scatter < 0.0F) {
        return SpectrumResult{.value = {}, .status = Status::not_representable, .reserved = {}};
    }

    const auto outgoing_loss = rough_diffuse_directional_loss_shape(outgoing_local);
    const auto incoming_loss = rough_diffuse_directional_loss_shape(incoming_local);
    if (!succeeded(outgoing_loss.status) || !succeeded(incoming_loss.status)) {
        return SpectrumResult{
            .value = {},
            .status =
                !succeeded(outgoing_loss.status) ? outgoing_loss.status : incoming_loss.status,
            .reserved = {},
        };
    }
    const auto average_albedo = a * (1.0F + FonAverageCoefficient * roughness);
    const auto loss_product_over_average =
        a * roughness * incoming_loss.value * outgoing_loss.value / FonAverageLossCoefficient;
    if (!isfinite(average_albedo) || !(average_albedo > 0.0F) || average_albedo > 1.0F ||
        !isfinite(loss_product_over_average) || loss_product_over_average < 0.0F) {
        return SpectrumResult{.value = {}, .status = Status::not_representable, .reserved = {}};
    }

    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        const auto rho = reflectance.values[lane];
        const auto denominator = (1.0F - rho) + rho * average_albedo;
        if (!isfinite(denominator) || !(denominator > 0.0F)) {
            return SpectrumResult{.value = {}, .status = Status::not_representable, .reserved = {}};
        }
        const auto multiple_scatter_albedo = rho * rho * average_albedo / denominator;
        const auto value =
            rho * single_scatter + multiple_scatter_albedo * InversePi * loss_product_over_average;
        if (!isfinite(value) || value < 0.0F) {
            return SpectrumResult{.value = {}, .status = Status::not_representable, .reserved = {}};
        }
        result.values[lane] = value;
    }
    return SpectrumResult{.value = result, .status = Status::success, .reserved = {}};
}

struct SmithTerms final {
    float normal;
    float tangent;
    float radius;
    float normal_over_radius;
    Status status;
    std::uint8_t reserved[3U];
};

[[nodiscard]] __host__ __device__ inline bool representable_ggx_width_pair(float alpha_x,
                                                                           float alpha_y) noexcept;

[[nodiscard]] __host__ __device__ inline SmithTerms ggx_smith_terms(float alpha_x, float alpha_y,
                                                                    Vector3 direction) noexcept;

[[nodiscard]] __host__ __device__ inline ScalarResult
ggx_smith_lambda(const float alpha_x, const float alpha_y, const Vector3 direction) noexcept {
    if (!representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(direction) ||
        !(direction.z > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    const auto terms = ggx_smith_terms(alpha_x, alpha_y, direction);
    if (!succeeded(terms.status)) {
        return ScalarResult{.value = 0.0F, .status = terms.status, .reserved = {}};
    }
    if (terms.tangent == 0.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }
    auto value = 0.0F;
    if (terms.tangent <= terms.normal) {
        const auto slope = terms.tangent / terms.normal;
        const auto root = hypotf(1.0F, slope);
        value = (slope * 0.5F) * (slope / (root + 1.0F));
    } else {
        value = (1.0F - terms.normal_over_radius) / (2.0F * terms.normal_over_radius);
    }
    if (!isfinite(value) || !(value > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = value, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ScalarResult
ggx_smith_g1(const float alpha_x, const float alpha_y, const Vector3 direction) noexcept {
    if (!representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(direction)) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    if (!(direction.z > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }
    const auto terms = ggx_smith_terms(alpha_x, alpha_y, direction);
    if (!succeeded(terms.status)) {
        return ScalarResult{.value = 0.0F, .status = terms.status, .reserved = {}};
    }
    const auto value = 2.0F * terms.normal_over_radius / (1.0F + terms.normal_over_radius);
    if (!isfinite(value) || !(value > 0.0F) || value > 1.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = value, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ScalarResult
ggx_smith_g2(const float alpha_x, const float alpha_y, const Vector3 outgoing_local,
             const Vector3 incoming_local) noexcept {
    if (!representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(outgoing_local) ||
        !unit_vector(incoming_local)) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    if (!(outgoing_local.z > 0.0F) || !(incoming_local.z > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }
    const auto outgoing_terms = ggx_smith_terms(alpha_x, alpha_y, outgoing_local);
    const auto incoming_terms = ggx_smith_terms(alpha_x, alpha_y, incoming_local);
    if (!succeeded(outgoing_terms.status) || !succeeded(incoming_terms.status)) {
        return ScalarResult{
            .value = 0.0F,
            .status =
                !succeeded(outgoing_terms.status) ? outgoing_terms.status : incoming_terms.status,
            .reserved = {},
        };
    }
    const auto smaller =
        fminf(outgoing_terms.normal_over_radius, incoming_terms.normal_over_radius);
    const auto larger = fmaxf(outgoing_terms.normal_over_radius, incoming_terms.normal_over_radius);
    const auto value = 2.0F * smaller / (1.0F + smaller / larger);
    if (!isfinite(value) || !(value > 0.0F) || value > 1.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = value, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline bool
representable_ggx_width_pair(const float alpha_x, const float alpha_y) noexcept {
    if (!isfinite(alpha_x) || !(alpha_x > 0.0F) || !isfinite(alpha_y) || !(alpha_y > 0.0F)) {
        return false;
    }
    const auto square_root_limit = sqrtf(FLT_MAX);
    const auto square_root_pi = sqrtf(Pi);
    const auto maximum_alpha = square_root_limit * square_root_pi;
    const auto minimum_alpha = 1.0F / maximum_alpha;
    if (alpha_x < minimum_alpha || alpha_x > maximum_alpha || alpha_y < minimum_alpha ||
        alpha_y > maximum_alpha) {
        return false;
    }
    if (alpha_x == alpha_y) {
        return true;
    }

    // Binary64 has ample exponent range for every binary32 alpha pair and evaluates the same
    // extrema as the host's long-double logarithmic representability proof.
    constexpr auto inverse_pi_wide = 0.31830988618379067154;
    const auto wide_alpha_x = static_cast<double>(alpha_x);
    const auto wide_alpha_y = static_cast<double>(alpha_y);
    const auto inverse_alpha_x = 1.0 / wide_alpha_x;
    const auto inverse_alpha_y = 1.0 / wide_alpha_y;
    const auto minimum_q = fmin(1.0, fmin(inverse_alpha_x, inverse_alpha_y));
    const auto maximum_q = fmax(1.0, fmax(inverse_alpha_x, inverse_alpha_y));
    const auto scale = inverse_pi_wide / (wide_alpha_x * wide_alpha_y);
    const auto minimum_q_squared = minimum_q * minimum_q;
    const auto maximum_q_squared = maximum_q * maximum_q;
    const auto maximum_density = scale / (minimum_q_squared * minimum_q_squared);
    const auto minimum_density = scale / (maximum_q_squared * maximum_q_squared);
    return isfinite(maximum_density) && isfinite(minimum_density) &&
           maximum_density <= static_cast<double>(FLT_MAX) &&
           minimum_density >= static_cast<double>(FLT_TRUE_MIN);
}

[[nodiscard]] __host__ __device__ inline SmithTerms
ggx_smith_terms(const float alpha_x, const float alpha_y, const Vector3 direction) noexcept {
    if (alpha_x == alpha_y) {
        const auto sine = hypotf(direction.x, direction.y);
        const auto inverse_alpha = 1.0F / alpha_x;
        const auto normal = alpha_x <= 1.0F ? direction.z : direction.z * inverse_alpha;
        const auto tangent = alpha_x <= 1.0F ? alpha_x * sine : sine;
        const auto radius = hypotf(normal, tangent);
        if (!isfinite(normal) || !isfinite(tangent) || !isfinite(radius) || !(normal > 0.0F) ||
            !(radius > 0.0F)) {
            return SmithTerms{.status = Status::not_representable, .reserved = {}};
        }
        const auto ratio = normal / radius;
        if (!isfinite(ratio) || !(ratio > 0.0F) || ratio > 1.0F) {
            return SmithTerms{.status = Status::not_representable, .reserved = {}};
        }
        return SmithTerms{.normal = normal,
                          .tangent = tangent,
                          .radius = radius,
                          .normal_over_radius = ratio,
                          .status = Status::success,
                          .reserved = {}};
    }

    const auto common_scale = fmaxf(1.0F, fmaxf(alpha_x, alpha_y));
    const auto normal = direction.z / common_scale;
    const auto tangent_x = (alpha_x / common_scale) * direction.x;
    const auto tangent_y = (alpha_y / common_scale) * direction.y;
    const auto tangent = hypotf(tangent_x, tangent_y);
    const auto radius = hypotf(normal, tangent);
    if (!isfinite(normal) || !isfinite(tangent_x) || !isfinite(tangent_y) || !isfinite(tangent) ||
        !isfinite(radius) || !(normal > 0.0F) || !(radius > 0.0F)) {
        return SmithTerms{.status = Status::not_representable, .reserved = {}};
    }
    const auto ratio = normal / radius;
    if (!isfinite(ratio) || !(ratio > 0.0F) || ratio > 1.0F) {
        return SmithTerms{.status = Status::not_representable, .reserved = {}};
    }
    return SmithTerms{.normal = normal,
                      .tangent = tangent,
                      .radius = radius,
                      .normal_over_radius = ratio,
                      .status = Status::success,
                      .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ScalarResult
ggx_normal_distribution(const float alpha_x, const float alpha_y,
                        const Vector3 microfacet_normal) noexcept {
    if (!representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(microfacet_normal)) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    if (!(microfacet_normal.z > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }

    auto denominator_root = 0.0F;
    auto density_root = 0.0F;
    if (alpha_x == alpha_y) {
        const auto radial = hypotf(microfacet_normal.x, microfacet_normal.y);
        if (alpha_x <= 1.0F) {
            denominator_root = hypotf(radial, alpha_x * microfacet_normal.z);
            density_root = (alpha_x / denominator_root) / denominator_root;
        } else {
            const auto inverse_alpha = 1.0F / alpha_x;
            denominator_root = hypotf(radial * inverse_alpha, microfacet_normal.z);
            density_root = (inverse_alpha / denominator_root) / denominator_root;
        }
        density_root *= sqrtf(InversePi);
    } else {
        denominator_root =
            hypotf(hypotf(microfacet_normal.x / alpha_x, microfacet_normal.y / alpha_y),
                   microfacet_normal.z);
        const auto normalization_root = sqrtf(InversePi) / (sqrtf(alpha_x) * sqrtf(alpha_y));
        density_root = (normalization_root / denominator_root) / denominator_root;
    }
    const auto value = density_root * density_root;
    if (!isfinite(denominator_root) || !(denominator_root > 0.0F) || !isfinite(value) ||
        !(value > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = value, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
rough_diffuse_pdf(const float roughness, const Vector3 outgoing_local,
                  const Vector3 incoming_local) noexcept {
    if (!isfinite(roughness) || roughness < 0.0F || roughness > 1.0F) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    return lambert_pdf(outgoing_local, incoming_local);
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult
sample_rough_diffuse(const shared::TransportSpectrum& reflectance, const float roughness,
                     const Vector3 outgoing_local, const float canonical_u,
                     const float canonical_v) noexcept {
    if (!spectrum_is_reflectance(reflectance) || !isfinite(roughness) || roughness < 0.0F ||
        roughness > 1.0F || !unit_vector(outgoing_local)) {
        return empty_lobe_sample(Status::invalid_argument);
    }
    const auto incoming = sample_cosine_hemisphere(canonical_u, canonical_v);
    if (!succeeded(incoming.status)) {
        return empty_lobe_sample(incoming.status);
    }
    if (!(outgoing_local.z > 0.0F) || !(incoming.value.z > 0.0F)) {
        auto result = empty_lobe_sample(Status::outside_support);
        result.incoming_local = incoming.value;
        return result;
    }
    const auto value = rough_diffuse_eval(reflectance, roughness, outgoing_local, incoming.value);
    const auto probability = rough_diffuse_pdf(roughness, outgoing_local, incoming.value);
    if (!succeeded(value.status) || !succeeded(probability.status)) {
        return empty_lobe_sample(!succeeded(value.status) ? value.status : probability.status);
    }
    return LobeSampleResult{
        .value = value.value,
        .incoming_local = incoming.value,
        .probability = probability.value,
        .lobes = ScatteringLobe::diffuse | ScatteringLobe::reflection,
        .eta_scale_multiplier = 1.0F,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline ScalarResult
checked_squared_magnitude_ratio(const float numerator_real, const float numerator_imaginary,
                                const float denominator_real,
                                const float denominator_imaginary) noexcept {
    const auto numerator = hypotf(numerator_real, numerator_imaginary);
    const auto denominator = hypotf(denominator_real, denominator_imaginary);
    if (!isfinite(numerator) || !isfinite(denominator) || !(denominator > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    const auto amplitude = numerator / denominator;
    const auto reflectance = amplitude * amplitude;
    if (!isfinite(reflectance) || reflectance < 0.0F || reflectance > 1.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = reflectance, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ScalarResult
dielectric_fresnel(const float incident_cosine, const float eta_incident,
                   const float eta_transmitted) noexcept {
    if (!isfinite(incident_cosine) || incident_cosine < 0.0F || incident_cosine > 1.0F ||
        !isfinite(eta_incident) || !(eta_incident > 0.0F) || !isfinite(eta_transmitted) ||
        !(eta_transmitted > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    if (eta_incident == eta_transmitted) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }
    if (incident_cosine == 0.0F) {
        return ScalarResult{.value = 1.0F, .status = Status::success, .reserved = {}};
    }

    const auto incident_is_denser = eta_incident > eta_transmitted;
    const auto normalized_eta =
        incident_is_denser ? eta_transmitted / eta_incident : eta_incident / eta_transmitted;
    if (incident_cosine == 1.0F) {
        const auto amplitude = (1.0F - normalized_eta) / (1.0F + normalized_eta);
        const auto reflectance = amplitude * amplitude;
        if (!isfinite(reflectance) || !(reflectance > 0.0F) || reflectance > 1.0F) {
            return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
        }
        return ScalarResult{.value = reflectance, .status = Status::success, .reserved = {}};
    }

    const auto incident_sine = sqrtf((1.0F - incident_cosine) * (1.0F + incident_cosine));
    auto transmitted_sine = 0.0F;
    if (incident_is_denser) {
        if (normalized_eta == 0.0F || incident_sine >= normalized_eta) {
            return ScalarResult{.value = 1.0F, .status = Status::success, .reserved = {}};
        }
        transmitted_sine = incident_sine / normalized_eta;
    } else {
        transmitted_sine = normalized_eta * incident_sine;
    }
    if (!(transmitted_sine < 1.0F)) {
        return ScalarResult{.value = 1.0F, .status = Status::success, .reserved = {}};
    }
    const auto transmitted_cosine = sqrtf((1.0F - transmitted_sine) * (1.0F + transmitted_sine));
    const auto first_eta_cosine = normalized_eta * transmitted_cosine;
    const auto first_amplitude =
        (incident_cosine - first_eta_cosine) / (incident_cosine + first_eta_cosine);
    const auto second_eta_cosine = normalized_eta * incident_cosine;
    const auto second_amplitude =
        (second_eta_cosine - transmitted_cosine) / (second_eta_cosine + transmitted_cosine);
    const auto reflectance =
        0.5F * fmaf(first_amplitude, first_amplitude, second_amplitude * second_amplitude);
    if (!isfinite(reflectance) || !(reflectance > 0.0F) || reflectance > 1.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = reflectance, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ScalarResult
conductor_fresnel_lane(const float incident_cosine, const float relative_eta,
                       const float relative_k) noexcept {
    if (relative_eta == 1.0F && relative_k == 0.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }
    if (incident_cosine == 0.0F) {
        return ScalarResult{.value = 1.0F, .status = Status::success, .reserved = {}};
    }
    const auto scale = fmaxf(1.0F, fmaxf(relative_eta, relative_k));
    const auto normalized_incident_eta = 1.0F / scale;
    const auto normalized_eta = relative_eta / scale;
    const auto normalized_k = relative_k / scale;
    if (incident_cosine == 1.0F) {
        const auto reflectance =
            checked_squared_magnitude_ratio(normalized_eta - normalized_incident_eta, normalized_k,
                                            normalized_eta + normalized_incident_eta, normalized_k);
        if (!succeeded(reflectance.status) || !(reflectance.value > 0.0F)) {
            return ScalarResult{.value = 0.0F,
                                .status = succeeded(reflectance.status) ? Status::not_representable
                                                                        : reflectance.status,
                                .reserved = {}};
        }
        return reflectance;
    }

    const auto incident_sine_squared = (1.0F - incident_cosine) * (1.0F + incident_cosine);
    const auto scaled_incident_sine_squared =
        normalized_incident_eta * normalized_incident_eta * incident_sine_squared;
    const auto x = fmaf(normalized_eta, normalized_eta,
                        -fmaf(normalized_k, normalized_k, scaled_incident_sine_squared));
    const auto y = 2.0F * normalized_eta * normalized_k;
    const auto magnitude = hypotf(x, y);
    auto root_real = 0.0F;
    auto root_imaginary = 0.0F;
    if (x >= 0.0F) {
        root_real = sqrtf(0.5F * (magnitude + x));
        root_imaginary = root_real == 0.0F ? 0.0F : y / (2.0F * root_real);
    } else {
        root_imaginary = sqrtf(0.5F * (magnitude - x));
        root_real = root_imaginary == 0.0F ? 0.0F : y / (2.0F * root_imaginary);
    }
    const auto scaled_incident_cosine = normalized_incident_eta * incident_cosine;
    const auto perpendicular =
        checked_squared_magnitude_ratio(scaled_incident_cosine - root_real, -root_imaginary,
                                        scaled_incident_cosine + root_real, root_imaginary);
    if (!succeeded(perpendicular.status)) {
        return perpendicular;
    }
    const auto squared_eta_real =
        fmaf(normalized_eta, normalized_eta, -(normalized_k * normalized_k));
    const auto squared_eta_imaginary = y;
    const auto scaled_root_real = normalized_incident_eta * root_real;
    const auto scaled_root_imaginary = normalized_incident_eta * root_imaginary;
    const auto parallel_base_real = incident_cosine * squared_eta_real;
    const auto parallel_base_imaginary = incident_cosine * squared_eta_imaginary;
    const auto parallel = checked_squared_magnitude_ratio(
        parallel_base_real - scaled_root_real, parallel_base_imaginary - scaled_root_imaginary,
        parallel_base_real + scaled_root_real, parallel_base_imaginary + scaled_root_imaginary);
    if (!succeeded(parallel.status)) {
        return parallel;
    }
    const auto reflectance = 0.5F * (perpendicular.value + parallel.value);
    if (!isfinite(reflectance) || !(reflectance > 0.0F) || reflectance > 1.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = reflectance, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
conductor_fresnel(const float incident_cosine, const shared::TransportSpectrum& relative_eta,
                  const shared::TransportSpectrum& relative_k) noexcept {
    if (!isfinite(incident_cosine) || incident_cosine < 0.0F || incident_cosine > 1.0F) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        if (!isfinite(relative_eta.values[lane]) || !(relative_eta.values[lane] > 0.0F) ||
            !isfinite(relative_k.values[lane]) || relative_k.values[lane] < 0.0F) {
            return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
        }
    }
    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        const auto evaluated = conductor_fresnel_lane(incident_cosine, relative_eta.values[lane],
                                                      relative_k.values[lane]);
        if (!succeeded(evaluated.status)) {
            return SpectrumResult{.value = {}, .status = evaluated.status, .reserved = {}};
        }
        result.values[lane] = evaluated.value;
    }
    return SpectrumResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
ggx_visible_normal_pdf(const float alpha_x, const float alpha_y, const Vector3 outgoing_local,
                       const Vector3 microfacet_normal) noexcept {
    const auto zero = probability_density(0.0F, ProbabilityMeasure::solid_angle);
    if (!representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(outgoing_local) ||
        !unit_vector(microfacet_normal)) {
        return ProbabilityResult{.value = zero, .status = Status::invalid_argument, .reserved = {}};
    }
    const auto visible_dot = dot(outgoing_local, microfacet_normal);
    if (!(outgoing_local.z > 0.0F) || !(microfacet_normal.z > 0.0F) || !(visible_dot > 0.0F)) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    const auto distribution = ggx_normal_distribution(alpha_x, alpha_y, microfacet_normal);
    if (!succeeded(distribution.status)) {
        return ProbabilityResult{.value = zero, .status = distribution.status, .reserved = {}};
    }
    const auto physical_radius =
        alpha_x == alpha_y
            ? hypotf(outgoing_local.z, alpha_x * hypotf(outgoing_local.x, outgoing_local.y))
            : hypotf(hypotf(outgoing_local.z, alpha_x * outgoing_local.x),
                     alpha_y * outgoing_local.y);
    const auto physical_denominator = outgoing_local.z + physical_radius;
    const auto value = distribution.value * (2.0F * visible_dot / physical_denominator);
    if (!isfinite(physical_denominator) || !(physical_denominator > 0.0F) || !isfinite(value) ||
        !(value > 0.0F)) {
        return ProbabilityResult{
            .value = zero, .status = Status::not_representable, .reserved = {}};
    }
    return ProbabilityResult{
        .value = probability_density(value, ProbabilityMeasure::solid_angle),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline ScalarResult
ggx_nonnegative_roundoff(const float value, const float scale) noexcept {
    if (value >= 0.0F) {
        return ScalarResult{.value = value, .status = Status::success, .reserved = {}};
    }
    const auto bound = FloatEpsilon * 16.0F * fmaxf(1.0F, fabsf(scale));
    return -value <= bound
               ? ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}}
               : ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline VisibleNormalSampleResult
ggx_sample_visible_normal(const float alpha_x, const float alpha_y, const Vector3 outgoing_local,
                          const float canonical_u, const float canonical_v) noexcept {
    const auto zero = probability_density(0.0F, ProbabilityMeasure::solid_angle);
    if (!representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(outgoing_local) ||
        !isfinite(canonical_u) || canonical_u < 0.0F || !(canonical_u < 1.0F) ||
        !isfinite(canonical_v) || canonical_v < 0.0F || !(canonical_v < 1.0F)) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    if (!(outgoing_local.z > 0.0F)) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = Status::outside_support,
            .reserved = {},
        };
    }

    const auto isotropic = alpha_x == alpha_y;
    const auto inverse_alpha = 1.0F / alpha_x;
    const auto common_scale = fmaxf(1.0F, fmaxf(alpha_x, alpha_y));
    const auto scaled_alpha_x = alpha_x / common_scale;
    const auto scaled_alpha_y = alpha_y / common_scale;
    const auto stretched = isotropic
                               ? (alpha_x <= 1.0F ? Vector3{.x = alpha_x * outgoing_local.x,
                                                            .y = alpha_x * outgoing_local.y,
                                                            .z = outgoing_local.z}
                                                  : Vector3{.x = outgoing_local.x,
                                                            .y = outgoing_local.y,
                                                            .z = outgoing_local.z * inverse_alpha})
                               : Vector3{.x = scaled_alpha_x * outgoing_local.x,
                                         .y = scaled_alpha_y * outgoing_local.y,
                                         .z = outgoing_local.z / common_scale};
    if ((isotropic && alpha_x <= 1.0F &&
         ((outgoing_local.x != 0.0F && stretched.x == 0.0F) ||
          (outgoing_local.y != 0.0F && stretched.y == 0.0F))) ||
        (isotropic && alpha_x > 1.0F && stretched.z == 0.0F) ||
        (!isotropic &&
         ((outgoing_local.x != 0.0F && stretched.x == 0.0F) ||
          (outgoing_local.y != 0.0F && stretched.y == 0.0F) || stretched.z == 0.0F))) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = Status::not_representable,
            .reserved = {},
        };
    }
    const auto hemisphere_view = normalized_vector(stretched);
    if (!succeeded(hemisphere_view.status)) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = hemisphere_view.status,
            .reserved = {},
        };
    }
    const auto tangent_length = hypotf(hemisphere_view.value.x, hemisphere_view.value.y);
    const auto tangent = tangent_length > 0.0F
                             ? Vector3{.x = -hemisphere_view.value.y / tangent_length,
                                       .y = hemisphere_view.value.x / tangent_length,
                                       .z = 0.0F}
                             : Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F};
    const auto bitangent = cross(hemisphere_view.value, tangent);
    const auto disk_radius = sqrtf(canonical_u);
    const auto azimuth = TwoPi * canonical_v;
    const auto disk_x = disk_radius * cosf(azimuth);
    const auto initial_disk_y = disk_radius * sinf(azimuth);
    const auto disk_y_limit_squared = (1.0F - disk_x) * (1.0F + disk_x);
    const auto checked_disk_y_limit_squared =
        ggx_nonnegative_roundoff(disk_y_limit_squared, 1.0F + fabsf(disk_x));
    if (!succeeded(checked_disk_y_limit_squared.status)) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = checked_disk_y_limit_squared.status,
            .reserved = {},
        };
    }
    const auto disk_y_limit = sqrtf(checked_disk_y_limit_squared.value);
    const auto projection = 0.5F * (1.0F + hemisphere_view.value.z);
    const auto disk_y = fmaf(projection, initial_disk_y, (1.0F - projection) * disk_y_limit);
    const auto hemisphere_z_squared = fmaf(-disk_y, disk_y, checked_disk_y_limit_squared.value);
    const auto checked_hemisphere_z_squared = ggx_nonnegative_roundoff(
        hemisphere_z_squared, checked_disk_y_limit_squared.value + fabsf(disk_y * disk_y));
    if (!succeeded(checked_hemisphere_z_squared.status)) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = checked_hemisphere_z_squared.status,
            .reserved = {},
        };
    }
    const auto hemisphere_z = sqrtf(checked_hemisphere_z_squared.value);
    const auto hemisphere_normal = Vector3{
        .x = fmaf(disk_x, tangent.x,
                  fmaf(disk_y, bitangent.x, hemisphere_z * hemisphere_view.value.x)),
        .y = fmaf(disk_x, tangent.y,
                  fmaf(disk_y, bitangent.y, hemisphere_z * hemisphere_view.value.y)),
        .z = fmaf(disk_x, tangent.z,
                  fmaf(disk_y, bitangent.z, hemisphere_z * hemisphere_view.value.z)),
    };
    if (!(hemisphere_normal.z > 0.0F)) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = Status::not_representable,
            .reserved = {},
        };
    }
    const auto unstretched =
        isotropic ? (alpha_x <= 1.0F ? Vector3{.x = alpha_x * hemisphere_normal.x,
                                               .y = alpha_x * hemisphere_normal.y,
                                               .z = hemisphere_normal.z}
                                     : Vector3{.x = hemisphere_normal.x,
                                               .y = hemisphere_normal.y,
                                               .z = hemisphere_normal.z * inverse_alpha})
                  : Vector3{.x = scaled_alpha_x * hemisphere_normal.x,
                            .y = scaled_alpha_y * hemisphere_normal.y,
                            .z = hemisphere_normal.z / common_scale};
    if ((isotropic && alpha_x <= 1.0F &&
         ((hemisphere_normal.x != 0.0F && unstretched.x == 0.0F) ||
          (hemisphere_normal.y != 0.0F && unstretched.y == 0.0F))) ||
        (isotropic && alpha_x > 1.0F && unstretched.z == 0.0F) ||
        (!isotropic &&
         ((hemisphere_normal.x != 0.0F && unstretched.x == 0.0F) ||
          (hemisphere_normal.y != 0.0F && unstretched.y == 0.0F) || unstretched.z == 0.0F))) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = Status::not_representable,
            .reserved = {},
        };
    }
    const auto microfacet_normal = normalized_vector(unstretched);
    if (!succeeded(microfacet_normal.status) || !(microfacet_normal.value.z > 0.0F) ||
        !(dot(outgoing_local, microfacet_normal.value) > 0.0F)) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = succeeded(microfacet_normal.status) ? Status::not_representable
                                                          : microfacet_normal.status,
            .reserved = {},
        };
    }
    const auto probability =
        ggx_visible_normal_pdf(alpha_x, alpha_y, outgoing_local, microfacet_normal.value);
    if (!succeeded(probability.status)) {
        return VisibleNormalSampleResult{
            .microfacet_normal = {},
            .probability = zero,
            .status = probability.status,
            .reserved = {},
        };
    }
    return VisibleNormalSampleResult{
        .microfacet_normal = microfacet_normal.value,
        .probability = probability.value,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline bool
positive_spectrum(const shared::TransportSpectrum& value) noexcept {
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        if (!isfinite(value.values[lane]) || !(value.values[lane] > 0.0F)) {
            return false;
        }
    }
    return true;
}

struct ReflectionGeometry final {
    Vector3 microfacet_normal;
    float half_angle_cosine;
    Status status;
    std::uint8_t reserved[3U];
};

[[nodiscard]] __host__ __device__ inline ReflectionGeometry
reflection_geometry(const Vector3 outgoing_local, const Vector3 incoming_local) noexcept {
    const auto half_vector = Vector3{
        .x = outgoing_local.x + incoming_local.x,
        .y = outgoing_local.y + incoming_local.y,
        .z = outgoing_local.z + incoming_local.z,
    };
    const auto magnitude = length(half_vector);
    if (!isfinite(magnitude) || !(magnitude > 0.0F)) {
        return ReflectionGeometry{.status = Status::not_representable, .reserved = {}};
    }
    const auto normal = Vector3{.x = half_vector.x / magnitude,
                                .y = half_vector.y / magnitude,
                                .z = half_vector.z / magnitude};
    const auto half_angle_cosine = 0.5F * magnitude;
    if (!unit_vector(normal) || !(normal.z > 0.0F) || !isfinite(half_angle_cosine) ||
        !(half_angle_cosine > 0.0F) || half_angle_cosine > 1.0F) {
        return ReflectionGeometry{.status = Status::not_representable, .reserved = {}};
    }
    return ReflectionGeometry{
        .microfacet_normal = normal,
        .half_angle_cosine = half_angle_cosine,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline SpectrumResult rough_conductor_eval(
    const shared::TransportSpectrum& coefficient, const shared::TransportSpectrum& relative_eta,
    const shared::TransportSpectrum& relative_k, const float alpha_x, const float alpha_y,
    const Vector3 outgoing_local, const Vector3 incoming_local) noexcept {
    if (!spectrum_is_reflectance(coefficient) || !positive_spectrum(relative_eta) ||
        !spectrum_is_finite_nonnegative(relative_k) ||
        !representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(outgoing_local) ||
        !unit_vector(incoming_local)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    if (!(outgoing_local.z > 0.0F) || !(incoming_local.z > 0.0F) || spectrum_is_zero(coefficient)) {
        return SpectrumResult{.value = {}, .status = Status::success, .reserved = {}};
    }
    const auto geometry = reflection_geometry(outgoing_local, incoming_local);
    if (!succeeded(geometry.status)) {
        return SpectrumResult{.value = {}, .status = geometry.status, .reserved = {}};
    }
    const auto distribution = ggx_normal_distribution(alpha_x, alpha_y, geometry.microfacet_normal);
    const auto masking = ggx_smith_g2(alpha_x, alpha_y, outgoing_local, incoming_local);
    const auto fresnel = conductor_fresnel(geometry.half_angle_cosine, relative_eta, relative_k);
    if (!succeeded(distribution.status) || !succeeded(masking.status) ||
        !succeeded(fresnel.status)) {
        return SpectrumResult{
            .value = {},
            .status = !succeeded(distribution.status)
                          ? distribution.status
                          : (!succeeded(masking.status) ? masking.status : fresnel.status),
            .reserved = {},
        };
    }
    const auto denominator = 4.0F * outgoing_local.z * incoming_local.z;
    const auto scale = distribution.value * masking.value / denominator;
    if (!isfinite(denominator) || !(denominator > 0.0F) || !isfinite(scale) || !(scale > 0.0F)) {
        return SpectrumResult{.value = {}, .status = Status::not_representable, .reserved = {}};
    }
    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        result.values[lane] = coefficient.values[lane] * fresnel.value.values[lane] * scale;
        if (!isfinite(result.values[lane]) || result.values[lane] < 0.0F) {
            return SpectrumResult{.value = {}, .status = Status::not_representable, .reserved = {}};
        }
    }
    return SpectrumResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
rough_conductor_pdf(const float alpha_x, const float alpha_y, const Vector3 outgoing_local,
                    const Vector3 incoming_local) noexcept {
    const auto zero = probability_density(0.0F, ProbabilityMeasure::solid_angle);
    if (!representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(outgoing_local) ||
        !unit_vector(incoming_local)) {
        return ProbabilityResult{.value = zero, .status = Status::invalid_argument, .reserved = {}};
    }
    if (!(outgoing_local.z > 0.0F) || !(incoming_local.z > 0.0F)) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    const auto geometry = reflection_geometry(outgoing_local, incoming_local);
    if (!succeeded(geometry.status)) {
        return ProbabilityResult{.value = zero, .status = geometry.status, .reserved = {}};
    }
    const auto visible =
        ggx_visible_normal_pdf(alpha_x, alpha_y, outgoing_local, geometry.microfacet_normal);
    if (!succeeded(visible.status)) {
        return ProbabilityResult{.value = zero, .status = visible.status, .reserved = {}};
    }
    const auto value = visible.value.value / (4.0F * geometry.half_angle_cosine);
    if (!isfinite(value) || !(value > 0.0F)) {
        return ProbabilityResult{
            .value = zero, .status = Status::not_representable, .reserved = {}};
    }
    return ProbabilityResult{
        .value = probability_density(value, ProbabilityMeasure::solid_angle),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult sample_rough_conductor(
    const shared::TransportSpectrum& coefficient, const shared::TransportSpectrum& relative_eta,
    const shared::TransportSpectrum& relative_k, const float alpha_x, const float alpha_y,
    const Vector3 outgoing_local, const float canonical_u, const float canonical_v) noexcept {
    if (!spectrum_is_reflectance(coefficient) || !positive_spectrum(relative_eta) ||
        !spectrum_is_finite_nonnegative(relative_k) ||
        !representable_ggx_width_pair(alpha_x, alpha_y) || !unit_vector(outgoing_local)) {
        return empty_lobe_sample(Status::invalid_argument);
    }
    const auto visible =
        ggx_sample_visible_normal(alpha_x, alpha_y, outgoing_local, canonical_u, canonical_v);
    if (!succeeded(visible.status)) {
        return empty_lobe_sample(visible.status);
    }
    const auto half_angle_cosine = dot(outgoing_local, visible.microfacet_normal);
    if (!isfinite(half_angle_cosine) || !(half_angle_cosine > 0.0F) || half_angle_cosine > 1.0F) {
        return empty_lobe_sample(Status::not_representable);
    }
    const auto incoming = Vector3{
        .x = 2.0F * half_angle_cosine * visible.microfacet_normal.x - outgoing_local.x,
        .y = 2.0F * half_angle_cosine * visible.microfacet_normal.y - outgoing_local.y,
        .z = 2.0F * half_angle_cosine * visible.microfacet_normal.z - outgoing_local.z,
    };
    if (!unit_vector(incoming)) {
        return empty_lobe_sample(Status::not_representable);
    }
    if (!(incoming.z > 0.0F)) {
        return empty_lobe_sample(Status::outside_support);
    }
    const auto value = rough_conductor_eval(coefficient, relative_eta, relative_k, alpha_x, alpha_y,
                                            outgoing_local, incoming);
    const auto probability = rough_conductor_pdf(alpha_x, alpha_y, outgoing_local, incoming);
    if (!succeeded(value.status) || !succeeded(probability.status)) {
        return empty_lobe_sample(!succeeded(value.status) ? value.status : probability.status);
    }
    return LobeSampleResult{
        .value = value.value,
        .incoming_local = incoming,
        .probability = probability.value,
        .lobes = ScatteringLobe::glossy | ScatteringLobe::reflection,
        .eta_scale_multiplier = 1.0F,
        .status = Status::success,
        .reserved = {},
    };
}

struct DielectricInterface final {
    float face_sign;
    float incident_eta;
    float transmitted_eta;
    float normalized_incident_eta;
    float normalized_transmitted_eta;
};

struct TransmissionGeometry final {
    Vector3 microfacet_normal;
    float outgoing_dot_microfacet;
    float incoming_dot_microfacet;
    float normalized_half_vector_length;
    Status status;
    std::uint8_t supported;
    std::uint8_t reserved[2U];
};

[[nodiscard]] __host__ __device__ inline DielectricInterface
dielectric_interface(const Vector3 outgoing_local, const float exterior_eta,
                     const float interior_eta) noexcept {
    const auto face_sign = outgoing_local.z > 0.0F ? 1.0F : -1.0F;
    const auto incident_eta = face_sign > 0.0F ? exterior_eta : interior_eta;
    const auto transmitted_eta = face_sign > 0.0F ? interior_eta : exterior_eta;
    const auto maximum_eta = fmaxf(incident_eta, transmitted_eta);
    return DielectricInterface{
        .face_sign = face_sign,
        .incident_eta = incident_eta,
        .transmitted_eta = transmitted_eta,
        .normalized_incident_eta = incident_eta / maximum_eta,
        .normalized_transmitted_eta = transmitted_eta / maximum_eta,
    };
}

[[nodiscard]] __host__ __device__ inline Vector3 scaled_direction(const Vector3 direction,
                                                                  const float scale) noexcept {
    return Vector3{.x = scale * direction.x, .y = scale * direction.y, .z = scale * direction.z};
}

[[nodiscard]] __host__ __device__ inline TransmissionGeometry
transmission_geometry(const Vector3 outgoing_face, const Vector3 incoming_face,
                      const DielectricInterface interface) noexcept {
    auto half_vector = Vector3{
        .x = fmaf(interface.normalized_incident_eta, outgoing_face.x,
                  interface.normalized_transmitted_eta * incoming_face.x),
        .y = fmaf(interface.normalized_incident_eta, outgoing_face.y,
                  interface.normalized_transmitted_eta * incoming_face.y),
        .z = fmaf(interface.normalized_incident_eta, outgoing_face.z,
                  interface.normalized_transmitted_eta * incoming_face.z),
    };
    const auto magnitude = length(half_vector);
    if (!isfinite(magnitude) || !(magnitude > 0.0F)) {
        return TransmissionGeometry{.status = Status::not_representable, .reserved = {}};
    }
    half_vector = scaled_direction(half_vector, 1.0F / magnitude);
    if (half_vector.z < 0.0F) {
        half_vector = scaled_direction(half_vector, -1.0F);
    }
    if (half_vector.z == 0.0F) {
        return TransmissionGeometry{.status = Status::success, .supported = 0U, .reserved = {}};
    }
    if (!unit_vector(half_vector)) {
        return TransmissionGeometry{.status = Status::not_representable, .reserved = {}};
    }
    const auto outgoing_dot_microfacet = dot(outgoing_face, half_vector);
    const auto incoming_dot_microfacet = dot(incoming_face, half_vector);
    if (!isfinite(outgoing_dot_microfacet) || !isfinite(incoming_dot_microfacet)) {
        return TransmissionGeometry{.status = Status::not_representable, .reserved = {}};
    }
    if (!(outgoing_dot_microfacet > 0.0F) || !(incoming_dot_microfacet < 0.0F)) {
        return TransmissionGeometry{.status = Status::success, .supported = 0U, .reserved = {}};
    }
    if (outgoing_dot_microfacet > 1.0F || incoming_dot_microfacet < -1.0F) {
        return TransmissionGeometry{.status = Status::not_representable, .reserved = {}};
    }
    return TransmissionGeometry{
        .microfacet_normal = half_vector,
        .outgoing_dot_microfacet = outgoing_dot_microfacet,
        .incoming_dot_microfacet = incoming_dot_microfacet,
        .normalized_half_vector_length = magnitude,
        .status = Status::success,
        .supported = 1U,
        .reserved = {},
    };
}

template <std::size_t NumeratorCount, std::size_t DenominatorCount>
[[nodiscard]] __host__ __device__ inline ScalarResult
checked_lobe_product_ratio(const float (&numerators)[NumeratorCount],
                           const float (&denominators)[DenominatorCount]) noexcept {
    auto has_zero = false;
    for (const auto value : numerators) {
        if (!isfinite(value) || value < 0.0F) {
            return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
        }
        has_zero = has_zero || value == 0.0F;
    }
    for (const auto value : denominators) {
        if (!isfinite(value) || !(value > 0.0F)) {
            return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
        }
    }
    if (has_zero) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }
    auto significand = 1.0F;
    auto exponent = 0;
    for (const auto value : numerators) {
        auto value_exponent = 0;
        const auto value_significand = frexpf(value, &value_exponent);
        auto normalization_exponent = 0;
        significand = frexpf(significand * value_significand, &normalization_exponent);
        exponent += value_exponent + normalization_exponent;
    }
    for (const auto value : denominators) {
        auto value_exponent = 0;
        const auto value_significand = frexpf(value, &value_exponent);
        auto normalization_exponent = 0;
        significand = frexpf(significand / value_significand, &normalization_exponent);
        exponent += normalization_exponent - value_exponent;
    }
    const auto result = scalbnf(significand, exponent);
    if (!isfinite(result) || !(result > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
scale_spectrum(const shared::TransportSpectrum& coefficient, const float scale) noexcept {
    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        const float numerators[]{coefficient.values[lane], scale};
        const float denominators[]{1.0F};
        const auto value = checked_lobe_product_ratio(numerators, denominators);
        if (!succeeded(value.status)) {
            return SpectrumResult{.value = {}, .status = value.status, .reserved = {}};
        }
        result.values[lane] = value.value;
    }
    return SpectrumResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline bool
valid_rough_dielectric(const shared::TransportSpectrum& coefficient, const float exterior_eta,
                       const float interior_eta, const float alpha_x,
                       const float alpha_y) noexcept {
    return spectrum_is_reflectance(coefficient) && isfinite(exterior_eta) && exterior_eta > 0.0F &&
           isfinite(interior_eta) && interior_eta > 0.0F && exterior_eta != interior_eta &&
           representable_ggx_width_pair(alpha_x, alpha_y);
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
rough_dielectric_eval(const shared::TransportSpectrum& coefficient, const float exterior_eta,
                      const float interior_eta, const float alpha_x, const float alpha_y,
                      const Vector3 outgoing_local, const Vector3 incoming_local,
                      const TransportMode mode) noexcept {
    if (!valid_rough_dielectric(coefficient, exterior_eta, interior_eta, alpha_x, alpha_y) ||
        !known_transport_mode(mode) || !unit_vector(outgoing_local) ||
        !unit_vector(incoming_local)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    if (outgoing_local.z == 0.0F || incoming_local.z == 0.0F) {
        return SpectrumResult{.value = {}, .status = Status::success, .reserved = {}};
    }
    const auto interface = dielectric_interface(outgoing_local, exterior_eta, interior_eta);
    const auto outgoing_face = scaled_direction(outgoing_local, interface.face_sign);
    const auto incoming_face = scaled_direction(incoming_local, interface.face_sign);
    if (incoming_face.z > 0.0F) {
        const auto geometry = reflection_geometry(outgoing_face, incoming_face);
        if (!succeeded(geometry.status)) {
            return SpectrumResult{.value = {}, .status = geometry.status, .reserved = {}};
        }
        const auto fresnel = dielectric_fresnel(geometry.half_angle_cosine, interface.incident_eta,
                                                interface.transmitted_eta);
        const auto distribution =
            ggx_normal_distribution(alpha_x, alpha_y, geometry.microfacet_normal);
        const auto masking = ggx_smith_g2(alpha_x, alpha_y, outgoing_face, incoming_face);
        if (!succeeded(fresnel.status) || !succeeded(distribution.status) ||
            !succeeded(masking.status)) {
            return SpectrumResult{
                .value = {},
                .status =
                    !succeeded(fresnel.status)
                        ? fresnel.status
                        : (!succeeded(distribution.status) ? distribution.status : masking.status),
                .reserved = {},
            };
        }
        const float numerators[]{fresnel.value, distribution.value, masking.value};
        const float denominators[]{4.0F, outgoing_face.z, incoming_face.z};
        const auto scale = checked_lobe_product_ratio(numerators, denominators);
        return succeeded(scale.status)
                   ? scale_spectrum(coefficient, scale.value)
                   : SpectrumResult{.value = {}, .status = scale.status, .reserved = {}};
    }

    const auto geometry = transmission_geometry(outgoing_face, incoming_face, interface);
    if (!succeeded(geometry.status) || geometry.supported == 0U) {
        return SpectrumResult{
            .value = {},
            .status = succeeded(geometry.status) ? Status::success : geometry.status,
            .reserved = {},
        };
    }
    const auto fresnel = dielectric_fresnel(geometry.outgoing_dot_microfacet,
                                            interface.incident_eta, interface.transmitted_eta);
    if (!succeeded(fresnel.status)) {
        return SpectrumResult{.value = {}, .status = fresnel.status, .reserved = {}};
    }
    if (fresnel.value == 1.0F) {
        return SpectrumResult{.value = {}, .status = Status::success, .reserved = {}};
    }
    const auto distribution = ggx_normal_distribution(alpha_x, alpha_y, geometry.microfacet_normal);
    const auto incoming_mask = scaled_direction(incoming_face, -1.0F);
    const auto masking = ggx_smith_g2(alpha_x, alpha_y, outgoing_face, incoming_mask);
    if (!succeeded(distribution.status) || !succeeded(masking.status)) {
        return SpectrumResult{
            .value = {},
            .status = !succeeded(distribution.status) ? distribution.status : masking.status,
            .reserved = {},
        };
    }
    const auto transport_eta = mode == TransportMode::radiance
                                   ? interface.normalized_incident_eta
                                   : interface.normalized_transmitted_eta;
    const float numerators[]{1.0F - fresnel.value,
                             distribution.value,
                             masking.value,
                             transport_eta,
                             transport_eta,
                             geometry.outgoing_dot_microfacet,
                             -geometry.incoming_dot_microfacet};
    const float denominators[]{outgoing_face.z, -incoming_face.z,
                               geometry.normalized_half_vector_length,
                               geometry.normalized_half_vector_length};
    const auto scale = checked_lobe_product_ratio(numerators, denominators);
    return succeeded(scale.status)
               ? scale_spectrum(coefficient, scale.value)
               : SpectrumResult{.value = {}, .status = scale.status, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
rough_dielectric_pdf(const shared::TransportSpectrum& coefficient, const float exterior_eta,
                     const float interior_eta, const float alpha_x, const float alpha_y,
                     const Vector3 outgoing_local, const Vector3 incoming_local,
                     const TransportMode mode) noexcept {
    const auto zero = probability_density(0.0F, ProbabilityMeasure::solid_angle);
    if (!valid_rough_dielectric(coefficient, exterior_eta, interior_eta, alpha_x, alpha_y) ||
        !known_transport_mode(mode) || !unit_vector(outgoing_local) ||
        !unit_vector(incoming_local)) {
        return ProbabilityResult{.value = zero, .status = Status::invalid_argument, .reserved = {}};
    }
    if (outgoing_local.z == 0.0F || incoming_local.z == 0.0F) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    const auto interface = dielectric_interface(outgoing_local, exterior_eta, interior_eta);
    const auto outgoing_face = scaled_direction(outgoing_local, interface.face_sign);
    const auto incoming_face = scaled_direction(incoming_local, interface.face_sign);
    if (incoming_face.z > 0.0F) {
        const auto geometry = reflection_geometry(outgoing_face, incoming_face);
        if (!succeeded(geometry.status)) {
            return ProbabilityResult{.value = zero, .status = geometry.status, .reserved = {}};
        }
        const auto fresnel = dielectric_fresnel(geometry.half_angle_cosine, interface.incident_eta,
                                                interface.transmitted_eta);
        const auto visible =
            ggx_visible_normal_pdf(alpha_x, alpha_y, outgoing_face, geometry.microfacet_normal);
        if (!succeeded(fresnel.status) || !succeeded(visible.status)) {
            return ProbabilityResult{
                .value = zero,
                .status = !succeeded(fresnel.status) ? fresnel.status : visible.status,
                .reserved = {},
            };
        }
        const float numerators[]{fresnel.value, visible.value.value};
        const float denominators[]{4.0F, geometry.half_angle_cosine};
        const auto probability = checked_lobe_product_ratio(numerators, denominators);
        if (!succeeded(probability.status)) {
            return ProbabilityResult{.value = zero, .status = probability.status, .reserved = {}};
        }
        return ProbabilityResult{
            .value = probability_density(probability.value, ProbabilityMeasure::solid_angle),
            .status = Status::success,
            .reserved = {},
        };
    }
    const auto geometry = transmission_geometry(outgoing_face, incoming_face, interface);
    if (!succeeded(geometry.status) || geometry.supported == 0U) {
        return ProbabilityResult{
            .value = zero,
            .status = succeeded(geometry.status) ? Status::success : geometry.status,
            .reserved = {},
        };
    }
    const auto fresnel = dielectric_fresnel(geometry.outgoing_dot_microfacet,
                                            interface.incident_eta, interface.transmitted_eta);
    if (!succeeded(fresnel.status)) {
        return ProbabilityResult{.value = zero, .status = fresnel.status, .reserved = {}};
    }
    if (fresnel.value == 1.0F) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    const auto visible =
        ggx_visible_normal_pdf(alpha_x, alpha_y, outgoing_face, geometry.microfacet_normal);
    if (!succeeded(visible.status)) {
        return ProbabilityResult{.value = zero, .status = visible.status, .reserved = {}};
    }
    const float numerators[]{
        1.0F - fresnel.value, visible.value.value, interface.normalized_transmitted_eta,
        interface.normalized_transmitted_eta, -geometry.incoming_dot_microfacet};
    const float denominators[]{geometry.normalized_half_vector_length,
                               geometry.normalized_half_vector_length};
    const auto probability = checked_lobe_product_ratio(numerators, denominators);
    if (!succeeded(probability.status)) {
        return ProbabilityResult{.value = zero, .status = probability.status, .reserved = {}};
    }
    return ProbabilityResult{
        .value = probability_density(probability.value, ProbabilityMeasure::solid_angle),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult
sample_rough_dielectric(const shared::TransportSpectrum& coefficient, const float exterior_eta,
                        const float interior_eta, const float alpha_x, const float alpha_y,
                        const Vector3 outgoing_local, const float event_sample,
                        const float canonical_u, const float canonical_v,
                        const TransportMode mode) noexcept {
    if (!valid_rough_dielectric(coefficient, exterior_eta, interior_eta, alpha_x, alpha_y) ||
        !known_transport_mode(mode) || !unit_vector(outgoing_local) || !isfinite(event_sample) ||
        event_sample < 0.0F || !(event_sample < 1.0F)) {
        return empty_lobe_sample(Status::invalid_argument);
    }
    if (outgoing_local.z == 0.0F) {
        return empty_lobe_sample(Status::outside_support);
    }
    const auto interface = dielectric_interface(outgoing_local, exterior_eta, interior_eta);
    const auto outgoing_face = scaled_direction(outgoing_local, interface.face_sign);
    const auto sampled_normal =
        ggx_sample_visible_normal(alpha_x, alpha_y, outgoing_face, canonical_u, canonical_v);
    if (!succeeded(sampled_normal.status)) {
        return empty_lobe_sample(sampled_normal.status);
    }
    const auto outgoing_dot = dot(outgoing_face, sampled_normal.microfacet_normal);
    if (!isfinite(outgoing_dot) || !(outgoing_dot > 0.0F) || outgoing_dot > 1.0F) {
        return empty_lobe_sample(Status::not_representable);
    }
    const auto fresnel =
        dielectric_fresnel(outgoing_dot, interface.incident_eta, interface.transmitted_eta);
    if (!succeeded(fresnel.status)) {
        return empty_lobe_sample(fresnel.status);
    }

    auto incoming_face = Vector3{};
    auto lobes = ScatteringLobe::none;
    auto eta_scale_multiplier = 1.0F;
    if (event_sample < fresnel.value) {
        incoming_face = Vector3{
            .x = 2.0F * outgoing_dot * sampled_normal.microfacet_normal.x - outgoing_face.x,
            .y = 2.0F * outgoing_dot * sampled_normal.microfacet_normal.y - outgoing_face.y,
            .z = 2.0F * outgoing_dot * sampled_normal.microfacet_normal.z - outgoing_face.z,
        };
        lobes = ScatteringLobe::glossy | ScatteringLobe::reflection;
        if (!unit_vector(incoming_face)) {
            return empty_lobe_sample(Status::not_representable);
        }
        if (!(incoming_face.z > 0.0F)) {
            return empty_lobe_sample(Status::outside_support);
        }
    } else {
        const auto tangent = Vector3{
            .x = outgoing_face.x - outgoing_dot * sampled_normal.microfacet_normal.x,
            .y = outgoing_face.y - outgoing_dot * sampled_normal.microfacet_normal.y,
            .z = outgoing_face.z - outgoing_dot * sampled_normal.microfacet_normal.z,
        };
        const auto incident_sine = length(tangent);
        if (!isfinite(incident_sine) || incident_sine > 1.0F) {
            return empty_lobe_sample(Status::not_representable);
        }
        auto incoming_cosine = 1.0F;
        auto tangent_scale = 0.0F;
        if (incident_sine > 0.0F) {
            auto transmitted_sine = 0.0F;
            if (interface.incident_eta > interface.transmitted_eta) {
                const auto critical_sine = interface.transmitted_eta / interface.incident_eta;
                if (critical_sine == 0.0F || incident_sine >= critical_sine) {
                    return empty_lobe_sample(Status::not_representable);
                }
                transmitted_sine = incident_sine / critical_sine;
            } else {
                const auto eta_ratio = interface.incident_eta / interface.transmitted_eta;
                if (eta_ratio == 0.0F) {
                    return empty_lobe_sample(Status::not_representable);
                }
                transmitted_sine = eta_ratio * incident_sine;
            }
            if (!isfinite(transmitted_sine) || !(transmitted_sine < 1.0F)) {
                return empty_lobe_sample(Status::not_representable);
            }
            incoming_cosine = sqrtf((1.0F - transmitted_sine) * (1.0F + transmitted_sine));
            tangent_scale = transmitted_sine / incident_sine;
            if (!isfinite(incoming_cosine) || !(incoming_cosine > 0.0F) ||
                !isfinite(tangent_scale) || !(tangent_scale > 0.0F)) {
                return empty_lobe_sample(Status::not_representable);
            }
        }
        incoming_face = Vector3{
            .x = fmaf(-tangent_scale, tangent.x,
                      -incoming_cosine * sampled_normal.microfacet_normal.x),
            .y = fmaf(-tangent_scale, tangent.y,
                      -incoming_cosine * sampled_normal.microfacet_normal.y),
            .z = fmaf(-tangent_scale, tangent.z,
                      -incoming_cosine * sampled_normal.microfacet_normal.z),
        };
        lobes = ScatteringLobe::glossy | ScatteringLobe::transmission;
        if (!unit_vector(incoming_face)) {
            return empty_lobe_sample(Status::not_representable);
        }
        if (!(incoming_face.z < 0.0F)) {
            return empty_lobe_sample(Status::outside_support);
        }
        if (mode == TransportMode::radiance) {
            const float numerators[]{interface.normalized_transmitted_eta,
                                     interface.normalized_transmitted_eta};
            const float denominators[]{interface.normalized_incident_eta,
                                       interface.normalized_incident_eta};
            const auto eta_scale = checked_lobe_product_ratio(numerators, denominators);
            if (!succeeded(eta_scale.status)) {
                return empty_lobe_sample(eta_scale.status);
            }
            eta_scale_multiplier = eta_scale.value;
        }
    }
    const auto incoming_local = scaled_direction(incoming_face, interface.face_sign);
    const auto value = rough_dielectric_eval(coefficient, exterior_eta, interior_eta, alpha_x,
                                             alpha_y, outgoing_local, incoming_local, mode);
    const auto probability = rough_dielectric_pdf(coefficient, exterior_eta, interior_eta, alpha_x,
                                                  alpha_y, outgoing_local, incoming_local, mode);
    if (!succeeded(value.status) || !succeeded(probability.status)) {
        return empty_lobe_sample(!succeeded(value.status) ? value.status : probability.status);
    }
    return LobeSampleResult{
        .value = value.value,
        .incoming_local = incoming_local,
        .probability = probability.value,
        .lobes = lobes,
        .eta_scale_multiplier = eta_scale_multiplier,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline SpectrumResult rough_dielectric_eval_with_direction_mask(
    const shared::TransportSpectrum& coefficient, const float exterior_eta,
    const float interior_eta, const float alpha_x, const float alpha_y,
    const Vector3 outgoing_local, const Vector3 incoming_local, const TransportMode mode,
    const ScatteringLobe allowed_directions) noexcept {
    if (!valid_rough_dielectric_direction_mask(allowed_directions)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    const auto evaluated = rough_dielectric_eval(coefficient, exterior_eta, interior_eta, alpha_x,
                                                 alpha_y, outgoing_local, incoming_local, mode);
    if (!succeeded(evaluated.status) || outgoing_local.z == 0.0F || incoming_local.z == 0.0F) {
        return evaluated;
    }
    const auto direction = (outgoing_local.z > 0.0F) == (incoming_local.z > 0.0F)
                               ? ScatteringLobe::reflection
                               : ScatteringLobe::transmission;
    return has_scattering_lobe(allowed_directions, direction)
               ? evaluated
               : SpectrumResult{.value = {}, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult rough_dielectric_pdf_with_direction_mask(
    const shared::TransportSpectrum& coefficient, const float exterior_eta,
    const float interior_eta, const float alpha_x, const float alpha_y,
    const Vector3 outgoing_local, const Vector3 incoming_local, const TransportMode mode,
    const ScatteringLobe allowed_directions) noexcept {
    const auto zero = probability_density(0.0F, ProbabilityMeasure::solid_angle);
    if (!valid_rough_dielectric_direction_mask(allowed_directions)) {
        return ProbabilityResult{.value = zero, .status = Status::invalid_argument, .reserved = {}};
    }
    const auto base = rough_dielectric_pdf(coefficient, exterior_eta, interior_eta, alpha_x,
                                           alpha_y, outgoing_local, incoming_local, mode);
    if (!succeeded(base.status) || outgoing_local.z == 0.0F || incoming_local.z == 0.0F ||
        base.value.value == 0.0F) {
        return base;
    }

    const auto reflection = (outgoing_local.z > 0.0F) == (incoming_local.z > 0.0F);
    const auto direction = reflection ? ScatteringLobe::reflection : ScatteringLobe::transmission;
    if (!has_scattering_lobe(allowed_directions, direction)) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    if (allowed_directions == ScatteringDirectionMask) {
        return base;
    }

    const auto interface = dielectric_interface(outgoing_local, exterior_eta, interior_eta);
    const auto outgoing_face = scaled_direction(outgoing_local, interface.face_sign);
    const auto incoming_face = scaled_direction(incoming_local, interface.face_sign);
    auto branch_probability = 0.0F;
    if (reflection) {
        const auto geometry = reflection_geometry(outgoing_face, incoming_face);
        if (!succeeded(geometry.status)) {
            return ProbabilityResult{.value = zero, .status = geometry.status, .reserved = {}};
        }
        const auto fresnel = dielectric_fresnel(geometry.half_angle_cosine, interface.incident_eta,
                                                interface.transmitted_eta);
        if (!succeeded(fresnel.status)) {
            return ProbabilityResult{.value = zero, .status = fresnel.status, .reserved = {}};
        }
        branch_probability = fresnel.value;
    } else {
        const auto geometry = transmission_geometry(outgoing_face, incoming_face, interface);
        if (!succeeded(geometry.status) || geometry.supported == 0U) {
            return ProbabilityResult{
                .value = zero,
                .status = succeeded(geometry.status) ? Status::success : geometry.status,
                .reserved = {},
            };
        }
        const auto fresnel = dielectric_fresnel(geometry.outgoing_dot_microfacet,
                                                interface.incident_eta, interface.transmitted_eta);
        if (!succeeded(fresnel.status)) {
            return ProbabilityResult{.value = zero, .status = fresnel.status, .reserved = {}};
        }
        branch_probability = 1.0F - fresnel.value;
    }
    if (!isfinite(branch_probability) || !(branch_probability > 0.0F) ||
        branch_probability > 1.0F) {
        return ProbabilityResult{
            .value = zero, .status = Status::not_representable, .reserved = {}};
    }
    const float numerators[]{base.value.value};
    const float denominators[]{branch_probability};
    const auto conditional = checked_lobe_product_ratio(numerators, denominators);
    if (!succeeded(conditional.status)) {
        return ProbabilityResult{.value = zero, .status = conditional.status, .reserved = {}};
    }
    return ProbabilityResult{
        .value = probability_density(conditional.value, ProbabilityMeasure::solid_angle),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult
sample_rough_dielectric_with_direction_mask(const shared::TransportSpectrum& coefficient,
                                            const float exterior_eta, const float interior_eta,
                                            const float alpha_x, const float alpha_y,
                                            const Vector3 outgoing_local, const float event_sample,
                                            const float canonical_u, const float canonical_v,
                                            const TransportMode mode,
                                            const ScatteringLobe allowed_directions) noexcept {
    if (!valid_rough_dielectric_direction_mask(allowed_directions) || !isfinite(event_sample) ||
        event_sample < 0.0F || !(event_sample < 1.0F)) {
        return empty_lobe_sample(Status::invalid_argument);
    }
    auto selected_event = event_sample;
    if (allowed_directions == ScatteringLobe::reflection) {
        selected_event = 0.0F;
    } else if (allowed_directions == ScatteringLobe::transmission) {
        selected_event = nextafterf(1.0F, 0.0F);
    }
    auto sampled =
        sample_rough_dielectric(coefficient, exterior_eta, interior_eta, alpha_x, alpha_y,
                                outgoing_local, selected_event, canonical_u, canonical_v, mode);
    if (!succeeded(sampled.status)) {
        return sampled;
    }
    const auto sampled_direction =
        static_cast<ScatteringLobe>(static_cast<std::uint32_t>(sampled.lobes) &
                                    static_cast<std::uint32_t>(ScatteringDirectionMask));
    if (!has_scattering_lobe(allowed_directions, sampled_direction)) {
        return empty_lobe_sample(Status::outside_support);
    }
    const auto probability = rough_dielectric_pdf_with_direction_mask(
        coefficient, exterior_eta, interior_eta, alpha_x, alpha_y, outgoing_local,
        sampled.incoming_local, mode, allowed_directions);
    if (!succeeded(probability.status)) {
        return empty_lobe_sample(probability.status);
    }
    sampled.probability = probability.value;
    return sampled;
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
specular_delta_eval(const shared::TransportSpectrum& coefficient, const Vector3 outgoing_local,
                    const Vector3 incoming_local) noexcept {
    if (!spectrum_is_reflectance(coefficient) || !unit_vector(outgoing_local) ||
        !unit_vector(incoming_local)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    return SpectrumResult{.value = {}, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
specular_delta_pdf(const shared::TransportSpectrum& coefficient, const Vector3 outgoing_local,
                   const Vector3 incoming_local) noexcept {
    if (!spectrum_is_reflectance(coefficient) || !unit_vector(outgoing_local) ||
        !unit_vector(incoming_local)) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    return ProbabilityResult{
        .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
specular_reflection_eval(const shared::TransportSpectrum& reflectance, const Vector3 outgoing_local,
                         const Vector3 incoming_local) noexcept {
    return specular_delta_eval(reflectance, outgoing_local, incoming_local);
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
specular_reflection_pdf(const shared::TransportSpectrum& reflectance, const Vector3 outgoing_local,
                        const Vector3 incoming_local) noexcept {
    return specular_delta_pdf(reflectance, outgoing_local, incoming_local);
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
specular_transmission_eval(const shared::TransportSpectrum& transmittance, const float exterior_eta,
                           const float interior_eta, const Vector3 outgoing_local,
                           const Vector3 incoming_local, const TransportMode mode) noexcept {
    if (!isfinite(exterior_eta) || !(exterior_eta > 0.0F) || !isfinite(interior_eta) ||
        !(interior_eta > 0.0F) || !known_transport_mode(mode)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    return specular_delta_eval(transmittance, outgoing_local, incoming_local);
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
specular_transmission_pdf(const shared::TransportSpectrum& transmittance, const float exterior_eta,
                          const float interior_eta, const Vector3 outgoing_local,
                          const Vector3 incoming_local, const TransportMode mode) noexcept {
    if (!isfinite(exterior_eta) || !(exterior_eta > 0.0F) || !isfinite(interior_eta) ||
        !(interior_eta > 0.0F) || !known_transport_mode(mode)) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    return specular_delta_pdf(transmittance, outgoing_local, incoming_local);
}

[[nodiscard]] __host__ __device__ inline ScalarResult
checked_positive_product_quotient(const float left, const float right,
                                  const float denominator) noexcept {
    if (left == 0.0F) {
        return ScalarResult{.value = 0.0F, .status = Status::success, .reserved = {}};
    }
    if (!isfinite(left) || !(left > 0.0F) || !isfinite(right) || !(right > 0.0F) ||
        !isfinite(denominator) || !(denominator > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    auto left_exponent = 0;
    auto right_exponent = 0;
    auto denominator_exponent = 0;
    const auto left_significand = frexpf(left, &left_exponent);
    const auto right_significand = frexpf(right, &right_exponent);
    const auto denominator_significand = frexpf(denominator, &denominator_exponent);
    auto normalization_exponent = 0;
    const auto normalized_significand = frexpf(
        (left_significand * right_significand) / denominator_significand, &normalization_exponent);
    const auto result =
        scalbnf(normalized_significand,
                left_exponent + right_exponent - denominator_exponent + normalization_exponent);
    if (!isfinite(result) || !(result > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
scaled_delta_value(const shared::TransportSpectrum& coefficient, const float transport_factor,
                   const float absolute_incoming_cosine) noexcept {
    auto result = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        const auto value = checked_positive_product_quotient(
            coefficient.values[lane], transport_factor, absolute_incoming_cosine);
        if (!succeeded(value.status)) {
            return SpectrumResult{.value = {}, .status = value.status, .reserved = {}};
        }
        result.values[lane] = value.value;
    }
    return SpectrumResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ScalarResult
checked_squared_ratio(const float numerator, const float denominator) noexcept {
    if (!isfinite(numerator) || !(numerator > 0.0F) || !isfinite(denominator) ||
        !(denominator > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::invalid_argument, .reserved = {}};
    }
    auto numerator_exponent = 0;
    auto denominator_exponent = 0;
    const auto numerator_significand = frexpf(numerator, &numerator_exponent);
    const auto denominator_significand = frexpf(denominator, &denominator_exponent);
    const auto ratio_significand = numerator_significand / denominator_significand;
    auto normalization_exponent = 0;
    const auto normalized_significand =
        frexpf(ratio_significand * ratio_significand, &normalization_exponent);
    const auto result =
        scalbnf(normalized_significand,
                2 * (numerator_exponent - denominator_exponent) + normalization_exponent);
    if (!isfinite(result) || !(result > 0.0F)) {
        return ScalarResult{.value = 0.0F, .status = Status::not_representable, .reserved = {}};
    }
    return ScalarResult{.value = result, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult
sample_specular_reflection(const shared::TransportSpectrum& reflectance,
                           const Vector3 outgoing_local) noexcept {
    if (!spectrum_is_reflectance(reflectance) || !unit_vector(outgoing_local)) {
        return empty_lobe_sample(Status::invalid_argument, ProbabilityMeasure::discrete);
    }
    if (outgoing_local.z == 0.0F) {
        return empty_lobe_sample(Status::outside_support, ProbabilityMeasure::discrete);
    }
    const auto incoming =
        Vector3{.x = -outgoing_local.x, .y = -outgoing_local.y, .z = outgoing_local.z};
    const auto value = scaled_delta_value(reflectance, 1.0F, fabsf(incoming.z));
    if (!succeeded(value.status)) {
        return empty_lobe_sample(value.status, ProbabilityMeasure::discrete);
    }
    return LobeSampleResult{
        .value = value.value,
        .incoming_local = incoming,
        .probability = probability_density(1.0F, ProbabilityMeasure::discrete),
        .lobes = ScatteringLobe::specular | ScatteringLobe::reflection,
        .eta_scale_multiplier = 1.0F,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult
sample_specular_transmission(const shared::TransportSpectrum& transmittance,
                             const float exterior_eta, const float interior_eta,
                             const Vector3 outgoing_local, const TransportMode mode) noexcept {
    if (!spectrum_is_reflectance(transmittance) || !isfinite(exterior_eta) ||
        !(exterior_eta > 0.0F) || !isfinite(interior_eta) || !(interior_eta > 0.0F) ||
        !known_transport_mode(mode) || !unit_vector(outgoing_local)) {
        return empty_lobe_sample(Status::invalid_argument, ProbabilityMeasure::discrete);
    }
    if (outgoing_local.z == 0.0F) {
        return empty_lobe_sample(Status::outside_support, ProbabilityMeasure::discrete);
    }
    const auto entering = outgoing_local.z > 0.0F;
    const auto incident_eta = entering ? exterior_eta : interior_eta;
    const auto transmitted_eta = entering ? interior_eta : exterior_eta;
    if (incident_eta == transmitted_eta) {
        const auto incoming =
            Vector3{.x = -outgoing_local.x, .y = -outgoing_local.y, .z = -outgoing_local.z};
        const auto value = scaled_delta_value(transmittance, 1.0F, fabsf(incoming.z));
        if (!succeeded(value.status)) {
            return empty_lobe_sample(value.status, ProbabilityMeasure::discrete);
        }
        return LobeSampleResult{
            .value = value.value,
            .incoming_local = incoming,
            .probability = probability_density(1.0F, ProbabilityMeasure::discrete),
            .lobes = ScatteringLobe::specular | ScatteringLobe::transmission,
            .eta_scale_multiplier = 1.0F,
            .status = Status::success,
            .reserved = {},
        };
    }
    const auto raw_incident_sine = hypotf(outgoing_local.x, outgoing_local.y);
    if (!isfinite(raw_incident_sine)) {
        return empty_lobe_sample(Status::not_representable, ProbabilityMeasure::discrete);
    }
    auto incident_sine = raw_incident_sine;
    auto tangent_normalization = 1.0F;
    if (raw_incident_sine > 1.0F) {
        const auto absolute_incident_cosine = fabsf(outgoing_local.z);
        if (absolute_incident_cosine > 1.0F) {
            return empty_lobe_sample(Status::not_representable, ProbabilityMeasure::discrete);
        }
        incident_sine =
            sqrtf((1.0F - absolute_incident_cosine) * (1.0F + absolute_incident_cosine));
        tangent_normalization = incident_sine / raw_incident_sine;
        if (!isfinite(incident_sine) || !isfinite(tangent_normalization) ||
            !(tangent_normalization > 0.0F)) {
            return empty_lobe_sample(Status::not_representable, ProbabilityMeasure::discrete);
        }
    }
    if (incident_sine > 0.0F) {
        const auto critical_sine = transmitted_eta / incident_eta;
        if (critical_sine <= 1.0F && incident_sine >= critical_sine) {
            return empty_lobe_sample(Status::outside_support, ProbabilityMeasure::discrete);
        }
    }
    const auto incident_over_transmitted_eta = incident_eta / transmitted_eta;
    if (!isfinite(incident_over_transmitted_eta) || !(incident_over_transmitted_eta > 0.0F)) {
        return empty_lobe_sample(Status::not_representable, ProbabilityMeasure::discrete);
    }
    const auto incoming_x =
        -incident_over_transmitted_eta * tangent_normalization * outgoing_local.x;
    const auto incoming_y =
        -incident_over_transmitted_eta * tangent_normalization * outgoing_local.y;
    if (!isfinite(incoming_x) || !isfinite(incoming_y) ||
        (outgoing_local.x != 0.0F && incoming_x == 0.0F) ||
        (outgoing_local.y != 0.0F && incoming_y == 0.0F)) {
        return empty_lobe_sample(Status::not_representable, ProbabilityMeasure::discrete);
    }
    const auto transmitted_sine = hypotf(incoming_x, incoming_y);
    if (!isfinite(transmitted_sine) || !(transmitted_sine < 1.0F)) {
        return empty_lobe_sample(Status::not_representable, ProbabilityMeasure::discrete);
    }
    const auto transmitted_cosine = sqrtf((1.0F - transmitted_sine) * (1.0F + transmitted_sine));
    if (!isfinite(transmitted_cosine) || !(transmitted_cosine > 0.0F)) {
        return empty_lobe_sample(Status::not_representable, ProbabilityMeasure::discrete);
    }
    const auto incoming = Vector3{
        .x = incoming_x,
        .y = incoming_y,
        .z = entering ? -transmitted_cosine : transmitted_cosine,
    };
    if (!unit_vector(incoming)) {
        return empty_lobe_sample(Status::not_representable, ProbabilityMeasure::discrete);
    }
    auto transport_factor = 1.0F;
    auto eta_scale_multiplier = 1.0F;
    if (mode == TransportMode::radiance) {
        const auto adjoint = checked_squared_ratio(incident_eta, transmitted_eta);
        const auto eta_scale = checked_squared_ratio(transmitted_eta, incident_eta);
        if (!succeeded(adjoint.status) || !succeeded(eta_scale.status)) {
            return empty_lobe_sample(!succeeded(adjoint.status) ? adjoint.status : eta_scale.status,
                                     ProbabilityMeasure::discrete);
        }
        transport_factor = adjoint.value;
        eta_scale_multiplier = eta_scale.value;
    }
    const auto value = scaled_delta_value(transmittance, transport_factor, transmitted_cosine);
    if (!succeeded(value.status)) {
        return empty_lobe_sample(value.status, ProbabilityMeasure::discrete);
    }
    return LobeSampleResult{
        .value = value.value,
        .incoming_local = incoming,
        .probability = probability_density(1.0F, ProbabilityMeasure::discrete),
        .lobes = ScatteringLobe::specular | ScatteringLobe::transmission,
        .eta_scale_multiplier = eta_scale_multiplier,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline bool
zero_closure_record(const ClosureRecord& closure) noexcept {
    if (closure.kind != ClosureKind::none || closure.lobes != ScatteringLobe::none) {
        return false;
    }
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        if (closure.weight[lane] != 0.0F) {
            return false;
        }
    }
    for (auto index = std::uint32_t{}; index < ClosureParameterScalarCount; ++index) {
        if (closure.parameters[index] != 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __host__ __device__ inline bool
valid_closure_record(const ClosureRecord& closure) noexcept {
    const auto weight = record_weight(closure);
    if (!spectrum_is_reflectance(weight)) {
        return false;
    }
    switch (closure.kind) {
    case ClosureKind::lambertian_reflection:
        if (closure.lobes != (ScatteringLobe::diffuse | ScatteringLobe::reflection)) {
            return false;
        }
        for (auto index = std::uint32_t{}; index < ClosureParameterScalarCount; ++index) {
            if (closure.parameters[index] != 0.0F) {
                return false;
            }
        }
        return true;
    case ClosureKind::rough_diffuse_reflection:
        if (closure.lobes != (ScatteringLobe::diffuse | ScatteringLobe::reflection) ||
            !isfinite(closure.parameters[0U]) || closure.parameters[0U] < 0.0F ||
            closure.parameters[0U] > 1.0F) {
            return false;
        }
        for (auto index = std::uint32_t{1U}; index < ClosureParameterScalarCount; ++index) {
            if (closure.parameters[index] != 0.0F) {
                return false;
            }
        }
        return true;
    case ClosureKind::rough_conductor_reflection: {
        if (closure.lobes != (ScatteringLobe::glossy | ScatteringLobe::reflection)) {
            return false;
        }
        auto eta = shared::TransportSpectrum{};
        auto k = shared::TransportSpectrum{};
        for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
            eta.values[lane] = closure.parameters[lane];
            k.values[lane] = closure.parameters[SpectrumLaneCount + lane];
        }
        return positive_spectrum(eta) && spectrum_is_finite_nonnegative(k) &&
               representable_ggx_width_pair(closure.parameters[8U], closure.parameters[9U]);
    }
    case ClosureKind::rough_dielectric:
        if (closure.lobes != (ScatteringLobe::glossy | ScatteringLobe::reflection |
                              ScatteringLobe::transmission) ||
            !valid_rough_dielectric(weight, closure.parameters[0U], closure.parameters[1U],
                                    closure.parameters[2U], closure.parameters[3U])) {
            return false;
        }
        for (auto index = std::uint32_t{4U}; index < ClosureParameterScalarCount; ++index) {
            if (closure.parameters[index] != 0.0F) {
                return false;
            }
        }
        return true;
    case ClosureKind::specular_reflection:
        if (closure.lobes != (ScatteringLobe::specular | ScatteringLobe::reflection)) {
            return false;
        }
        for (auto index = std::uint32_t{}; index < ClosureParameterScalarCount; ++index) {
            if (closure.parameters[index] != 0.0F) {
                return false;
            }
        }
        return true;
    case ClosureKind::specular_transmission:
        if (closure.lobes != (ScatteringLobe::specular | ScatteringLobe::transmission) ||
            !isfinite(closure.parameters[0U]) || !(closure.parameters[0U] > 0.0F) ||
            !isfinite(closure.parameters[1U]) || !(closure.parameters[1U] > 0.0F)) {
            return false;
        }
        for (auto index = std::uint32_t{2U}; index < ClosureParameterScalarCount; ++index) {
            if (closure.parameters[index] != 0.0F) {
                return false;
            }
        }
        return true;
    case ClosureKind::none:
        return false;
    }
    return false;
}

[[nodiscard]] __host__ __device__ inline bool
valid_closure_mixture_record(const ClosureMixtureRecord& mixture) noexcept {
    if (mixture.active_count > MaximumClosureCount || mixture.reserved_header[0U] != 0U ||
        mixture.reserved_header[1U] != 0U || mixture.reserved_header[2U] != 0U ||
        mixture.reserved_tail[0U] != 0U || mixture.reserved_tail[1U] != 0U ||
        mixture.reserved_tail[2U] != 0U || mixture.cdf[0U] != 0.0F) {
        return false;
    }
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (!valid_closure_record(mixture.closures[index])) {
            return false;
        }
        const auto probability = mixture.probabilities[index];
        if (!isfinite(probability) || !(probability > 0.0F) || probability > 1.0F ||
            !isfinite(mixture.cdf[index + 1U]) || !(mixture.cdf[index] < mixture.cdf[index + 1U]) ||
            mixture.cdf[index + 1U] - mixture.cdf[index] != probability) {
            return false;
        }
    }
    if (mixture.active_count > 0U && mixture.cdf[mixture.active_count] != 1.0F) {
        return false;
    }
    for (auto index = mixture.active_count; index < MaximumClosureCount; ++index) {
        if (!zero_closure_record(mixture.closures[index]) || mixture.probabilities[index] != 0.0F ||
            mixture.cdf[index + 1U] != 0.0F) {
            return false;
        }
    }
    return mixture.active_count != 1U || mixture.probabilities[0U] == 1.0F;
}

__host__ __device__ inline void conductor_parameters(const ClosureRecord& closure,
                                                     shared::TransportSpectrum& eta,
                                                     shared::TransportSpectrum& k) noexcept {
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        eta.values[lane] = closure.parameters[lane];
        k.values[lane] = closure.parameters[SpectrumLaneCount + lane];
    }
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
eval_closure_record(const ClosureRecord& closure, const Vector3 outgoing_local,
                    const Vector3 incoming_local, const TransportMode mode) noexcept {
    if (!valid_closure_record(closure)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    const auto weight = record_weight(closure);
    switch (closure.kind) {
    case ClosureKind::lambertian_reflection:
        return lambert_eval(weight, outgoing_local, incoming_local);
    case ClosureKind::rough_diffuse_reflection:
        return rough_diffuse_eval(weight, closure.parameters[0U], outgoing_local, incoming_local);
    case ClosureKind::rough_conductor_reflection: {
        auto eta = shared::TransportSpectrum{};
        auto k = shared::TransportSpectrum{};
        conductor_parameters(closure, eta, k);
        return rough_conductor_eval(weight, eta, k, closure.parameters[8U], closure.parameters[9U],
                                    outgoing_local, incoming_local);
    }
    case ClosureKind::rough_dielectric:
        return rough_dielectric_eval(weight, closure.parameters[0U], closure.parameters[1U],
                                     closure.parameters[2U], closure.parameters[3U], outgoing_local,
                                     incoming_local, mode);
    case ClosureKind::specular_reflection:
        return specular_reflection_eval(weight, outgoing_local, incoming_local);
    case ClosureKind::specular_transmission:
        return specular_transmission_eval(weight, closure.parameters[0U], closure.parameters[1U],
                                          outgoing_local, incoming_local, mode);
    case ClosureKind::none:
        break;
    }
    return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
pdf_closure_record(const ClosureRecord& closure, const Vector3 outgoing_local,
                   const Vector3 incoming_local, const TransportMode mode) noexcept {
    if (!valid_closure_record(closure)) {
        return ProbabilityResult{
            .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    const auto weight = record_weight(closure);
    switch (closure.kind) {
    case ClosureKind::lambertian_reflection:
        return lambert_pdf(outgoing_local, incoming_local);
    case ClosureKind::rough_diffuse_reflection:
        return rough_diffuse_pdf(closure.parameters[0U], outgoing_local, incoming_local);
    case ClosureKind::rough_conductor_reflection:
        return rough_conductor_pdf(closure.parameters[8U], closure.parameters[9U], outgoing_local,
                                   incoming_local);
    case ClosureKind::rough_dielectric:
        return rough_dielectric_pdf(weight, closure.parameters[0U], closure.parameters[1U],
                                    closure.parameters[2U], closure.parameters[3U], outgoing_local,
                                    incoming_local, mode);
    case ClosureKind::specular_reflection:
        return specular_reflection_pdf(weight, outgoing_local, incoming_local);
    case ClosureKind::specular_transmission:
        return specular_transmission_pdf(weight, closure.parameters[0U], closure.parameters[1U],
                                         outgoing_local, incoming_local, mode);
    case ClosureKind::none:
        break;
    }
    return ProbabilityResult{
        .value = probability_density(0.0F, ProbabilityMeasure::solid_angle),
        .status = Status::invalid_argument,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult
sample_closure_record(const ClosureRecord& closure, const Vector3 outgoing_local,
                      const float event_sample, const float canonical_u, const float canonical_v,
                      const TransportMode mode) noexcept {
    if (!valid_closure_record(closure)) {
        return empty_lobe_sample(Status::invalid_argument);
    }
    const auto weight = record_weight(closure);
    switch (closure.kind) {
    case ClosureKind::lambertian_reflection: {
        const auto sampled = sample_lambert(weight, outgoing_local, canonical_u, canonical_v);
        if (!succeeded(sampled.status)) {
            return empty_lobe_sample(sampled.status);
        }
        return LobeSampleResult{
            .value = sampled.value,
            .incoming_local = sampled.incoming_local,
            .probability = sampled.probability,
            .lobes = closure.lobes,
            .eta_scale_multiplier = 1.0F,
            .status = Status::success,
            .reserved = {},
        };
    }
    case ClosureKind::rough_diffuse_reflection:
        return sample_rough_diffuse(weight, closure.parameters[0U], outgoing_local, canonical_u,
                                    canonical_v);
    case ClosureKind::rough_conductor_reflection: {
        auto eta = shared::TransportSpectrum{};
        auto k = shared::TransportSpectrum{};
        conductor_parameters(closure, eta, k);
        return sample_rough_conductor(weight, eta, k, closure.parameters[8U],
                                      closure.parameters[9U], outgoing_local, canonical_u,
                                      canonical_v);
    }
    case ClosureKind::rough_dielectric:
        return sample_rough_dielectric(
            weight, closure.parameters[0U], closure.parameters[1U], closure.parameters[2U],
            closure.parameters[3U], outgoing_local, event_sample, canonical_u, canonical_v, mode);
    case ClosureKind::specular_reflection:
        return sample_specular_reflection(weight, outgoing_local);
    case ClosureKind::specular_transmission:
        return sample_specular_transmission(weight, closure.parameters[0U], closure.parameters[1U],
                                            outgoing_local, mode);
    case ClosureKind::none:
        break;
    }
    return empty_lobe_sample(Status::invalid_argument);
}

[[nodiscard]] __host__ __device__ inline SpectrumResult eval_closure_record_with_direction_mask(
    const ClosureRecord& closure, const ScatteringLobe allowed_directions,
    const Vector3 outgoing_local, const Vector3 incoming_local, const TransportMode mode) noexcept {
    if (closure.kind != ClosureKind::rough_dielectric) {
        return eval_closure_record(closure, outgoing_local, incoming_local, mode);
    }
    return rough_dielectric_eval_with_direction_mask(record_weight(closure), closure.parameters[0U],
                                                     closure.parameters[1U], closure.parameters[2U],
                                                     closure.parameters[3U], outgoing_local,
                                                     incoming_local, mode, allowed_directions);
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult pdf_closure_record_with_direction_mask(
    const ClosureRecord& closure, const ScatteringLobe allowed_directions,
    const Vector3 outgoing_local, const Vector3 incoming_local, const TransportMode mode) noexcept {
    if (closure.kind != ClosureKind::rough_dielectric) {
        return pdf_closure_record(closure, outgoing_local, incoming_local, mode);
    }
    return rough_dielectric_pdf_with_direction_mask(record_weight(closure), closure.parameters[0U],
                                                    closure.parameters[1U], closure.parameters[2U],
                                                    closure.parameters[3U], outgoing_local,
                                                    incoming_local, mode, allowed_directions);
}

[[nodiscard]] __host__ __device__ inline LobeSampleResult sample_closure_record_with_direction_mask(
    const ClosureRecord& closure, const ScatteringLobe allowed_directions,
    const Vector3 outgoing_local, const float event_sample, const float canonical_u,
    const float canonical_v, const TransportMode mode) noexcept {
    if (closure.kind != ClosureKind::rough_dielectric) {
        return sample_closure_record(closure, outgoing_local, event_sample, canonical_u,
                                     canonical_v, mode);
    }
    return sample_rough_dielectric_with_direction_mask(
        record_weight(closure), closure.parameters[0U], closure.parameters[1U],
        closure.parameters[2U], closure.parameters[3U], outgoing_local, event_sample, canonical_u,
        canonical_v, mode, allowed_directions);
}

[[nodiscard]] __host__ __device__ inline SpectrumResult
closure_mixture_eval(const ClosureMixtureRecord& mixture, const Vector3 outgoing_local,
                     const Vector3 incoming_local, const TransportMode mode) noexcept {
    if (!valid_closure_mixture_record(mixture) || !known_transport_mode(mode) ||
        !unit_vector(outgoing_local) || !unit_vector(incoming_local)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    auto accumulated = shared::TransportSpectrum{};
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        const auto component =
            eval_closure_record(mixture.closures[index], outgoing_local, incoming_local, mode);
        if (!succeeded(component.status)) {
            return SpectrumResult{.value = {}, .status = component.status, .reserved = {}};
        }
        const auto sum = checked_accumulate(accumulated, component.value);
        if (!succeeded(sum.status)) {
            return SpectrumResult{.value = {}, .status = sum.status, .reserved = {}};
        }
        accumulated = sum.value;
    }
    return SpectrumResult{.value = accumulated, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline bool compensated_add(const float value, float& sum,
                                                              float& correction) noexcept {
    if (!isfinite(value) || value < 0.0F) {
        return false;
    }
    const auto next = sum + value;
    if (!isfinite(next)) {
        return false;
    }
    correction += sum >= value ? (sum - next) + value : (value - next) + sum;
    if (!isfinite(correction)) {
        return false;
    }
    sum = next;
    return true;
}

[[nodiscard]] __host__ __device__ inline bool
valid_normalized_mixture_probability(const float probability) noexcept {
    return isfinite(probability) && probability > 0.0F && probability <= 1.0F;
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult
closure_mixture_pdf(const ClosureMixtureRecord& mixture, const Vector3 outgoing_local,
                    const Vector3 incoming_local, const TransportMode mode) noexcept {
    const auto zero = probability_density(0.0F, ProbabilityMeasure::solid_angle);
    if (!valid_closure_mixture_record(mixture) || !known_transport_mode(mode) ||
        !unit_vector(outgoing_local) || !unit_vector(incoming_local)) {
        return ProbabilityResult{.value = zero, .status = Status::invalid_argument, .reserved = {}};
    }
    if (mixture.active_count == 0U) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    float conditional[MaximumClosureCount]{};
    auto maximum = 0.0F;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        const auto component =
            pdf_closure_record(mixture.closures[index], outgoing_local, incoming_local, mode);
        if (!succeeded(component.status) ||
            component.value.measure != ProbabilityMeasure::solid_angle ||
            !isfinite(component.value.value) || component.value.value < 0.0F) {
            return ProbabilityResult{
                .value = zero,
                .status =
                    succeeded(component.status) ? Status::not_representable : component.status,
                .reserved = {},
            };
        }
        conditional[index] = component.value.value;
        maximum = fmaxf(maximum, component.value.value);
    }
    if (mixture.active_count == 1U) {
        return ProbabilityResult{
            .value = probability_density(conditional[0U], ProbabilityMeasure::solid_angle),
            .status = Status::success,
            .reserved = {},
        };
    }
    if (maximum == 0.0F) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    auto sum = 0.0F;
    auto correction = 0.0F;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (conditional[index] == 0.0F) {
            continue;
        }
        const auto ratio = conditional[index] / maximum;
        const auto contribution = checked_product(mixture.probabilities[index], ratio);
        if (!succeeded(contribution.status) ||
            !compensated_add(contribution.value, sum, correction)) {
            return ProbabilityResult{
                .value = zero,
                .status = succeeded(contribution.status) ? Status::not_representable
                                                         : contribution.status,
                .reserved = {},
            };
        }
    }
    const auto normalized = sum + correction;
    if (!valid_normalized_mixture_probability(normalized)) {
        return ProbabilityResult{
            .value = zero,
            .status = Status::not_representable,
            .reserved = {},
        };
    }
    const auto mixed = checked_product(normalized, maximum);
    if (!succeeded(mixed.status)) {
        return ProbabilityResult{.value = zero, .status = mixed.status, .reserved = {}};
    }
    return ProbabilityResult{
        .value = probability_density(mixed.value, ProbabilityMeasure::solid_angle),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ constexpr bool
is_delta_closure_kind(const ClosureKind kind) noexcept {
    return kind == ClosureKind::specular_reflection || kind == ClosureKind::specular_transmission;
}

struct DeltaAtomResult final {
    shared::TransportSpectrum value;
    ProbabilityDensity probability;
    Status status;
    std::uint8_t reserved[3U];
};

[[nodiscard]] __host__ __device__ inline DeltaAtomResult
aggregate_delta_atom(const ClosureMixtureRecord& mixture, const Vector3 outgoing_local,
                     const Vector3 incoming_local, const TransportMode mode) noexcept {
    auto value_sums = shared::TransportSpectrum{};
    auto value_corrections = shared::TransportSpectrum{};
    auto probability_sum = 0.0F;
    auto probability_correction = 0.0F;
    auto matching_count = std::uint32_t{};
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (!is_delta_closure_kind(mixture.closures[index].kind)) {
            continue;
        }
        const auto candidate =
            sample_closure_record(mixture.closures[index], outgoing_local, 0.0F, 0.0F, 0.0F, mode);
        if (candidate.status == Status::outside_support) {
            continue;
        }
        if (!succeeded(candidate.status)) {
            return DeltaAtomResult{
                .probability = probability_density(0.0F, ProbabilityMeasure::discrete),
                .status = candidate.status,
                .reserved = {},
            };
        }
        if (candidate.incoming_local.x != incoming_local.x ||
            candidate.incoming_local.y != incoming_local.y ||
            candidate.incoming_local.z != incoming_local.z) {
            continue;
        }
        if (candidate.probability.measure != ProbabilityMeasure::discrete ||
            !isfinite(candidate.probability.value) || !(candidate.probability.value > 0.0F) ||
            candidate.probability.value > 1.0F) {
            return DeltaAtomResult{
                .probability = probability_density(0.0F, ProbabilityMeasure::discrete),
                .status = Status::not_representable,
                .reserved = {},
            };
        }
        for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
            if (!compensated_add(candidate.value.values[lane], value_sums.values[lane],
                                 value_corrections.values[lane])) {
                return DeltaAtomResult{
                    .probability = probability_density(0.0F, ProbabilityMeasure::discrete),
                    .status = Status::not_representable,
                    .reserved = {},
                };
            }
        }
        const auto weighted_probability =
            checked_product(mixture.probabilities[index], candidate.probability.value);
        if (!succeeded(weighted_probability.status) ||
            !compensated_add(weighted_probability.value, probability_sum, probability_correction)) {
            return DeltaAtomResult{
                .probability = probability_density(0.0F, ProbabilityMeasure::discrete),
                .status = succeeded(weighted_probability.status) ? Status::not_representable
                                                                 : weighted_probability.status,
                .reserved = {},
            };
        }
        ++matching_count;
    }
    const auto probability = probability_sum + probability_correction;
    if (matching_count == 0U || !isfinite(probability) || !(probability > 0.0F) ||
        probability > 1.0F) {
        return DeltaAtomResult{
            .probability = probability_density(0.0F, ProbabilityMeasure::discrete),
            .status = Status::not_representable,
            .reserved = {},
        };
    }
    auto value = shared::TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < SpectrumLaneCount; ++lane) {
        value.values[lane] = value_sums.values[lane] + value_corrections.values[lane];
        if (!isfinite(value.values[lane]) || value.values[lane] < 0.0F) {
            return DeltaAtomResult{
                .probability = probability_density(0.0F, ProbabilityMeasure::discrete),
                .status = Status::not_representable,
                .reserved = {},
            };
        }
    }
    return DeltaAtomResult{
        .value = value,
        .probability = probability_density(probability, ProbabilityMeasure::discrete),
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline ClosureMixtureSampleResult
sample_closure_mixture(const ClosureMixtureRecord& mixture, const Vector3 outgoing_local,
                       const float component_sample, const float canonical_u,
                       const float canonical_v, const TransportMode mode) noexcept {
    if (!valid_closure_mixture_record(mixture) || !known_transport_mode(mode) ||
        !unit_vector(outgoing_local) || !isfinite(component_sample) || component_sample < 0.0F ||
        !(component_sample < 1.0F) || !isfinite(canonical_u) || canonical_u < 0.0F ||
        !(canonical_u < 1.0F) || !isfinite(canonical_v) || canonical_v < 0.0F ||
        !(canonical_v < 1.0F)) {
        return empty_mixture_sample(Status::invalid_argument);
    }
    if (mixture.active_count == 0U) {
        return empty_mixture_sample(Status::outside_support);
    }
    auto selected = mixture.active_count;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (component_sample < mixture.cdf[index + 1U]) {
            selected = index;
            break;
        }
    }
    if (selected == mixture.active_count) {
        return empty_mixture_sample(Status::not_representable);
    }
    const auto local_sample =
        (component_sample - mixture.cdf[selected]) / mixture.probabilities[selected];
    if (!isfinite(local_sample) || local_sample < 0.0F || !(local_sample < 1.0F)) {
        return empty_mixture_sample(Status::not_representable);
    }
    const auto sampled = sample_closure_record(mixture.closures[selected], outgoing_local,
                                               local_sample, canonical_u, canonical_v, mode);
    if (!succeeded(sampled.status)) {
        return empty_mixture_sample(sampled.status);
    }
    if (mixture.active_count == 1U) {
        return ClosureMixtureSampleResult{
            .value = sampled.value,
            .incoming_local = sampled.incoming_local,
            .probability = sampled.probability,
            .selection_probability = probability_density(1.0F, ProbabilityMeasure::discrete),
            .selected_closure = 0U,
            .lobes = sampled.lobes,
            .eta_scale_multiplier = sampled.eta_scale_multiplier,
            .status = Status::success,
            .reserved = {},
        };
    }

    auto value = shared::TransportSpectrum{};
    auto probability = probability_density(0.0F, sampled.probability.measure);
    if (sampled.probability.measure == ProbabilityMeasure::solid_angle) {
        const auto mixed_value =
            closure_mixture_eval(mixture, outgoing_local, sampled.incoming_local, mode);
        const auto mixed_probability =
            closure_mixture_pdf(mixture, outgoing_local, sampled.incoming_local, mode);
        if (!succeeded(mixed_value.status) || !succeeded(mixed_probability.status)) {
            return empty_mixture_sample(!succeeded(mixed_value.status) ? mixed_value.status
                                                                       : mixed_probability.status);
        }
        value = mixed_value.value;
        probability = mixed_probability.value;
    } else if (sampled.probability.measure == ProbabilityMeasure::discrete) {
        const auto atom =
            aggregate_delta_atom(mixture, outgoing_local, sampled.incoming_local, mode);
        if (!succeeded(atom.status)) {
            return empty_mixture_sample(atom.status);
        }
        value = atom.value;
        probability = atom.probability;
    } else {
        return empty_mixture_sample(Status::not_representable);
    }
    return ClosureMixtureSampleResult{
        .value = value,
        .incoming_local = sampled.incoming_local,
        .probability = probability,
        .selection_probability =
            probability_density(mixture.probabilities[selected], ProbabilityMeasure::discrete),
        .selected_closure = selected,
        .lobes = sampled.lobes,
        .eta_scale_multiplier = sampled.eta_scale_multiplier,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
scattering_lobe_bits(const ScatteringLobe lobes) noexcept {
    return static_cast<std::uint32_t>(lobes);
}

[[nodiscard]] __host__ __device__ constexpr bool
valid_surface_scattering_event(const ScatteringLobe lobes) noexcept {
    constexpr auto family_mask = scattering_lobe_bits(ScatteringLobe::diffuse) |
                                 scattering_lobe_bits(ScatteringLobe::glossy) |
                                 scattering_lobe_bits(ScatteringLobe::specular);
    constexpr auto direction_mask = scattering_lobe_bits(ScatteringDirectionMask);
    constexpr auto known_mask =
        family_mask | direction_mask | scattering_lobe_bits(ScatteringLobe::volume);
    const auto bits = scattering_lobe_bits(lobes);
    const auto family = bits & family_mask;
    const auto direction = bits & direction_mask;
    const auto one_family = family != 0U && (family & (family - 1U)) == 0U;
    const auto one_direction = direction != 0U && (direction & (direction - 1U)) == 0U;
    return (bits & ~known_mask) == 0U &&
           (bits & scattering_lobe_bits(ScatteringLobe::volume)) == 0U && one_family &&
           one_direction;
}

[[nodiscard]] __host__ __device__ inline Status
validate_depth_filter_state(const PathDepthLimitsRecord& limits,
                            const PathDepthCountersRecord& counters,
                            const std::uint32_t expected_total) noexcept {
    const auto total = static_cast<std::uint64_t>(counters.diffuse) + counters.glossy +
                       counters.specular + counters.volume;
    const auto surface_total =
        static_cast<std::uint64_t>(counters.diffuse) + counters.glossy + counters.specular;
    if (total > 0xFFFFFFFFULL) {
        return Status::not_representable;
    }
    if (total != expected_total || counters.transmission > surface_total ||
        counters.diffuse > limits.diffuse || counters.glossy > limits.glossy ||
        counters.specular > limits.specular || counters.transmission > limits.transmission ||
        counters.volume > limits.volume) {
        return Status::invalid_argument;
    }
    return Status::success;
}

struct DepthEventFilterResult final {
    ScatteringLobe blocked_lobes;
    Status status;
    std::uint8_t reserved[3U];
};

[[nodiscard]] __host__ __device__ inline DepthEventFilterResult
depth_event_filter(const PathDepthLimitsRecord& limits, const PathDepthCountersRecord& counters,
                   const ScatteringLobe event) noexcept {
    if (!valid_surface_scattering_event(event)) {
        return DepthEventFilterResult{
            .blocked_lobes = ScatteringLobe::none,
            .status = Status::invalid_argument,
            .reserved = {},
        };
    }
    auto blocked = ScatteringLobe::none;
    if (has_scattering_lobe(event, ScatteringLobe::diffuse) && counters.diffuse >= limits.diffuse) {
        blocked = blocked | ScatteringLobe::diffuse;
    }
    if (has_scattering_lobe(event, ScatteringLobe::glossy) && counters.glossy >= limits.glossy) {
        blocked = blocked | ScatteringLobe::glossy;
    }
    if (has_scattering_lobe(event, ScatteringLobe::specular) &&
        counters.specular >= limits.specular) {
        blocked = blocked | ScatteringLobe::specular;
    }
    if (has_scattering_lobe(event, ScatteringLobe::transmission) &&
        counters.transmission >= limits.transmission) {
        blocked = blocked | ScatteringLobe::transmission;
    }
    return DepthEventFilterResult{
        .blocked_lobes = blocked,
        .status = Status::success,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline bool
build_filtered_component_distribution(ClosureMixtureRecord& mixture) noexcept {
    if (mixture.active_count == 0U) {
        return true;
    }
    if (mixture.active_count == 1U) {
        mixture.probabilities[0U] = 1.0F;
        mixture.cdf[0U] = 0.0F;
        mixture.cdf[1U] = 1.0F;
        return true;
    }

    auto total_sum = 0.0F;
    auto total_correction = 0.0F;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (!compensated_add(mixture.probabilities[index], total_sum, total_correction)) {
            return false;
        }
    }
    const auto total = total_sum + total_correction;
    if (!isfinite(total) || !(total > 0.0F)) {
        return false;
    }

    constexpr auto scalar_digits = 24;
    constexpr auto probability_scale = std::uint64_t{1U} << scalar_digits;
    std::uint64_t masses[MaximumClosureCount]{};
    float remainders[MaximumClosureCount]{};
    auto base_mass = std::uint64_t{};
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        const auto normalized = mixture.probabilities[index] / total;
        if (!isfinite(normalized) || !(normalized > 0.0F) || !(normalized < 1.0F)) {
            return false;
        }
        const auto scaled = ldexpf(normalized, scalar_digits);
        const auto base = floorf(scaled);
        if (!isfinite(scaled) || !(scaled > 0.0F) || base < 0.0F ||
            base > static_cast<float>(probability_scale)) {
            return false;
        }
        masses[index] = static_cast<std::uint64_t>(base);
        remainders[index] = scaled - base;
        if (!isfinite(remainders[index]) || remainders[index] < 0.0F ||
            !(remainders[index] < 1.0F) || base_mass > probability_scale - masses[index]) {
            return false;
        }
        base_mass += masses[index];
    }
    if (base_mass > probability_scale) {
        return false;
    }

    auto remainder_sum = 0.0F;
    auto remainder_correction = 0.0F;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (!compensated_add(remainders[index], remainder_sum, remainder_correction)) {
            return false;
        }
    }
    const auto missing_mass = probability_scale - base_mass;
    const auto remainder_total = remainder_sum + remainder_correction;
    if (missing_mass > mixture.active_count ||
        fabsf(remainder_total - static_cast<float>(missing_mass)) > 0.5F + FLT_EPSILON * 16.0F) {
        return false;
    }
    bool received_remainder[MaximumClosureCount]{};
    for (auto unit = std::uint64_t{}; unit < missing_mass; ++unit) {
        auto selected = mixture.active_count;
        for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
            if (received_remainder[index] || !(remainders[index] > 0.0F)) {
                continue;
            }
            if (selected == mixture.active_count || remainders[index] > remainders[selected]) {
                selected = index;
            }
        }
        if (selected == mixture.active_count) {
            return false;
        }
        ++masses[selected];
        received_remainder[selected] = true;
    }

    auto cumulative_mass = std::uint64_t{};
    mixture.cdf[0U] = 0.0F;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (masses[index] == 0U || cumulative_mass > probability_scale - masses[index]) {
            return false;
        }
        cumulative_mass += masses[index];
        mixture.probabilities[index] = ldexpf(static_cast<float>(masses[index]), -scalar_digits);
        mixture.cdf[index + 1U] = ldexpf(static_cast<float>(cumulative_mass), -scalar_digits);
    }
    return cumulative_mass == probability_scale && mixture.cdf[mixture.active_count] == 1.0F;
}

[[nodiscard]] __host__ __device__ inline DepthFilteredClosureMixtureState
filter_closure_mixture_by_depth(ClosureMixtureRecord& mixture, const PathDepthLimitsRecord& limits,
                                const PathDepthCountersRecord& counters,
                                const std::uint32_t expected_total) noexcept {
    auto result = DepthFilteredClosureMixtureState{};
    result.source_count = mixture.active_count;
    result.status = valid_closure_mixture_record(mixture)
                        ? validate_depth_filter_state(limits, counters, expected_total)
                        : Status::invalid_argument;
    if (!succeeded(result.status)) {
        return result;
    }

    auto filtered_count = std::uint32_t{};
    for (auto source_index = std::uint32_t{}; source_index < result.source_count; ++source_index) {
        const auto source_closure = mixture.closures[source_index];
        const auto source_probability = mixture.probabilities[source_index];
        auto allowed_directions = ScatteringLobe::none;
        if (source_closure.kind == ClosureKind::rough_dielectric) {
            const auto reflection = depth_event_filter(
                limits, counters, ScatteringLobe::glossy | ScatteringLobe::reflection);
            const auto transmission = depth_event_filter(
                limits, counters, ScatteringLobe::glossy | ScatteringLobe::transmission);
            if (!succeeded(reflection.status) || !succeeded(transmission.status)) {
                result.status =
                    !succeeded(reflection.status) ? reflection.status : transmission.status;
                return result;
            }
            if (reflection.blocked_lobes == ScatteringLobe::none) {
                allowed_directions = allowed_directions | ScatteringLobe::reflection;
            } else {
                result.blocked_lobes = result.blocked_lobes | reflection.blocked_lobes;
            }
            if (transmission.blocked_lobes == ScatteringLobe::none) {
                allowed_directions = allowed_directions | ScatteringLobe::transmission;
            } else {
                result.blocked_lobes = result.blocked_lobes | transmission.blocked_lobes;
            }
        } else {
            const auto event = depth_event_filter(limits, counters, source_closure.lobes);
            if (!succeeded(event.status)) {
                result.status = event.status;
                return result;
            }
            if (event.blocked_lobes == ScatteringLobe::none) {
                allowed_directions =
                    static_cast<ScatteringLobe>(scattering_lobe_bits(source_closure.lobes) &
                                                scattering_lobe_bits(ScatteringDirectionMask));
            } else {
                result.blocked_lobes = result.blocked_lobes | event.blocked_lobes;
            }
        }
        if (allowed_directions == ScatteringLobe::none) {
            continue;
        }
        mixture.closures[filtered_count] = source_closure;
        mixture.probabilities[filtered_count] = source_probability;
        result.source_indices[filtered_count] = source_index;
        result.allowed_directions[filtered_count] = allowed_directions;
        ++filtered_count;
    }

    mixture.active_count = filtered_count;
    if (filtered_count == result.source_count) {
        return result;
    }
    for (auto index = filtered_count; index < MaximumClosureCount; ++index) {
        mixture.closures[index] = ClosureRecord{};
        mixture.probabilities[index] = 0.0F;
    }
    for (auto index = std::uint32_t{}; index <= MaximumClosureCount; ++index) {
        mixture.cdf[index] = 0.0F;
    }
    if (!build_filtered_component_distribution(mixture) || !valid_closure_mixture_record(mixture)) {
        result.status = Status::not_representable;
    }
    return result;
}

[[nodiscard]] __host__ __device__ inline bool
valid_depth_filtered_closure_mixture(const ClosureMixtureRecord& mixture,
                                     const DepthFilteredClosureMixtureState& filtered) noexcept {
    if (!succeeded(filtered.status) || !valid_closure_mixture_record(mixture) ||
        filtered.source_count > MaximumClosureCount ||
        mixture.active_count > filtered.source_count) {
        return false;
    }
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (filtered.source_indices[index] >= filtered.source_count ||
            !valid_rough_dielectric_direction_mask(filtered.allowed_directions[index])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __host__ __device__ inline SpectrumResult depth_filtered_closure_mixture_eval(
    const ClosureMixtureRecord& mixture, const DepthFilteredClosureMixtureState& filtered,
    const Vector3 outgoing_local, const Vector3 incoming_local, const TransportMode mode) noexcept {
    if (!valid_depth_filtered_closure_mixture(mixture, filtered) || !known_transport_mode(mode) ||
        !unit_vector(outgoing_local) || !unit_vector(incoming_local)) {
        return SpectrumResult{.value = {}, .status = Status::invalid_argument, .reserved = {}};
    }
    auto accumulated = shared::TransportSpectrum{};
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        const auto component = eval_closure_record_with_direction_mask(
            mixture.closures[index], filtered.allowed_directions[index], outgoing_local,
            incoming_local, mode);
        if (!succeeded(component.status)) {
            return SpectrumResult{.value = {}, .status = component.status, .reserved = {}};
        }
        const auto sum = checked_accumulate(accumulated, component.value);
        if (!succeeded(sum.status)) {
            return SpectrumResult{.value = {}, .status = sum.status, .reserved = {}};
        }
        accumulated = sum.value;
    }
    return SpectrumResult{.value = accumulated, .status = Status::success, .reserved = {}};
}

[[nodiscard]] __host__ __device__ inline ProbabilityResult depth_filtered_closure_mixture_pdf(
    const ClosureMixtureRecord& mixture, const DepthFilteredClosureMixtureState& filtered,
    const Vector3 outgoing_local, const Vector3 incoming_local, const TransportMode mode) noexcept {
    const auto zero = probability_density(0.0F, ProbabilityMeasure::solid_angle);
    if (!valid_depth_filtered_closure_mixture(mixture, filtered) || !known_transport_mode(mode) ||
        !unit_vector(outgoing_local) || !unit_vector(incoming_local)) {
        return ProbabilityResult{.value = zero, .status = Status::invalid_argument, .reserved = {}};
    }
    if (mixture.active_count == 0U) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    float conditional[MaximumClosureCount]{};
    auto maximum = 0.0F;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        const auto component = pdf_closure_record_with_direction_mask(
            mixture.closures[index], filtered.allowed_directions[index], outgoing_local,
            incoming_local, mode);
        if (!succeeded(component.status) ||
            component.value.measure != ProbabilityMeasure::solid_angle ||
            !isfinite(component.value.value) || component.value.value < 0.0F) {
            return ProbabilityResult{
                .value = zero,
                .status =
                    succeeded(component.status) ? Status::not_representable : component.status,
                .reserved = {},
            };
        }
        conditional[index] = component.value.value;
        maximum = fmaxf(maximum, component.value.value);
    }
    if (mixture.active_count == 1U) {
        return ProbabilityResult{
            .value = probability_density(conditional[0U], ProbabilityMeasure::solid_angle),
            .status = Status::success,
            .reserved = {},
        };
    }
    if (maximum == 0.0F) {
        return ProbabilityResult{.value = zero, .status = Status::success, .reserved = {}};
    }
    auto sum = 0.0F;
    auto correction = 0.0F;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        const auto ratio = conditional[index] / maximum;
        const auto contribution = checked_product(mixture.probabilities[index], ratio);
        if (!succeeded(contribution.status) ||
            !compensated_add(contribution.value, sum, correction)) {
            return ProbabilityResult{
                .value = zero,
                .status = succeeded(contribution.status) ? Status::not_representable
                                                         : contribution.status,
                .reserved = {},
            };
        }
    }
    const auto normalized = sum + correction;
    if (!valid_normalized_mixture_probability(normalized)) {
        return ProbabilityResult{
            .value = zero, .status = Status::not_representable, .reserved = {}};
    }
    const auto mixed = checked_product(normalized, maximum);
    return ProbabilityResult{
        .value = succeeded(mixed.status)
                     ? probability_density(mixed.value, ProbabilityMeasure::solid_angle)
                     : zero,
        .status = mixed.status,
        .reserved = {},
    };
}

[[nodiscard]] __host__ __device__ inline ClosureMixtureSampleResult
sample_depth_filtered_closure_mixture(const ClosureMixtureRecord& mixture,
                                      const DepthFilteredClosureMixtureState& filtered,
                                      const Vector3 outgoing_local, const float component_sample,
                                      const float canonical_u, const float canonical_v,
                                      const TransportMode mode) noexcept {
    if (!valid_depth_filtered_closure_mixture(mixture, filtered) || !known_transport_mode(mode) ||
        !unit_vector(outgoing_local) || !isfinite(component_sample) || component_sample < 0.0F ||
        !(component_sample < 1.0F) || !isfinite(canonical_u) || canonical_u < 0.0F ||
        !(canonical_u < 1.0F) || !isfinite(canonical_v) || canonical_v < 0.0F ||
        !(canonical_v < 1.0F)) {
        return empty_mixture_sample(Status::invalid_argument);
    }
    if (mixture.active_count == 0U) {
        return empty_mixture_sample(Status::outside_support);
    }
    auto selected = mixture.active_count;
    for (auto index = std::uint32_t{}; index < mixture.active_count; ++index) {
        if (component_sample < mixture.cdf[index + 1U]) {
            selected = index;
            break;
        }
    }
    if (selected == mixture.active_count) {
        return empty_mixture_sample(Status::not_representable);
    }
    const auto event_sample =
        (component_sample - mixture.cdf[selected]) / mixture.probabilities[selected];
    if (!isfinite(event_sample) || event_sample < 0.0F || !(event_sample < 1.0F)) {
        return empty_mixture_sample(Status::not_representable);
    }
    const auto sampled = sample_closure_record_with_direction_mask(
        mixture.closures[selected], filtered.allowed_directions[selected], outgoing_local,
        event_sample, canonical_u, canonical_v, mode);
    if (!succeeded(sampled.status)) {
        return empty_mixture_sample(sampled.status);
    }
    if (mixture.active_count == 1U) {
        return ClosureMixtureSampleResult{
            .value = sampled.value,
            .incoming_local = sampled.incoming_local,
            .probability = sampled.probability,
            .selection_probability = probability_density(1.0F, ProbabilityMeasure::discrete),
            .selected_closure = filtered.source_indices[0U],
            .lobes = sampled.lobes,
            .eta_scale_multiplier = sampled.eta_scale_multiplier,
            .status = Status::success,
            .reserved = {},
        };
    }

    auto value = shared::TransportSpectrum{};
    auto probability = probability_density(0.0F, sampled.probability.measure);
    if (sampled.probability.measure == ProbabilityMeasure::solid_angle) {
        const auto mixed_value = depth_filtered_closure_mixture_eval(
            mixture, filtered, outgoing_local, sampled.incoming_local, mode);
        const auto mixed_probability = depth_filtered_closure_mixture_pdf(
            mixture, filtered, outgoing_local, sampled.incoming_local, mode);
        if (!succeeded(mixed_value.status) || !succeeded(mixed_probability.status)) {
            return empty_mixture_sample(!succeeded(mixed_value.status) ? mixed_value.status
                                                                       : mixed_probability.status);
        }
        value = mixed_value.value;
        probability = mixed_probability.value;
    } else if (sampled.probability.measure == ProbabilityMeasure::discrete) {
        const auto atom =
            aggregate_delta_atom(mixture, outgoing_local, sampled.incoming_local, mode);
        if (!succeeded(atom.status)) {
            return empty_mixture_sample(atom.status);
        }
        value = atom.value;
        probability = atom.probability;
    } else {
        return empty_mixture_sample(Status::not_representable);
    }
    return ClosureMixtureSampleResult{
        .value = value,
        .incoming_local = sampled.incoming_local,
        .probability = probability,
        .selection_probability =
            probability_density(mixture.probabilities[selected], ProbabilityMeasure::discrete),
        .selected_closure = filtered.source_indices[selected],
        .lobes = sampled.lobes,
        .eta_scale_multiplier = sampled.eta_scale_multiplier,
        .status = Status::success,
        .reserved = {},
    };
}

static_assert(MaximumClosureCount == 8U);
static_assert(ClosureParameterScalarCount == 10U);
static_assert(sizeof(TransportMode) == sizeof(std::uint8_t));
static_assert(sizeof(ScatteringLobe) == sizeof(std::uint32_t));
static_assert(sizeof(ClosureKind) == sizeof(std::uint32_t));
static_assert(static_cast<std::uint32_t>(ClosureKind::none) == 0U);
static_assert(static_cast<std::uint32_t>(ClosureKind::lambertian_reflection) == 1U);
static_assert(static_cast<std::uint32_t>(ClosureKind::rough_diffuse_reflection) == 2U);
static_assert(static_cast<std::uint32_t>(ClosureKind::rough_conductor_reflection) == 3U);
static_assert(static_cast<std::uint32_t>(ClosureKind::rough_dielectric) == 4U);
static_assert(static_cast<std::uint32_t>(ClosureKind::specular_reflection) == 5U);
static_assert(static_cast<std::uint32_t>(ClosureKind::specular_transmission) == 6U);
static_assert(std::is_standard_layout_v<ClosureRecord>);
static_assert(std::is_trivially_copyable_v<ClosureRecord>);
static_assert(std::is_trivially_destructible_v<ClosureRecord>);
static_assert(sizeof(ClosureRecord) == 64U);
static_assert(alignof(ClosureRecord) == 16U);
static_assert(offsetof(ClosureRecord, kind) == 0U);
static_assert(offsetof(ClosureRecord, lobes) == 4U);
static_assert(offsetof(ClosureRecord, weight) == 8U);
static_assert(offsetof(ClosureRecord, parameters) == 24U);
static_assert(std::is_standard_layout_v<ClosureMixtureRecord>);
static_assert(std::is_trivially_copyable_v<ClosureMixtureRecord>);
static_assert(std::is_trivially_destructible_v<ClosureMixtureRecord>);
static_assert(sizeof(ClosureMixtureRecord) == 608U);
static_assert(alignof(ClosureMixtureRecord) == 16U);
static_assert(std::is_standard_layout_v<LobeSampleResult>);
static_assert(std::is_trivially_copyable_v<LobeSampleResult>);
static_assert(std::is_standard_layout_v<VisibleNormalSampleResult>);
static_assert(std::is_trivially_copyable_v<VisibleNormalSampleResult>);
static_assert(std::is_standard_layout_v<ClosureMixtureSampleResult>);
static_assert(std::is_trivially_copyable_v<ClosureMixtureSampleResult>);
static_assert(std::is_standard_layout_v<PathDepthLimitsRecord>);
static_assert(std::is_trivially_copyable_v<PathDepthLimitsRecord>);
static_assert(sizeof(PathDepthLimitsRecord) == 20U);
static_assert(std::is_standard_layout_v<PathDepthCountersRecord>);
static_assert(std::is_trivially_copyable_v<PathDepthCountersRecord>);
static_assert(sizeof(PathDepthCountersRecord) == 20U);
static_assert(std::is_standard_layout_v<DepthFilteredClosureMixtureState>);
static_assert(std::is_trivially_copyable_v<DepthFilteredClosureMixtureState>);
static_assert(sizeof(DepthFilteredClosureMixtureState) == 76U);
static_assert(std::is_standard_layout_v<DepthEventFilterResult>);
static_assert(std::is_trivially_copyable_v<DepthEventFilterResult>);
static_assert(sizeof(DepthEventFilterResult) == 8U);
static_assert(std::is_standard_layout_v<DeltaAtomResult>);
static_assert(std::is_trivially_copyable_v<DeltaAtomResult>);
static_assert(std::is_standard_layout_v<SmithTerms>);
static_assert(std::is_trivially_copyable_v<SmithTerms>);
static_assert(std::is_standard_layout_v<ReflectionGeometry>);
static_assert(std::is_trivially_copyable_v<ReflectionGeometry>);
static_assert(std::is_standard_layout_v<DielectricInterface>);
static_assert(std::is_trivially_copyable_v<DielectricInterface>);
static_assert(std::is_standard_layout_v<TransmissionGeometry>);
static_assert(std::is_trivially_copyable_v<TransmissionGeometry>);

} // namespace blackframe::xpu::cuda::transport_device
