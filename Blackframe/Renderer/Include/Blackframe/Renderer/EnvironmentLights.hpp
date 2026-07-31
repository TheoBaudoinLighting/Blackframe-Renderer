#pragma once

#include <Blackframe/Renderer/Detail/PacketLightSpectrum.hpp>
#include <Blackframe/Renderer/Light.hpp>
#include <Blackframe/Renderer/Quaternion.hpp>
#include <Blackframe/Renderer/Transforms.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace environment_light_detail {

[[nodiscard]] inline core::Error invalid_environment_light(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

[[nodiscard]] inline core::Error exhausted_environment_light(const char* const message) {
    return core::Error{
        .code = core::StatusCode::resource_exhausted,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_canonical_sample(const Point2T<Scalar> sample) {
    if (!std::isfinite(sample.x) || !std::isfinite(sample.y) || sample.x < Scalar{0} ||
        sample.x >= Scalar{1} || sample.y < Scalar{0} || sample.y >= Scalar{1}) {
        return std::unexpected(invalid_environment_light(
            "Environment-map sampling requires a finite canonical sample in [0, 1)^2."));
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool representable_open_interval(const Scalar lower, const Scalar upper) noexcept {
    if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper)) {
        return false;
    }
    const auto midpoint = std::midpoint(lower, upper);
    return midpoint > lower && midpoint < upper;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::size_t> checked_texel_count(const std::uint32_t width,
                                                            const std::uint32_t height) {
    if (width == 0U || height == 0U) {
        return std::unexpected(
            invalid_environment_light("An environment map requires non-zero width and height."));
    }

    constexpr auto scalar_digits = std::numeric_limits<Scalar>::digits;
    constexpr auto maximum_exact_index = scalar_digits >= 63
                                             ? std::numeric_limits<std::uint64_t>::max()
                                             : (std::uint64_t{1} << scalar_digits);
    if (static_cast<std::uint64_t>(width) > maximum_exact_index ||
        static_cast<std::uint64_t>(height) > maximum_exact_index) {
        return std::unexpected(invalid_environment_light(
            "Environment-map dimensions are not exactly indexable in the requested precision."));
    }

    const auto wide_width = static_cast<std::uint64_t>(width);
    const auto wide_height = static_cast<std::uint64_t>(height);
    const auto maximum_host_index =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (wide_width >= maximum_host_index || wide_height >= maximum_host_index) {
        return std::unexpected(exhausted_environment_light(
            "Environment-map angular partitions exceed the host index domain."));
    }
    if (wide_height > std::numeric_limits<std::uint64_t>::max() / wide_width) {
        return std::unexpected(exhausted_environment_light(
            "Environment-map dimensions exceed the supported texel domain."));
    }
    const auto wide_count = wide_width * wide_height;
    if (wide_count > std::numeric_limits<std::size_t>::max()) {
        return std::unexpected(exhausted_environment_light(
            "Environment-map dimensions exceed the host index domain."));
    }
    return static_cast<std::size_t>(wide_count);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>>
robust_unit_direction(const Vector3T<Scalar> direction, const char* const error_message) {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    const auto scale =
        std::max({std::abs(direction.x), std::abs(direction.y), std::abs(direction.z)});
    if (!(scale > Scalar{0}) || !std::isfinite(scale)) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    const auto scaled = direction / scale;
    const auto squared_length =
        std::fma(scaled.x, scaled.x, std::fma(scaled.y, scaled.y, scaled.z * scaled.z));
    const auto length = std::sqrt(squared_length);
    if (!std::isfinite(squared_length) || !(squared_length > Scalar{0}) || !std::isfinite(length) ||
        !(length > Scalar{0})) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    const auto result = scaled / length;
    if (!light_detail::unit_direction(result)) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    return result;
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
[[nodiscard]] core::Result<Scalar> compensated_sum(const std::span<const Scalar> values,
                                                   const char* const error_message) {
    auto sum = Scalar{0};
    auto correction = Scalar{0};
    for (const auto value : values) {
        if (!add_compensated(value, sum, correction)) {
            return std::unexpected(invalid_environment_light(error_message));
        }
    }
    const auto total = sum + correction;
    if (!std::isfinite(total) || total < Scalar{0}) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    return total == Scalar{0} ? Scalar{0} : total;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<bool> build_cdf(const std::span<const Scalar> weights,
                                           const std::span<Scalar> cdf,
                                           const char* const error_message) {
    if (weights.empty() || weights.size() != cdf.size()) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    const auto total = compensated_sum(weights, error_message);
    if (!total) {
        return std::unexpected(total.error());
    }
    if (*total == Scalar{0}) {
        std::fill(cdf.begin(), cdf.end(), Scalar{0});
        return false;
    }

    auto last_positive = weights.size();
    for (auto index = weights.size(); index > 0U; --index) {
        if (weights[index - 1U] > Scalar{0}) {
            last_positive = index - 1U;
            break;
        }
    }
    if (last_positive == weights.size()) {
        return std::unexpected(invalid_environment_light(error_message));
    }

    auto cumulative = Scalar{0};
    for (auto index = std::size_t{0}; index < weights.size(); ++index) {
        const auto weight = weights[index];
        if (!std::isfinite(weight) || weight < Scalar{0}) {
            return std::unexpected(invalid_environment_light(error_message));
        }
        if (weight == Scalar{0}) {
            cdf[index] = cumulative;
            continue;
        }

        const auto probability = weight / *total;
        const auto candidate = cumulative + probability;
        if (!std::isfinite(probability) || !(probability > Scalar{0}) ||
            !std::isfinite(candidate) || !(candidate > cumulative)) {
            return std::unexpected(invalid_environment_light(error_message));
        }
        auto next = Scalar{1};
        if (index != last_positive) {
            if (!(candidate < Scalar{1})) {
                return std::unexpected(invalid_environment_light(error_message));
            }
            next = candidate;
        }
        cdf[index] = next;
        cumulative = next;
    }
    if (cdf.back() != Scalar{1}) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    return true;
}

template <SpectrumScalar Scalar> struct CdfSelectionT final {
    std::size_t index;
    Scalar probability;
    Scalar remapped;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<CdfSelectionT<Scalar>>
select_cdf_interval(const std::span<const Scalar> cdf, const Scalar canonical,
                    const char* const error_message) {
    const auto selected = std::upper_bound(cdf.begin(), cdf.end(), canonical);
    if (selected == cdf.end()) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    const auto index = static_cast<std::size_t>(selected - cdf.begin());
    const auto previous = index == 0U ? Scalar{0} : cdf[index - 1U];
    const auto probability = cdf[index] - previous;
    const auto lower_distance = canonical - previous;
    const auto upper_distance = cdf[index] - canonical;
    const auto remapped = lower_distance <= upper_distance
                              ? lower_distance / probability
                              : Scalar{1} - upper_distance / probability;
    if (!std::isfinite(probability) || !(probability > Scalar{0}) || !std::isfinite(remapped) ||
        !std::isfinite(lower_distance) || lower_distance < Scalar{0} ||
        !std::isfinite(upper_distance) || !(upper_distance > Scalar{0}) || remapped < Scalar{0} ||
        !(remapped < Scalar{1})) {
        return std::unexpected(invalid_environment_light(error_message));
    }
    return CdfSelectionT<Scalar>{
        .index = index,
        .probability = probability,
        .remapped = remapped,
    };
}

template <SpectrumScalar Scalar> struct DirectionTexelT final {
    std::size_t column;
    std::size_t row;
    std::size_t index;
};

// Local environment coordinates match Sphere UVs: +Y is north, the seam is
// +X, and longitude increases toward +Z.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<DirectionTexelT<Scalar>>
direction_texel(const Vector3T<Scalar> local_direction,
                const std::span<const Scalar> azimuth_boundaries,
                const std::span<const Scalar> cosine_boundaries) {
    if (azimuth_boundaries.size() < 2U || cosine_boundaries.size() < 2U) {
        return std::unexpected(invalid_environment_light(
            "Environment-map lookup requires explicit angular partitions."));
    }
    const auto direction = robust_unit_direction(
        local_direction, "Environment-map lookup requires a representable non-zero direction.");
    if (!direction) {
        return std::unexpected(direction.error());
    }

    const auto equatorial_radius = std::hypot(direction->x, direction->z);
    auto azimuth = Scalar{0};
    if (equatorial_radius != Scalar{0}) {
        azimuth = std::atan2(direction->z, direction->x);
        if (azimuth < Scalar{0}) {
            azimuth += azimuth_boundaries.back();
        }
    }
    if (!std::isfinite(azimuth) || azimuth < Scalar{0}) {
        return std::unexpected(invalid_environment_light(
            "Environment-map direction projection is not representable."));
    }
    if (!(azimuth < azimuth_boundaries.back())) {
        azimuth = Scalar{0};
    }

    const auto column_end =
        std::upper_bound(azimuth_boundaries.begin(), azimuth_boundaries.end(), azimuth);
    const auto row_end = std::upper_bound(cosine_boundaries.begin(), cosine_boundaries.end(),
                                          direction->y, std::greater<Scalar>{});
    const auto width = azimuth_boundaries.size() - 1U;
    const auto height = cosine_boundaries.size() - 1U;
    if (column_end == azimuth_boundaries.begin() || row_end == cosine_boundaries.begin()) {
        return std::unexpected(
            invalid_environment_light("Environment-map angular lookup escaped its partition."));
    }
    const auto column = std::min(
        static_cast<std::size_t>(column_end - azimuth_boundaries.begin()) - 1U, width - 1U);
    const auto row =
        std::min(static_cast<std::size_t>(row_end - cosine_boundaries.begin()) - 1U, height - 1U);
    return DirectionTexelT<Scalar>{
        .column = column,
        .row = row,
        .index = row * static_cast<std::size_t>(width) + column,
    };
}

template <SpectrumScalar Scalar> struct BinaryHalfExtentT final {
    Scalar fraction;
    int exponent;
    bool is_zero;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BinaryHalfExtentT<Scalar>> binary_half_extent(const Scalar lower,
                                                                         const Scalar upper) {
    if (lower == upper) {
        return BinaryHalfExtentT<Scalar>{.fraction = Scalar{0}, .exponent = 0, .is_zero = true};
    }
    const auto full_extent = upper - lower;
    auto exponent = 0;
    if (std::isfinite(full_extent)) {
        if (!(full_extent > Scalar{0})) {
            return std::unexpected(
                invalid_environment_light("Environment-light scene extent is not representable."));
        }
        return BinaryHalfExtentT<Scalar>{
            .fraction = std::frexp(full_extent, &exponent),
            .exponent = exponent - 1,
            .is_zero = false,
        };
    }
    const auto half_extent = upper / Scalar{2} - lower / Scalar{2};
    if (!std::isfinite(half_extent) || !(half_extent > Scalar{0})) {
        return std::unexpected(
            invalid_environment_light("Environment-light scene extent is not representable."));
    }
    return BinaryHalfExtentT<Scalar>{
        .fraction = std::frexp(half_extent, &exponent),
        .exponent = exponent,
        .is_zero = false,
    };
}

template <SpectrumScalar Scalar> struct BoundingSphereScaleT final {
    Scalar scaled_squared_radius;
    int radius_exponent;
    bool is_point;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BoundingSphereScaleT<Scalar>>
bounding_sphere_scale(const Bounds3T<Scalar>& scene_bounds) {
    if (scene_bounds.is_empty() || !light_detail::finite(scene_bounds.minimum()) ||
        !light_detail::finite(scene_bounds.maximum())) {
        return std::unexpected(invalid_environment_light(
            "Environment-light power requires explicit non-empty finite scene bounds."));
    }
    const auto x = binary_half_extent(scene_bounds.minimum().x, scene_bounds.maximum().x);
    const auto y = binary_half_extent(scene_bounds.minimum().y, scene_bounds.maximum().y);
    const auto z = binary_half_extent(scene_bounds.minimum().z, scene_bounds.maximum().z);
    if (!x) {
        return std::unexpected(x.error());
    }
    if (!y) {
        return std::unexpected(y.error());
    }
    if (!z) {
        return std::unexpected(z.error());
    }
    if (x->is_zero && y->is_zero && z->is_zero) {
        return BoundingSphereScaleT<Scalar>{
            .scaled_squared_radius = Scalar{0},
            .radius_exponent = 0,
            .is_point = true,
        };
    }

    auto maximum_exponent = std::numeric_limits<int>::min();
    for (const auto extent : std::array{*x, *y, *z}) {
        if (!extent.is_zero) {
            maximum_exponent = std::max(maximum_exponent, extent.exponent);
        }
    }
    const auto scaled_extent = [maximum_exponent](const BinaryHalfExtentT<Scalar> extent) {
        return extent.is_zero ? Scalar{0}
                              : std::scalbn(extent.fraction, extent.exponent - maximum_exponent);
    };
    const auto scaled_x = scaled_extent(*x);
    const auto scaled_y = scaled_extent(*y);
    const auto scaled_z = scaled_extent(*z);
    const auto squared_radius =
        std::fma(scaled_x, scaled_x, std::fma(scaled_y, scaled_y, scaled_z * scaled_z));
    if (!std::isfinite(squared_radius) || !(squared_radius > Scalar{0})) {
        return std::unexpected(
            invalid_environment_light("Environment-light bounding sphere is not representable."));
    }
    return BoundingSphereScaleT<Scalar>{
        .scaled_squared_radius = squared_radius,
        .radius_exponent = maximum_exponent,
        .is_point = false,
    };
}

template <SpectrumScalar Scalar, std::size_t FactorCount>
[[nodiscard]] core::Result<Scalar>
scale_radiometric_value(const Scalar value, const std::array<Scalar, FactorCount>& factors,
                        const long long binary_exponent) {
    if (!std::isfinite(value) || value < Scalar{0}) {
        return std::unexpected(
            invalid_environment_light("Environment-light radiometric input is not representable."));
    }
    for (const auto factor : factors) {
        if (!std::isfinite(factor) || !(factor > Scalar{0})) {
            return std::unexpected(invalid_environment_light(
                "Environment-light radiometric scale is not representable."));
        }
    }
    if (value == Scalar{0}) {
        return Scalar{0};
    }

    auto value_exponent = 0;
    auto fraction = std::frexp(value, &value_exponent);
    auto accumulated_exponent = static_cast<long long>(value_exponent) + binary_exponent;
    for (const auto factor : factors) {
        auto factor_exponent = 0;
        fraction *= std::frexp(factor, &factor_exponent);
        accumulated_exponent += factor_exponent;
        auto normalization_exponent = 0;
        fraction = std::frexp(fraction, &normalization_exponent);
        accumulated_exponent += normalization_exponent;
    }
    if (accumulated_exponent < std::numeric_limits<int>::min() ||
        accumulated_exponent > std::numeric_limits<int>::max()) {
        return std::unexpected(invalid_environment_light(
            "Environment-light radiometric exponent is not representable."));
    }
    const auto result = std::scalbn(fraction, static_cast<int>(accumulated_exponent));
    if (!std::isfinite(result) || !(result > Scalar{0})) {
        return std::unexpected(
            invalid_environment_light("Positive environment-light power is not representable."));
    }
    return result;
}

template <SpectrumScalar Scalar> struct IntegratedLaneT final {
    Scalar peak;
    Scalar normalized_solid_angle_integral;
};

} // namespace environment_light_detail

// A packet-bound, piecewise-constant latitude-longitude environment map.
// Local +Y is north, u=0 is the +X seam, and u increases toward +Z. Texels
// are importance-sampled by their mean packet radiance times their exact solid
// angle. The supplied unit quaternion rotates map-local directions into world
// space; no image loading, RGB conversion, filtering, or fallback distribution
// is hidden by this model.
template <SpectrumScalar Scalar> class EnvironmentMapLightT final {
  public:
    [[nodiscard]] static core::Result<EnvironmentMapLightT>
    create(const std::uint32_t width, const std::uint32_t height,
           std::vector<LightSpectrumT<Scalar>> texels,
           const SampledWavelengthsT<Scalar>& wavelengths,
           const QuaternionT<Scalar> environment_to_world) {
        const auto texel_count =
            environment_light_detail::checked_texel_count<Scalar>(width, height);
        if (!texel_count) {
            return std::unexpected(texel_count.error());
        }
        if (*texel_count > texels.max_size()) {
            return std::unexpected(environment_light_detail::exhausted_environment_light(
                "Environment-map dimensions exceed host container limits."));
        }
        if (texels.size() != *texel_count) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map dimensions must match the row-major texel count exactly."));
        }

        const auto wavelength_status = light_detail::validate_light_wavelength_packet(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        const auto rotation = AffineTransformT<Scalar>::rotation(environment_to_world);
        if (!rotation) {
            return std::unexpected(rotation.error());
        }

        auto global_peak = Scalar{0};
        for (const auto& texel : texels) {
            if (!light_detail::finite_non_negative(texel)) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "Environment-map texels require finite non-negative spectral lanes."));
            }
            for (const auto lane : texel.values) {
                global_peak = std::max(global_peak, lane);
            }
        }

        auto nanometers = std::array<Scalar, TransportSpectrumSampleCount>{};
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            nanometers[lane] = wavelengths[lane].nanometers;
        }

        const auto theta_step = std::numbers::pi_v<Scalar> / static_cast<Scalar>(height);
        const auto azimuth_step =
            (Scalar{2} * std::numbers::pi_v<Scalar>) / static_cast<Scalar>(width);
        if (!std::isfinite(theta_step) || !(theta_step > Scalar{0}) ||
            !std::isfinite(azimuth_step) || !(azimuth_step > Scalar{0})) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map angular resolution is not representable."));
        }

        auto importances = std::vector<Scalar>{};
        auto azimuth_boundaries = std::vector<Scalar>{};
        auto cosine_boundaries = std::vector<Scalar>{};
        auto azimuth_spans = std::vector<Scalar>{};
        auto cosine_spans = std::vector<Scalar>{};
        auto column_weights = std::vector<Scalar>{};
        auto row_weights = std::vector<Scalar>{};
        auto row_cdf = std::vector<Scalar>{};
        auto column_cdfs = std::vector<Scalar>{};
        auto directional_densities = std::vector<Scalar>{};
        try {
            importances.resize(*texel_count);
            azimuth_boundaries.resize(static_cast<std::size_t>(width) + 1U);
            cosine_boundaries.resize(static_cast<std::size_t>(height) + 1U);
            azimuth_spans.resize(width);
            cosine_spans.resize(height);
            column_weights.resize(width);
            row_weights.resize(height);
            row_cdf.resize(height);
            column_cdfs.resize(*texel_count);
            directional_densities.resize(*texel_count);
        } catch (const std::bad_alloc&) {
            return std::unexpected(environment_light_detail::exhausted_environment_light(
                "Environment-map distribution exhausted host memory."));
        } catch (const std::length_error&) {
            return std::unexpected(environment_light_detail::exhausted_environment_light(
                "Environment-map distribution exceeded host container limits."));
        }

        const auto full_azimuth = Scalar{2} * std::numbers::pi_v<Scalar>;
        azimuth_boundaries.front() = Scalar{0};
        for (auto column = std::size_t{1}; column < width; ++column) {
            const auto boundary = static_cast<Scalar>(column) * azimuth_step;
            if (!(boundary < full_azimuth) ||
                !environment_light_detail::representable_open_interval(
                    azimuth_boundaries[column - 1U], boundary)) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "Environment-map azimuth bins are not individually representable."));
            }
            azimuth_boundaries[column] = boundary;
        }
        azimuth_boundaries.back() = full_azimuth;
        if (!environment_light_detail::representable_open_interval(
                azimuth_boundaries[azimuth_boundaries.size() - 2U], full_azimuth)) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map azimuth bins are not individually representable."));
        }
        for (auto column = std::size_t{0}; column < width; ++column) {
            azimuth_spans[column] = azimuth_boundaries[column + 1U] - azimuth_boundaries[column];
        }

        cosine_boundaries.front() = Scalar{1};
        for (auto row = std::size_t{1}; row < height; ++row) {
            const auto boundary = std::cos(static_cast<Scalar>(row) * theta_step);
            if (!environment_light_detail::representable_open_interval(
                    boundary, cosine_boundaries[row - 1U])) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "Environment-map polar bins are not individually representable."));
            }
            cosine_boundaries[row] = boundary;
        }
        cosine_boundaries.back() = Scalar{-1};
        if (!environment_light_detail::representable_open_interval(
                Scalar{-1}, cosine_boundaries[cosine_boundaries.size() - 2U])) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map polar bins are not individually representable."));
        }
        for (auto row = std::size_t{0}; row < height; ++row) {
            cosine_spans[row] = cosine_boundaries[row] - cosine_boundaries[row + 1U];
        }

        if (global_peak > Scalar{0}) {
            for (auto index = std::size_t{0}; index < texels.size(); ++index) {
                auto ratios = std::array<Scalar, TransportSpectrumSampleCount>{};
                for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
                    ratios[lane] = texels[index][lane] / global_peak;
                    if (texels[index][lane] > Scalar{0} && !(ratios[lane] > Scalar{0})) {
                        return std::unexpected(environment_light_detail::invalid_environment_light(
                            "Environment-map spectral importance is not representable."));
                    }
                }
                const auto ratio_sum = environment_light_detail::compensated_sum<Scalar>(
                    ratios, "Environment-map spectral importance is not representable.");
                if (!ratio_sum) {
                    return std::unexpected(ratio_sum.error());
                }
                const auto importance = *ratio_sum / Scalar{TransportSpectrumSampleCount};
                if ((*ratio_sum > Scalar{0} && !(importance > Scalar{0})) ||
                    !std::isfinite(importance)) {
                    return std::unexpected(environment_light_detail::invalid_environment_light(
                        "Environment-map spectral importance is not representable."));
                }
                importances[index] = importance;
            }

            const auto row_width = static_cast<std::size_t>(width);
            for (auto row = std::size_t{0}; row < height; ++row) {
                for (auto column = std::size_t{0}; column < row_width; ++column) {
                    const auto index = row * row_width + column;
                    column_weights[column] = importances[index] * azimuth_spans[column];
                    if (importances[index] > Scalar{0} && (!std::isfinite(column_weights[column]) ||
                                                           !(column_weights[column] > Scalar{0}))) {
                        return std::unexpected(environment_light_detail::invalid_environment_light(
                            "Environment-map texel probability is not representable."));
                    }
                }
                const auto row_sum = environment_light_detail::compensated_sum<Scalar>(
                    column_weights, "Environment-map row importance is not representable.");
                if (!row_sum) {
                    return std::unexpected(row_sum.error());
                }
                row_weights[row] = *row_sum * cosine_spans[row];
                if (*row_sum > Scalar{0} &&
                    (!std::isfinite(row_weights[row]) || !(row_weights[row] > Scalar{0}))) {
                    return std::unexpected(environment_light_detail::invalid_environment_light(
                        "Environment-map row probability is not representable."));
                }

                auto row_columns = std::span<Scalar>{
                    column_cdfs.data() + row * row_width,
                    row_width,
                };
                const auto built = environment_light_detail::build_cdf<Scalar>(
                    column_weights, row_columns,
                    "Environment-map conditional CDF is not representable.");
                if (!built) {
                    return std::unexpected(built.error());
                }
                if ((*row_sum > Scalar{0}) != *built) {
                    return std::unexpected(environment_light_detail::invalid_environment_light(
                        "Environment-map conditional CDF lost positive support."));
                }
            }

            const auto rows_built = environment_light_detail::build_cdf<Scalar>(
                row_weights, row_cdf, "Environment-map marginal CDF is not representable.");
            if (!rows_built) {
                return std::unexpected(rows_built.error());
            }
            if (!*rows_built) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "A non-black environment map requires a positive sampling distribution."));
            }

            for (auto row = std::size_t{0}; row < height; ++row) {
                const auto row_previous = row == 0U ? Scalar{0} : row_cdf[row - 1U];
                const auto row_probability = row_cdf[row] - row_previous;
                for (auto column = std::size_t{0}; column < width; ++column) {
                    const auto index = row * static_cast<std::size_t>(width) + column;
                    if (importances[index] == Scalar{0}) {
                        directional_densities[index] = Scalar{0};
                        continue;
                    }
                    const auto column_previous = column == 0U ? Scalar{0} : column_cdfs[index - 1U];
                    const auto column_probability = column_cdfs[index] - column_previous;
                    const auto texel_probability = row_probability * column_probability;
                    const auto solid_angle = cosine_spans[row] * azimuth_spans[column];
                    const auto density = texel_probability / solid_angle;
                    if (!std::isfinite(row_probability) || !(row_probability > Scalar{0}) ||
                        !std::isfinite(column_probability) || !(column_probability > Scalar{0}) ||
                        !std::isfinite(texel_probability) || !(texel_probability > Scalar{0}) ||
                        !std::isfinite(solid_angle) || !(solid_angle > Scalar{0}) ||
                        !std::isfinite(density) || !(density > Scalar{0})) {
                        return std::unexpected(environment_light_detail::invalid_environment_light(
                            "Environment-map directional PDF is not representable."));
                    }
                    directional_densities[index] = density;
                }
            }
        }

        auto integrated_lanes = std::array<environment_light_detail::IntegratedLaneT<Scalar>,
                                           TransportSpectrumSampleCount>{};
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            auto lane_peak = Scalar{0};
            for (const auto& texel : texels) {
                lane_peak = std::max(lane_peak, texel[lane]);
            }
            integrated_lanes[lane].peak = lane_peak;
            if (lane_peak == Scalar{0}) {
                continue;
            }

            auto integral_sum = Scalar{0};
            auto integral_correction = Scalar{0};
            for (auto row = std::size_t{0}; row < height; ++row) {
                for (auto column = std::size_t{0}; column < width; ++column) {
                    const auto index = row * static_cast<std::size_t>(width) + column;
                    const auto normalized_value = texels[index][lane] / lane_peak;
                    const auto solid_angle = cosine_spans[row] * azimuth_spans[column];
                    const auto contribution = normalized_value * solid_angle;
                    if (texels[index][lane] > Scalar{0} &&
                        (!(normalized_value > Scalar{0}) || !std::isfinite(solid_angle) ||
                         !(solid_angle > Scalar{0}) || !(contribution > Scalar{0}))) {
                        return std::unexpected(environment_light_detail::invalid_environment_light(
                            "Environment-map spectral integral is not representable."));
                    }
                    if (!environment_light_detail::add_compensated(contribution, integral_sum,
                                                                   integral_correction)) {
                        return std::unexpected(environment_light_detail::invalid_environment_light(
                            "Environment-map spectral integral is not representable."));
                    }
                }
            }
            const auto normalized_integral = integral_sum + integral_correction;
            if (!std::isfinite(normalized_integral) || !(normalized_integral > Scalar{0})) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "Environment-map spectral integral is not representable."));
            }
            integrated_lanes[lane].normalized_solid_angle_integral = normalized_integral;
        }

        return EnvironmentMapLightT{
            width,
            height,
            std::move(texels),
            nanometers,
            environment_to_world,
            *rotation,
            std::move(azimuth_boundaries),
            std::move(cosine_boundaries),
            std::move(row_cdf),
            std::move(column_cdfs),
            std::move(directional_densities),
            integrated_lanes,
            global_peak == Scalar{0},
        };
    }

    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>&, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto sample_status =
            environment_light_detail::validate_canonical_sample(canonical_sample);
        if (!sample_status) {
            return std::unexpected(sample_status.error());
        }
        const auto wavelength_status = validate_wavelengths(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        if (is_black_) {
            return std::optional<IncidentLightSampleT<Scalar>>{};
        }

        const auto row_selection = environment_light_detail::select_cdf_interval<Scalar>(
            row_cdf_, canonical_sample.y, "Environment-map marginal sampling escaped its CDF.");
        if (!row_selection) {
            return std::unexpected(row_selection.error());
        }
        const auto row_width = static_cast<std::size_t>(width_);
        const auto row_columns = std::span<const Scalar>{
            column_cdfs_.data() + row_selection->index * row_width,
            row_width,
        };
        const auto column_selection = environment_light_detail::select_cdf_interval<Scalar>(
            row_columns, canonical_sample.x,
            "Environment-map conditional sampling escaped its CDF.");
        if (!column_selection) {
            return std::unexpected(column_selection.error());
        }
        const auto index = row_selection->index * row_width + column_selection->index;

        auto row_interior_steps = std::size_t{0};
        auto column_interior_steps = std::size_t{0};
        auto stable_world_direction = std::optional<Vector3T<Scalar>>{};
        constexpr auto maximum_interior_steps =
            static_cast<std::size_t>(std::numeric_limits<Scalar>::digits);
        for (auto attempt = std::size_t{0}; attempt <= maximum_interior_steps * 2U; ++attempt) {
            const auto local_direction = sample_local_direction(
                row_selection->index, column_selection->index, row_selection->remapped,
                column_selection->remapped, row_interior_steps, column_interior_steps);
            if (!local_direction) {
                return std::unexpected(local_direction.error());
            }
            const auto rotated = environment_to_world_transform_.apply(*local_direction);
            const auto world_direction = environment_light_detail::robust_unit_direction(
                rotated,
                "Environment-map rotation did not produce a representable unit direction.");
            if (!world_direction) {
                return std::unexpected(world_direction.error());
            }
            const auto replayed = world_direction_texel(*world_direction);
            if (!replayed) {
                return std::unexpected(replayed.error());
            }
            if (replayed->index == index) {
                stable_world_direction = *world_direction;
                break;
            }

            // Keep the selected CDF cell authoritative. Only a dimension that crossed a shared
            // floating-point boundary moves one representable value toward that cell's center.
            if (replayed->row != row_selection->index) {
                if (row_interior_steps == maximum_interior_steps) {
                    break;
                }
                ++row_interior_steps;
            }
            if (replayed->column != column_selection->index) {
                if (column_interior_steps == maximum_interior_steps) {
                    break;
                }
                ++column_interior_steps;
            }
        }
        if (!stable_world_direction.has_value()) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map rotation cannot preserve the selected texel in transport "
                "precision."));
        }
        const auto probability = LightProbabilityDensityT<Scalar>{
            .value = directional_densities_[index],
            .measure = ProbabilityMeasure::solid_angle,
        };
        const auto sampled = IncidentLightSampleT<Scalar>::create_infinite(
            *stable_world_direction, texels_[index], probability);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        return std::optional<IncidentLightSampleT<Scalar>>{*sampled};
    }

    [[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>&, const Vector3T<Scalar> direction_to_light,
           const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto wavelength_status = validate_wavelengths(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        if (!light_detail::unit_direction(direction_to_light)) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map PDF evaluation requires a finite unit direction."));
        }
        if (is_black_) {
            return DirectionalLightPdfT<Scalar>::create(Scalar{0});
        }
        const auto texel = world_direction_texel(direction_to_light);
        if (!texel) {
            return std::unexpected(texel.error());
        }
        return DirectionalLightPdfT<Scalar>::create(directional_densities_[texel->index]);
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    le(const RayT<Scalar>& escaped_ray, const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto wavelength_status = validate_wavelengths(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        const auto texel = world_direction_texel(escaped_ray.direction());
        if (!texel) {
            return std::unexpected(texel.error());
        }
        return texels_[texel->index];
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    power(const Bounds3T<Scalar>& scene_bounds,
          const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto sphere_scale = environment_light_detail::bounding_sphere_scale(scene_bounds);
        if (!sphere_scale) {
            return std::unexpected(sphere_scale.error());
        }
        const auto wavelength_status = validate_wavelengths(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        auto result = LightSpectrumT<Scalar>{};
        if (sphere_scale->is_point || is_black_) {
            return result;
        }
        const auto binary_exponent = static_cast<long long>(sphere_scale->radius_exponent) * 2;
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            if (integrated_lanes_[lane].peak == Scalar{0}) {
                continue;
            }
            const auto scaled = environment_light_detail::scale_radiometric_value(
                integrated_lanes_[lane].peak,
                std::array{
                    integrated_lanes_[lane].normalized_solid_angle_integral,
                    std::numbers::pi_v<Scalar>,
                    sphere_scale->scaled_squared_radius,
                },
                binary_exponent);
            if (!scaled) {
                return std::unexpected(scaled.error());
            }
            result[lane] = *scaled;
        }
        return result;
    }

    [[nodiscard]] static constexpr Bounds3T<Scalar> bounds() noexcept {
        return Bounds3T<Scalar>::unbounded();
    }
    [[nodiscard]] constexpr std::uint32_t width() const noexcept {
        return width_;
    }
    [[nodiscard]] constexpr std::uint32_t height() const noexcept {
        return height_;
    }
    [[nodiscard]] const std::vector<LightSpectrumT<Scalar>>& texels() const noexcept {
        return texels_;
    }
    [[nodiscard]] constexpr QuaternionT<Scalar> environment_to_world() const noexcept {
        return environment_to_world_;
    }

  private:
    EnvironmentMapLightT(const std::uint32_t width, const std::uint32_t height,
                         std::vector<LightSpectrumT<Scalar>> texels,
                         const std::array<Scalar, TransportSpectrumSampleCount> nanometers,
                         const QuaternionT<Scalar> environment_to_world,
                         const AffineTransformT<Scalar> environment_to_world_transform,
                         std::vector<Scalar> azimuth_boundaries,
                         std::vector<Scalar> cosine_boundaries, std::vector<Scalar> row_cdf,
                         std::vector<Scalar> column_cdfs, std::vector<Scalar> directional_densities,
                         const std::array<environment_light_detail::IntegratedLaneT<Scalar>,
                                          TransportSpectrumSampleCount>
                             integrated_lanes,
                         const bool is_black) noexcept
        : width_{width}, height_{height}, texels_{std::move(texels)}, nanometers_{nanometers},
          environment_to_world_{environment_to_world},
          environment_to_world_transform_{environment_to_world_transform},
          azimuth_boundaries_{std::move(azimuth_boundaries)},
          cosine_boundaries_{std::move(cosine_boundaries)}, row_cdf_{std::move(row_cdf)},
          column_cdfs_{std::move(column_cdfs)},
          directional_densities_{std::move(directional_densities)},
          integrated_lanes_{integrated_lanes}, is_black_{is_black} {}

    [[nodiscard]] core::Status
    validate_wavelengths(const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto status = light_detail::validate_light_wavelength_packet(wavelengths);
        if (!status) {
            return std::unexpected(status.error());
        }
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            if (wavelengths[lane].nanometers != nanometers_[lane]) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "A packet-bound environment map cannot be evaluated at different "
                    "wavelengths."));
            }
        }
        return {};
    }

    [[nodiscard]] core::Result<environment_light_detail::DirectionTexelT<Scalar>>
    world_direction_texel(const Vector3T<Scalar> world_direction) const {
        const auto normalized = environment_light_detail::robust_unit_direction(
            world_direction,
            "Environment-map lookup requires a representable non-zero world direction.");
        if (!normalized) {
            return std::unexpected(normalized.error());
        }
        const auto local = environment_to_world_transform_.apply_inverse(*normalized);
        return environment_light_detail::direction_texel(
            local, std::span<const Scalar>{azimuth_boundaries_},
            std::span<const Scalar>{cosine_boundaries_});
    }

    [[nodiscard]] core::Result<Vector3T<Scalar>>
    sample_local_direction(const std::size_t row, const std::size_t column,
                           const Scalar row_remapped, const Scalar column_remapped,
                           const std::size_t row_interior_steps,
                           const std::size_t column_interior_steps) const {
        const auto cosine0 = cosine_boundaries_[row];
        const auto cosine1 = cosine_boundaries_[row + 1U];
        const auto cosine_span = cosine0 - cosine1;
        const auto cosine_midpoint = std::midpoint(cosine0, cosine1);
        auto cosine = std::fma(-row_remapped, cosine_span, cosine0);
        if (!(cosine < cosine0) || !(cosine > cosine1)) {
            cosine = row_remapped == Scalar{0}
                         ? cosine_midpoint
                         : (cosine >= cosine0 ? std::nextafter(cosine0, cosine1)
                                              : std::nextafter(cosine1, cosine0));
            if (!(cosine < cosine0) || !(cosine > cosine1)) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "Environment-map polar sample is not interior in transport precision."));
            }
        }
        for (auto step = std::size_t{0}; step < row_interior_steps; ++step) {
            const auto moved = std::nextafter(cosine, cosine_midpoint);
            if (moved == cosine || !(moved < cosine0) || !(moved > cosine1)) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "Environment-map polar sample cannot move farther into its selected bin."));
            }
            cosine = moved;
        }
        if (!std::isfinite(cosine) || !(cosine > Scalar{-1}) || !(cosine < Scalar{1})) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map polar sampling is not representable."));
        }
        const auto radial_squared = (Scalar{1} - cosine) * (Scalar{1} + cosine);
        const auto radial = std::sqrt(radial_squared);
        if (!std::isfinite(radial_squared) || !(radial_squared > Scalar{0}) ||
            !std::isfinite(radial) || !(radial > Scalar{0})) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map polar sampling is not representable."));
        }

        const auto azimuth0 = azimuth_boundaries_[column];
        const auto azimuth1 = azimuth_boundaries_[column + 1U];
        const auto azimuth_span = azimuth1 - azimuth0;
        const auto azimuth_midpoint = std::midpoint(azimuth0, azimuth1);
        auto azimuth = std::fma(column_remapped, azimuth_span, azimuth0);
        if (!(azimuth > azimuth0) || !(azimuth < azimuth1)) {
            azimuth = column_remapped == Scalar{0}
                          ? azimuth_midpoint
                          : (azimuth <= azimuth0 ? std::nextafter(azimuth0, azimuth1)
                                                 : std::nextafter(azimuth1, azimuth0));
            if (!(azimuth > azimuth0) || !(azimuth < azimuth1)) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "Environment-map azimuth sample is not interior in transport precision."));
            }
        }
        for (auto step = std::size_t{0}; step < column_interior_steps; ++step) {
            const auto moved = std::nextafter(azimuth, azimuth_midpoint);
            if (moved == azimuth || !(moved > azimuth0) || !(moved < azimuth1)) {
                return std::unexpected(environment_light_detail::invalid_environment_light(
                    "Environment-map azimuth sample cannot move farther into its selected bin."));
            }
            azimuth = moved;
        }
        if (!std::isfinite(azimuth) || azimuth < Scalar{0} ||
            !(azimuth < azimuth_boundaries_.back())) {
            return std::unexpected(environment_light_detail::invalid_environment_light(
                "Environment-map azimuth sampling is not representable."));
        }

        return environment_light_detail::robust_unit_direction(
            Vector3T<Scalar>{
                .x = radial * std::cos(azimuth),
                .y = cosine,
                .z = radial * std::sin(azimuth),
            },
            "Environment-map sampling did not produce a representable unit direction.");
    }

    std::uint32_t width_;
    std::uint32_t height_;
    std::vector<LightSpectrumT<Scalar>> texels_;
    std::array<Scalar, TransportSpectrumSampleCount> nanometers_;
    QuaternionT<Scalar> environment_to_world_;
    AffineTransformT<Scalar> environment_to_world_transform_;
    std::vector<Scalar> azimuth_boundaries_;
    std::vector<Scalar> cosine_boundaries_;
    std::vector<Scalar> row_cdf_;
    std::vector<Scalar> column_cdfs_;
    std::vector<Scalar> directional_densities_;
    std::array<environment_light_detail::IntegratedLaneT<Scalar>, TransportSpectrumSampleCount>
        integrated_lanes_;
    bool is_black_;
};

using EnvironmentMapLight = EnvironmentMapLightT<TransportScalar>;
using ReferenceEnvironmentMapLight = EnvironmentMapLightT<ReferenceScalar>;

static_assert(LightModelFor<EnvironmentMapLight, TransportScalar>);
static_assert(LightModelFor<ReferenceEnvironmentMapLight, ReferenceScalar>);

} // namespace blackframe::renderer
