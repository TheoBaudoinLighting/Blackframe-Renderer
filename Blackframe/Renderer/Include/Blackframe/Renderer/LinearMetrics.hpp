#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

namespace blackframe::renderer {

struct LinearMetrics final {
    ReferenceScalar mse{};
    ReferenceScalar rmse{};
    ReferenceScalar mean_bias{};
    ReferenceScalar maximum_absolute_error{};

    [[nodiscard]] constexpr bool operator==(const LinearMetrics&) const noexcept = default;
};

namespace linear_metrics_detail {

[[nodiscard]] inline core::Error invalid_metrics_input(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

template <AccumulationPrecision EvaluatedPrecision, AccumulationPrecision ReferencePrecision>
[[nodiscard]] core::Status validate_domains(const FilmT<EvaluatedPrecision>& evaluated,
                                            const FilmT<ReferencePrecision>& reference) {
    const auto evaluated_extent = evaluated.extent();
    const auto reference_extent = reference.extent();
    if (evaluated_extent.width != reference_extent.width ||
        evaluated_extent.height != reference_extent.height) {
        return std::unexpected(
            invalid_metrics_input("Linear metrics require identical full film extents."));
    }
    if (evaluated.crop() != reference.crop()) {
        return std::unexpected(
            invalid_metrics_input("Linear metrics require identical active film crops."));
    }
    return {};
}

[[nodiscard]] inline core::Status accumulate_error(const ReferenceScalar error,
                                                   ReferenceScalar& squared_error_sum,
                                                   ReferenceScalar& bias_sum,
                                                   ReferenceScalar& maximum_absolute_error) {
    if (!std::isfinite(error)) {
        return std::unexpected(invalid_metrics_input(
            "Linear metric subtraction produced a non-finite component error."));
    }

    const auto squared_error = error * error;
    if (!std::isfinite(squared_error)) {
        return std::unexpected(
            invalid_metrics_input("A linear metric component error cannot be squared in double."));
    }

    squared_error_sum += squared_error;
    bias_sum += error;
    if (!std::isfinite(squared_error_sum) || !std::isfinite(bias_sum)) {
        return std::unexpected(
            invalid_metrics_input("Linear metric accumulation overflowed double precision."));
    }
    maximum_absolute_error = std::max(maximum_absolute_error, std::abs(error));
    return {};
}

} // namespace linear_metrics_detail

// Computes component-wise metrics over every RGB component in the shared active
// crop. Bias is signed as evaluated minus reference. Inputs may use either film
// accumulation precision, while all metric arithmetic and results use double.
template <AccumulationPrecision EvaluatedPrecision, AccumulationPrecision ReferencePrecision>
[[nodiscard]] core::Result<LinearMetrics>
compute_linear_metrics(const FilmT<EvaluatedPrecision>& evaluated,
                       const FilmT<ReferencePrecision>& reference) {
    const auto domain_status = linear_metrics_detail::validate_domains(evaluated, reference);
    if (!domain_status.has_value()) {
        return std::unexpected(domain_status.error());
    }

    auto squared_error_sum = ReferenceScalar{};
    auto bias_sum = ReferenceScalar{};
    auto maximum_absolute_error = ReferenceScalar{};
    const auto crop = evaluated.crop();
    for (auto y = crop.minimum_y; y < crop.maximum_y; ++y) {
        for (auto x = crop.minimum_x; x < crop.maximum_x; ++x) {
            const auto evaluated_pixel = evaluated.resolved_pixel(x, y);
            if (!evaluated_pixel.has_value()) {
                return std::unexpected(evaluated_pixel.error());
            }
            const auto reference_pixel = reference.resolved_pixel(x, y);
            if (!reference_pixel.has_value()) {
                return std::unexpected(reference_pixel.error());
            }

            const auto errors = std::array{
                static_cast<ReferenceScalar>(evaluated_pixel->red) -
                    static_cast<ReferenceScalar>(reference_pixel->red),
                static_cast<ReferenceScalar>(evaluated_pixel->green) -
                    static_cast<ReferenceScalar>(reference_pixel->green),
                static_cast<ReferenceScalar>(evaluated_pixel->blue) -
                    static_cast<ReferenceScalar>(reference_pixel->blue),
            };
            for (const auto error : errors) {
                const auto status = linear_metrics_detail::accumulate_error(
                    error, squared_error_sum, bias_sum, maximum_absolute_error);
                if (!status.has_value()) {
                    return std::unexpected(status.error());
                }
            }
        }
    }

    constexpr auto rgb_component_count = std::size_t{3};
    const auto observation_count = evaluated.pixel_count() * rgb_component_count;
    const auto denominator = static_cast<ReferenceScalar>(observation_count);
    const auto mse = squared_error_sum / denominator;
    const auto mean_bias = bias_sum / denominator;
    const auto rmse = std::sqrt(mse);
    if (!std::isfinite(mse) || !std::isfinite(rmse) || !std::isfinite(mean_bias)) {
        return std::unexpected(linear_metrics_detail::invalid_metrics_input(
            "Linear metric normalization produced a non-finite result."));
    }

    return LinearMetrics{
        .mse = mse,
        .rmse = rmse,
        .mean_bias = mean_bias,
        .maximum_absolute_error = maximum_absolute_error,
    };
}

} // namespace blackframe::renderer
