#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace blackframe::renderer {

struct QualityThresholds final {
    ReferenceScalar maximum_linear_mse{};
    ReferenceScalar minimum_display_psnr{};

    [[nodiscard]] constexpr bool operator==(const QualityThresholds&) const noexcept = default;
};

// A checkpoint records cumulative render time from a monotonic clock after
// every active film pixel has received the stated number of samples. The
// duration covers sample generation, transport, sensor conversion, and film
// accumulation. Scene/backend construction, film allocation, reference I/O,
// metric evaluation, and output encoding remain outside it.
struct QualityCheckpoint final {
    std::uint64_t cumulative_samples_per_pixel{};
    std::chrono::nanoseconds cumulative_render_time{};
    ReferenceScalar linear_mse{};
    ReferenceScalar display_psnr{};

    [[nodiscard]] constexpr bool operator==(const QualityCheckpoint&) const noexcept = default;
};

struct QualityThresholdHit final {
    std::uint64_t cumulative_samples_per_pixel{};
    std::chrono::nanoseconds cumulative_render_time{};
    ReferenceScalar observed_value{};

    [[nodiscard]] constexpr bool operator==(const QualityThresholdHit&) const noexcept = default;
};

struct TimeToQualityResult final {
    // An empty hit means the corresponding threshold was not observed. The
    // final checkpoint is never substituted for an unreached target.
    std::optional<QualityThresholdHit> linear_mse;
    std::optional<QualityThresholdHit> display_psnr;

    [[nodiscard]] constexpr bool operator==(const TimeToQualityResult&) const noexcept = default;
};

namespace convergence_metrics_detail {

[[nodiscard]] inline core::Error invalid_convergence_input(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

[[nodiscard]] inline bool valid_display_psnr(const ReferenceScalar value) noexcept {
    return (std::isfinite(value) && value >= ReferenceScalar{0}) ||
           (std::isinf(value) && value > ReferenceScalar{0});
}

} // namespace convergence_metrics_detail

// Returns the first observed inclusive threshold crossings. Values between
// checkpoints are never interpolated or extrapolated, and stochastic quality
// values need not be monotonic. The full timeline is validated before either
// hit is selected so malformed trailing data cannot be silently ignored.
[[nodiscard]] inline core::Result<TimeToQualityResult>
compute_time_to_quality(std::span<const QualityCheckpoint> checkpoints,
                        const QualityThresholds thresholds) {
    using convergence_metrics_detail::invalid_convergence_input;

    if (!std::isfinite(thresholds.maximum_linear_mse) ||
        thresholds.maximum_linear_mse < ReferenceScalar{0}) {
        return std::unexpected(invalid_convergence_input(
            "The time-to-MSE threshold must be finite and non-negative."));
    }
    if (!std::isfinite(thresholds.minimum_display_psnr) ||
        thresholds.minimum_display_psnr < ReferenceScalar{0}) {
        return std::unexpected(invalid_convergence_input(
            "The time-to-PSNR threshold must be finite and non-negative."));
    }
    if (checkpoints.empty()) {
        return std::unexpected(
            invalid_convergence_input("Time-to-quality requires at least one quality checkpoint."));
    }

    auto previous_samples_per_pixel = std::uint64_t{};
    auto previous_render_time = std::chrono::nanoseconds{};
    for (auto index = std::size_t{}; index < checkpoints.size(); ++index) {
        const auto& checkpoint = checkpoints[index];
        if (checkpoint.cumulative_samples_per_pixel == 0U) {
            return std::unexpected(invalid_convergence_input(
                "A quality checkpoint must contain at least one sample per pixel."));
        }
        if (index != 0U && checkpoint.cumulative_samples_per_pixel <= previous_samples_per_pixel) {
            return std::unexpected(invalid_convergence_input(
                "Quality checkpoint sample counts must be strictly increasing."));
        }
        if (checkpoint.cumulative_render_time.count() < 0) {
            return std::unexpected(
                invalid_convergence_input("A cumulative render duration cannot be negative."));
        }
        if (index != 0U && checkpoint.cumulative_render_time < previous_render_time) {
            return std::unexpected(invalid_convergence_input(
                "Quality checkpoint render durations must be non-decreasing."));
        }
        if (!std::isfinite(checkpoint.linear_mse) || checkpoint.linear_mse < ReferenceScalar{0}) {
            return std::unexpected(invalid_convergence_input(
                "A quality checkpoint linear MSE must be finite and non-negative."));
        }
        if (!convergence_metrics_detail::valid_display_psnr(checkpoint.display_psnr)) {
            return std::unexpected(invalid_convergence_input(
                "A quality checkpoint display PSNR must be finite or positive infinity."));
        }

        previous_samples_per_pixel = checkpoint.cumulative_samples_per_pixel;
        previous_render_time = checkpoint.cumulative_render_time;
    }

    auto result = TimeToQualityResult{};
    for (const auto& checkpoint : checkpoints) {
        if (!result.linear_mse && checkpoint.linear_mse <= thresholds.maximum_linear_mse) {
            result.linear_mse = QualityThresholdHit{
                .cumulative_samples_per_pixel = checkpoint.cumulative_samples_per_pixel,
                .cumulative_render_time = checkpoint.cumulative_render_time,
                .observed_value = checkpoint.linear_mse,
            };
        }
        if (!result.display_psnr && checkpoint.display_psnr >= thresholds.minimum_display_psnr) {
            result.display_psnr = QualityThresholdHit{
                .cumulative_samples_per_pixel = checkpoint.cumulative_samples_per_pixel,
                .cumulative_render_time = checkpoint.cumulative_render_time,
                .observed_value = checkpoint.display_psnr,
            };
        }
    }
    return result;
}

} // namespace blackframe::renderer
