#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/LocalPcg32.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

namespace blackframe::renderer {

template <GeometryScalar Scalar>
inline constexpr std::uint64_t MaximumStratifiedSampleCount =
    std::uint64_t{1} << std::numeric_limits<Scalar>::digits;

namespace stratified_sampling_detail {

[[nodiscard]] inline core::Error invalid_input(const char* const message) {
    return {
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

[[nodiscard]] inline core::Error unsupported_size(const char* const message) {
    return {
        .code = core::StatusCode::resource_exhausted,
        .message = message,
    };
}

[[nodiscard]] inline bool product_overflows(const std::uint64_t left,
                                            const std::uint64_t right) noexcept {
    return right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right;
}

[[nodiscard]] inline bool exceeds_host_size(const std::uint64_t value) noexcept {
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        return value > std::numeric_limits<std::size_t>::max();
    }
    return false;
}

template <GeometryScalar Scalar>
[[nodiscard]] inline std::uint64_t classified_stratum(const Scalar sample,
                                                      const std::uint64_t stratum_count) noexcept {
    return static_cast<std::uint64_t>(static_cast<ReferenceScalar>(sample) *
                                      static_cast<ReferenceScalar>(stratum_count));
}

template <GeometryScalar Scalar>
[[nodiscard]] inline Scalar sample_stratum(const std::uint64_t stratum,
                                           const std::uint64_t stratum_count,
                                           LocalPcg32& rng) noexcept {
    const auto count = static_cast<Scalar>(stratum_count);
    auto sample = (static_cast<Scalar>(stratum) + rng.next_1d<Scalar>()) / count;

    if (!(sample < Scalar{1})) {
        sample = std::nextafter(Scalar{1}, Scalar{0});
    }

    // Rounded scalar division can cross an internal boundary. Move by representable values until
    // the public histogram classification matches the stratum that owns this sample.
    auto classified = classified_stratum(sample, stratum_count);
    while (classified < stratum) {
        sample = std::nextafter(sample, Scalar{1});
        classified = classified_stratum(sample, stratum_count);
    }
    while (classified > stratum) {
        sample = std::nextafter(sample, Scalar{0});
        classified = classified_stratum(sample, stratum_count);
    }
    return sample;
}

[[nodiscard]] inline std::uint64_t
bounded_index(LocalPcg32& rng, const std::uint64_t exclusive_upper_bound) noexcept {
    const auto threshold = (std::uint64_t{0} - exclusive_upper_bound) % exclusive_upper_bound;
    for (;;) {
        const auto candidate = rng.next_u64();
        if (candidate >= threshold) {
            return candidate % exclusive_upper_bound;
        }
    }
}

template <GeometryScalar Scalar>
[[nodiscard]] inline core::Status validate_precision(const std::uint64_t stratum_count,
                                                     const char* const message) {
    if (stratum_count > MaximumStratifiedSampleCount<Scalar>) {
        return std::unexpected(unsupported_size(message));
    }
    return {};
}

} // namespace stratified_sampling_detail

// The output order is the stratum order. Every accepted call fills the complete span.
template <GeometryScalar Scalar, std::size_t Extent>
[[nodiscard]] core::Status generate_stratified_1d(const std::span<Scalar, Extent> samples,
                                                  LocalPcg32& rng) {
    if (samples.empty()) {
        return std::unexpected(stratified_sampling_detail::invalid_input(
            "Stratified 1D sampling requires at least one output stratum."));
    }

    const auto stratum_count = static_cast<std::uint64_t>(samples.size());
    const auto precision_status = stratified_sampling_detail::validate_precision<Scalar>(
        stratum_count, "Stratified 1D sampling exceeds the scalar precision limit.");
    if (!precision_status.has_value()) {
        return precision_status;
    }

    for (std::uint64_t stratum = 0; stratum < stratum_count; ++stratum) {
        samples[static_cast<std::size_t>(stratum)] =
            stratified_sampling_detail::sample_stratum<Scalar>(stratum, stratum_count, rng);
    }
    return {};
}

// The output order is row-major in strata: x varies first, then y.
template <GeometryScalar Scalar, std::size_t Extent>
[[nodiscard]] core::Status generate_stratified_2d(const std::span<Point2T<Scalar>, Extent> samples,
                                                  const std::uint64_t x_strata,
                                                  const std::uint64_t y_strata, LocalPcg32& rng) {
    if (x_strata == 0 || y_strata == 0) {
        return std::unexpected(stratified_sampling_detail::invalid_input(
            "Stratified 2D sampling requires non-zero axis counts."));
    }
    if (stratified_sampling_detail::product_overflows(x_strata, y_strata)) {
        return std::unexpected(stratified_sampling_detail::unsupported_size(
            "Stratified 2D sampling axis product overflows uint64."));
    }

    const auto sample_count = x_strata * y_strata;
    if (stratified_sampling_detail::exceeds_host_size(sample_count)) {
        return std::unexpected(stratified_sampling_detail::unsupported_size(
            "Stratified 2D sampling exceeds the host span size."));
    }

    const auto x_precision = stratified_sampling_detail::validate_precision<Scalar>(
        x_strata, "Stratified 2D sampling exceeds the x-axis scalar precision limit.");
    if (!x_precision.has_value()) {
        return x_precision;
    }
    const auto y_precision = stratified_sampling_detail::validate_precision<Scalar>(
        y_strata, "Stratified 2D sampling exceeds the y-axis scalar precision limit.");
    if (!y_precision.has_value()) {
        return y_precision;
    }
    if (samples.size() != static_cast<std::size_t>(sample_count)) {
        return std::unexpected(stratified_sampling_detail::invalid_input(
            "Stratified 2D sampling requires exactly x_strata * y_strata outputs."));
    }

    for (std::uint64_t y = 0; y < y_strata; ++y) {
        for (std::uint64_t x = 0; x < x_strata; ++x) {
            samples[static_cast<std::size_t>(y * x_strata + x)] = {
                .x = stratified_sampling_detail::sample_stratum<Scalar>(x, x_strata, rng),
                .y = stratified_sampling_detail::sample_stratum<Scalar>(y, y_strata, rng),
            };
        }
    }
    return {};
}

// Samples are stored row-major as [sample][dimension]. Each dimension is independently permuted.
template <GeometryScalar Scalar, std::size_t Extent>
[[nodiscard]] core::Status
generate_latin_hypercube(const std::span<Scalar, Extent> samples, const std::uint64_t sample_count,
                         const std::uint64_t dimension_count, LocalPcg32& rng) {
    if (sample_count == 0 || dimension_count == 0) {
        return std::unexpected(stratified_sampling_detail::invalid_input(
            "Latin hypercube sampling requires non-zero sample and dimension counts."));
    }
    if (stratified_sampling_detail::product_overflows(sample_count, dimension_count)) {
        return std::unexpected(stratified_sampling_detail::unsupported_size(
            "Latin hypercube sampling shape overflows uint64."));
    }

    const auto value_count = sample_count * dimension_count;
    if (stratified_sampling_detail::exceeds_host_size(value_count)) {
        return std::unexpected(stratified_sampling_detail::unsupported_size(
            "Latin hypercube sampling exceeds the host span size."));
    }

    const auto precision_status = stratified_sampling_detail::validate_precision<Scalar>(
        sample_count, "Latin hypercube sampling exceeds the scalar precision limit.");
    if (!precision_status.has_value()) {
        return precision_status;
    }
    if (samples.size() != static_cast<std::size_t>(value_count)) {
        return std::unexpected(stratified_sampling_detail::invalid_input(
            "Latin hypercube sampling requires exactly sample_count * dimension_count outputs."));
    }

    for (std::uint64_t sample = 0; sample < sample_count; ++sample) {
        for (std::uint64_t dimension = 0; dimension < dimension_count; ++dimension) {
            samples[static_cast<std::size_t>(sample * dimension_count + dimension)] =
                stratified_sampling_detail::sample_stratum<Scalar>(sample, sample_count, rng);
        }
    }

    for (std::uint64_t dimension = 0; dimension < dimension_count; ++dimension) {
        for (auto remaining = sample_count; remaining > 1; --remaining) {
            const auto current = remaining - 1;
            const auto selected = stratified_sampling_detail::bounded_index(rng, remaining);
            std::swap(samples[static_cast<std::size_t>(current * dimension_count + dimension)],
                      samples[static_cast<std::size_t>(selected * dimension_count + dimension)]);
        }
    }
    return {};
}

static_assert(MaximumStratifiedSampleCount<TransportScalar> == (std::uint64_t{1} << 24U));
static_assert(MaximumStratifiedSampleCount<ReferenceScalar> == (std::uint64_t{1} << 53U));
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));

} // namespace blackframe::renderer
