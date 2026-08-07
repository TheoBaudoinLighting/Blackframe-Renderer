#include <Blackframe/Renderer/HostImageFilter.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <GeometryScalar Scalar> struct AxisStencilT final {
    std::array<std::optional<std::int64_t>, 4> indices{};
    std::size_t count{};
    Scalar fraction{};
};

[[nodiscard]] core::Error filter_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

[[nodiscard]] core::Status validate_image(const HostImage& image) {
    if (image.width() == 0U || image.height() == 0U || image.channel_count() == 0U) {
        return std::unexpected(
            filter_error(core::StatusCode::incompatible,
                         "A filtered host image must have non-zero dimensions and channels."));
    }
    if (image.channel_names().size() != image.channel_count()) {
        return std::unexpected(
            filter_error(core::StatusCode::incompatible,
                         "A filtered host image must preserve one name for every stored channel."));
    }

    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    const auto width = static_cast<std::size_t>(image.width());
    const auto height = static_cast<std::size_t>(image.height());
    const auto channels = static_cast<std::size_t>(image.channel_count());
    if (width > maximum / height || width * height > maximum / channels ||
        width * height * channels != image.pixels().size()) {
        return std::unexpected(
            filter_error(core::StatusCode::incompatible,
                         "A filtered host image must have exact interleaved pixel storage."));
    }
    return {};
}

[[nodiscard]] core::Result<std::int64_t> checked_add(const std::int64_t left,
                                                     const std::int64_t right) {
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if ((right > 0 && left > maximum - right) || (right < 0 && left < minimum - right)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "A filtered texture tap is outside the representable integer range."));
    }
    return left + right;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> scaled_coordinate(const Scalar coordinate,
                                                     const std::uint32_t extent) {
    if (!std::isfinite(coordinate)) {
        return std::unexpected(filter_error(core::StatusCode::invalid_argument,
                                            "A filtered texture coordinate must be finite."));
    }
    const auto scalar_extent = static_cast<Scalar>(extent);
    if (!std::isfinite(scalar_extent) ||
        static_cast<ReferenceScalar>(scalar_extent) != static_cast<ReferenceScalar>(extent)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "A filtered texture extent is not exact in the requested precision."));
    }
    const auto scaled = coordinate * scalar_extent;
    constexpr auto exact_texel_limit = [] {
        if constexpr (std::same_as<Scalar, TransportScalar>) {
            return Scalar{8'388'608.0}; // 2^23
        } else {
            return Scalar{4'503'599'627'370'496.0}; // 2^52
        }
    }();
    if (!std::isfinite(scaled) || scaled <= -exact_texel_limit || scaled >= exact_texel_limit) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "A filtered texture coordinate does not preserve an exact texel phase."));
    }
    return scaled;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<AxisStencilT<Scalar>>
make_axis_stencil(const Scalar coordinate, const std::int32_t origin, const std::uint32_t extent,
                  const TextureFilterMode filter, const TextureWrapMode wrap) {
    const auto scaled = scaled_coordinate(coordinate, extent);
    if (!scaled) {
        return std::unexpected(scaled.error());
    }

    auto stencil = AxisStencilT<Scalar>{};
    auto base = std::int64_t{};
    auto first_offset = std::int64_t{};
    switch (filter) {
    case TextureFilterMode::nearest:
        base = static_cast<std::int64_t>(std::floor(*scaled));
        stencil.count = 1U;
        break;
    case TextureFilterMode::bilinear:
    case TextureFilterMode::bicubic: {
        const auto centered = *scaled - Scalar{0.5};
        const auto floored = std::floor(centered);
        base = static_cast<std::int64_t>(floored);
        stencil.fraction = centered - floored;
        if (filter == TextureFilterMode::bilinear) {
            stencil.count = 2U;
        } else {
            stencil.count = 4U;
            first_offset = -1;
        }
        break;
    }
    }

    const auto absolute_base = checked_add(base, static_cast<std::int64_t>(origin));
    if (!absolute_base) {
        return std::unexpected(absolute_base.error());
    }
    for (auto tap = std::size_t{}; tap < stencil.count; ++tap) {
        const auto offset = first_offset + static_cast<std::int64_t>(tap);
        const auto unwrapped = checked_add(*absolute_base, offset);
        if (!unwrapped) {
            return std::unexpected(unwrapped.error());
        }
        const auto wrapped = wrap_texture_index(*unwrapped, origin, extent, wrap);
        if (!wrapped) {
            return std::unexpected(wrapped.error());
        }
        stencil.indices[tap] = *wrapped;
    }
    return stencil;
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar fetch_channel(const HostImage& image, const std::optional<std::int64_t> x,
                                   const std::optional<std::int64_t> y,
                                   const std::uint32_t channel) noexcept {
    if (!x || !y) {
        return Scalar{};
    }
    const auto column = static_cast<std::size_t>(*x - image.origin_x());
    const auto row = static_cast<std::size_t>(*y - image.origin_y());
    const auto pixel = row * static_cast<std::size_t>(image.width()) + column;
    const auto element = pixel * static_cast<std::size_t>(image.channel_count()) + channel;
    return static_cast<Scalar>(image.pixels()[element]);
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar catmull_rom(const Scalar p0, const Scalar p1, const Scalar p2, const Scalar p3,
                                 const Scalar t) noexcept {
    const auto t2 = t * t;
    const auto t3 = t2 * t;
    const auto weight0 = -Scalar{0.5} * t + t2 - Scalar{0.5} * t3;
    const auto weight2 = Scalar{0.5} * t + Scalar{2} * t2 - Scalar{1.5} * t3;
    const auto weight3 = -Scalar{0.5} * t2 + Scalar{0.5} * t3;
    return p1 + weight0 * (p0 - p1) + weight2 * (p2 - p1) + weight3 * (p3 - p1);
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar bicubic_filter(const std::array<std::array<Scalar, 4>, 4>& samples,
                                    const Scalar x_fraction, const Scalar y_fraction) noexcept {
    if (x_fraction == Scalar{} && y_fraction == Scalar{}) {
        return samples[1][1];
    }

    if (x_fraction == Scalar{} || y_fraction == Scalar{}) {
        auto axis_samples = std::array<Scalar, 4>{};
        if (x_fraction == Scalar{}) {
            for (auto index = std::size_t{}; index < axis_samples.size(); ++index) {
                axis_samples[index] = samples[index][1];
            }
        } else {
            axis_samples = samples[1];
        }

        const auto fraction = x_fraction == Scalar{} ? y_fraction : x_fraction;
        auto scale = Scalar{1};
        auto normalize = false;
        if constexpr (std::same_as<Scalar, TransportScalar>) {
            scale = Scalar{};
            for (const auto value : axis_samples) {
                scale = std::max(scale, std::abs(value));
            }
            normalize = scale > std::numeric_limits<Scalar>::max() / Scalar{4};
        }
        if (normalize) {
            return catmull_rom(axis_samples[0] / scale, axis_samples[1] / scale,
                               axis_samples[2] / scale, axis_samples[3] / scale, fraction) *
                   scale;
        }
        return catmull_rom(axis_samples[0], axis_samples[1], axis_samples[2], axis_samples[3],
                           fraction);
    }

    auto scale = Scalar{1};
    auto normalize = false;
    if constexpr (std::same_as<Scalar, TransportScalar>) {
        scale = Scalar{};
        for (const auto& row : samples) {
            for (const auto value : row) {
                scale = std::max(scale, std::abs(value));
            }
        }
        normalize = scale > std::numeric_limits<Scalar>::max() / Scalar{4};
    }

    auto rows = std::array<Scalar, 4>{};
    for (auto row = std::size_t{}; row < rows.size(); ++row) {
        if (normalize) {
            rows[row] = catmull_rom(samples[row][0] / scale, samples[row][1] / scale,
                                    samples[row][2] / scale, samples[row][3] / scale, x_fraction);
        } else {
            rows[row] = catmull_rom(samples[row][0], samples[row][1], samples[row][2],
                                    samples[row][3], x_fraction);
        }
    }
    const auto filtered = catmull_rom(rows[0], rows[1], rows[2], rows[3], y_fraction);
    return normalize ? filtered * scale : filtered;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
filter_channel(const HostImage& image, const Point2T<Scalar> uv, const std::uint32_t channel,
               const TextureFilterMode filter, const TextureWrapMode u_wrap,
               const TextureWrapMode v_wrap) {
    if (!is_valid_texture_filter_mode(filter)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "A texture filter must be a supported explicit value."));
    }
    if (!is_valid_texture_wrap_mode(u_wrap) || !is_valid_texture_wrap_mode(v_wrap)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "A filtered texture wrap mode must be a supported explicit value."));
    }
    const auto image_status = validate_image(image);
    if (!image_status) {
        return std::unexpected(image_status.error());
    }
    if (channel >= image.channel_count()) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "A filtered texture channel must address the immutable image snapshot."));
    }
    if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) {
        return std::unexpected(filter_error(core::StatusCode::invalid_argument,
                                            "Filtered texture coordinates must both be finite."));
    }

    const auto x_stencil = make_axis_stencil(uv.x, image.origin_x(), image.width(), filter, u_wrap);
    if (!x_stencil) {
        return std::unexpected(x_stencil.error());
    }
    const auto y_stencil =
        make_axis_stencil(uv.y, image.origin_y(), image.height(), filter, v_wrap);
    if (!y_stencil) {
        return std::unexpected(y_stencil.error());
    }

    auto filtered = Scalar{};
    switch (filter) {
    case TextureFilterMode::nearest:
        filtered =
            fetch_channel<Scalar>(image, x_stencil->indices[0], y_stencil->indices[0], channel);
        break;
    case TextureFilterMode::bilinear: {
        const auto upper = std::lerp(
            fetch_channel<Scalar>(image, x_stencil->indices[0], y_stencil->indices[0], channel),
            fetch_channel<Scalar>(image, x_stencil->indices[1], y_stencil->indices[0], channel),
            x_stencil->fraction);
        const auto lower = std::lerp(
            fetch_channel<Scalar>(image, x_stencil->indices[0], y_stencil->indices[1], channel),
            fetch_channel<Scalar>(image, x_stencil->indices[1], y_stencil->indices[1], channel),
            x_stencil->fraction);
        filtered = std::lerp(upper, lower, y_stencil->fraction);
        break;
    }
    case TextureFilterMode::bicubic: {
        auto samples = std::array<std::array<Scalar, 4>, 4>{};
        for (auto row = std::size_t{}; row < samples.size(); ++row) {
            for (auto column = std::size_t{}; column < samples[row].size(); ++column) {
                samples[row][column] = fetch_channel<Scalar>(image, x_stencil->indices[column],
                                                             y_stencil->indices[row], channel);
            }
        }
        filtered = bicubic_filter(samples, x_stencil->fraction, y_stencil->fraction);
        break;
    }
    }
    if (!std::isfinite(filtered)) {
        return std::unexpected(filter_error(core::StatusCode::invalid_argument,
                                            "A filtered texture value must remain finite."));
    }

    return filtered;
}

} // namespace

core::Result<TransportScalar> filter_host_image_channel(const HostImage& image, const Point2 uv,
                                                        const std::uint32_t channel,
                                                        const TextureFilterMode filter,
                                                        const TextureWrapMode u_wrap,
                                                        const TextureWrapMode v_wrap) {
    return filter_channel(image, uv, channel, filter, u_wrap, v_wrap);
}

core::Result<ReferenceScalar>
filter_host_image_channel(const HostImage& image, const ReferencePoint2 uv,
                          const std::uint32_t channel, const TextureFilterMode filter,
                          const TextureWrapMode u_wrap, const TextureWrapMode v_wrap) {
    return filter_channel(image, uv, channel, filter, u_wrap, v_wrap);
}

} // namespace blackframe::renderer
