#include <Blackframe/Renderer/HostImageFilter.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
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

template <GeometryScalar Scalar> struct EwaFootprintT final {
    Scalar major_x{1};
    Scalar major_y{};
    Scalar minor_x{};
    Scalar minor_y{1};
    Scalar major_radius{1};
    Scalar minor_radius{1};
    Scalar minor_length{};
};

template <GeometryScalar Scalar> struct EwaEllipseT final {
    Scalar center_x{};
    Scalar center_y{};
    Scalar major_x{1};
    Scalar major_y{};
    Scalar minor_x{};
    Scalar minor_y{1};
    Scalar inverse_major_radius{1};
    Scalar inverse_minor_radius{1};
    std::int64_t minimum_x{};
    std::int64_t maximum_x{};
    std::int64_t minimum_y{};
    std::int64_t maximum_y{};
    std::uint64_t visit_count{};
};

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> scaled_differential(const Scalar differential,
                                                       const std::uint32_t extent) {
    if (!std::isfinite(differential)) {
        return std::unexpected(filter_error(core::StatusCode::invalid_argument,
                                            "EWA texture differentials must be finite."));
    }
    const auto scalar_extent = static_cast<Scalar>(extent);
    if (!std::isfinite(scalar_extent) ||
        static_cast<ReferenceScalar>(scalar_extent) != static_cast<ReferenceScalar>(extent)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "An EWA texture extent is not exact in the requested precision."));
    }
    const auto scaled = differential * scalar_extent;
    constexpr auto exact_texel_limit = [] {
        if constexpr (std::same_as<Scalar, TransportScalar>) {
            return Scalar{8'388'608.0}; // 2^23
        } else {
            return Scalar{4'503'599'627'370'496.0}; // 2^52
        }
    }();
    if (!std::isfinite(scaled) || std::abs(scaled) >= exact_texel_limit) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "An EWA texture differential is not representable in texel space."));
    }
    return scaled;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<EwaFootprintT<Scalar>>
make_ewa_footprint(const HostImage& image,
                   const TextureCoordinateDifferentialsT<Scalar> differentials,
                   const HostImageEwaLimits limits) {
    const auto dudx = scaled_differential(differentials.dudx, image.width());
    const auto dvdx = scaled_differential(differentials.dvdx, image.height());
    const auto dudy = scaled_differential(differentials.dudy, image.width());
    const auto dvdy = scaled_differential(differentials.dvdy, image.height());
    if (!dudx) {
        return std::unexpected(dudx.error());
    }
    if (!dvdx) {
        return std::unexpected(dvdx.error());
    }
    if (!dudy) {
        return std::unexpected(dudy.error());
    }
    if (!dvdy) {
        return std::unexpected(dvdy.error());
    }

    const auto covariance_xx = std::fma(*dudx, *dudx, *dudy * *dudy);
    const auto covariance_xy = std::fma(*dudx, *dvdx, *dudy * *dvdy);
    const auto covariance_yy = std::fma(*dvdx, *dvdx, *dvdy * *dvdy);
    if (!std::isfinite(covariance_xx) || !std::isfinite(covariance_xy) ||
        !std::isfinite(covariance_yy)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "An EWA texture footprint covariance is not representable."));
    }

    const auto trace = covariance_xx + covariance_yy;
    const auto discriminant = std::hypot(covariance_xx - covariance_yy, Scalar{2} * covariance_xy);
    const auto maximum_eigenvalue = Scalar{0.5} * (trace + discriminant);
    const auto jacobian_determinant = std::fma(*dudx, *dvdy, -*dudy * *dvdx);
    const auto determinant = jacobian_determinant * jacobian_determinant;
    auto minimum_eigenvalue =
        maximum_eigenvalue > Scalar{} ? determinant / maximum_eigenvalue : Scalar{};
    const auto maximum_anisotropy = static_cast<Scalar>(limits.maximum_anisotropy);
    const auto minimum_allowed = maximum_eigenvalue / (maximum_anisotropy * maximum_anisotropy);
    minimum_eigenvalue = std::max(minimum_eigenvalue, minimum_allowed);
    const auto angle =
        maximum_eigenvalue > Scalar{}
            ? Scalar{0.5} * std::atan2(Scalar{2} * covariance_xy, covariance_xx - covariance_yy)
            : Scalar{};
    const auto major_x = std::cos(angle);
    const auto major_y = std::sin(angle);
    const auto major_length = std::sqrt(maximum_eigenvalue);
    const auto minor_length = std::sqrt(minimum_eigenvalue);
    auto footprint = EwaFootprintT<Scalar>{
        .major_x = major_x,
        .major_y = major_y,
        .minor_x = -major_y,
        .minor_y = major_x,
        .major_radius = std::hypot(Scalar{1}, major_length),
        .minor_radius = std::hypot(Scalar{1}, minor_length),
        .minor_length = minor_length,
    };
    if (!std::isfinite(footprint.major_x) || !std::isfinite(footprint.major_y) ||
        !std::isfinite(footprint.major_radius) || !std::isfinite(footprint.minor_radius) ||
        !std::isfinite(footprint.minor_length)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "The anisotropy-limited EWA footprint is not representable."));
    }
    return footprint;
}

[[nodiscard]] core::Result<std::uint64_t> checked_visit_product(const std::uint64_t left,
                                                                const std::uint64_t right) {
    if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
        return std::unexpected(
            filter_error(core::StatusCode::resource_exhausted,
                         "An EWA texture footprint visit count is not representable."));
    }
    return left * right;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<EwaEllipseT<Scalar>>
make_ewa_ellipse(const HostImage& image, const Point2T<Scalar> uv,
                 const EwaFootprintT<Scalar> footprint) {
    const auto scaled_x = scaled_coordinate(uv.x, image.width());
    const auto scaled_y = scaled_coordinate(uv.y, image.height());
    if (!scaled_x) {
        return std::unexpected(scaled_x.error());
    }
    if (!scaled_y) {
        return std::unexpected(scaled_y.error());
    }

    auto ellipse = EwaEllipseT<Scalar>{
        .center_x = *scaled_x - Scalar{0.5},
        .center_y = *scaled_y - Scalar{0.5},
        .major_x = footprint.major_x,
        .major_y = footprint.major_y,
        .minor_x = footprint.minor_x,
        .minor_y = footprint.minor_y,
        .inverse_major_radius = Scalar{1} / footprint.major_radius,
        .inverse_minor_radius = Scalar{1} / footprint.minor_radius,
    };
    const auto radius_x = std::hypot(footprint.major_radius * footprint.major_x,
                                     footprint.minor_radius * footprint.minor_x);
    const auto radius_y = std::hypot(footprint.major_radius * footprint.major_y,
                                     footprint.minor_radius * footprint.minor_y);
    const auto minimum_x = std::ceil(ellipse.center_x - radius_x);
    const auto maximum_x = std::floor(ellipse.center_x + radius_x);
    const auto minimum_y = std::ceil(ellipse.center_y - radius_y);
    const auto maximum_y = std::floor(ellipse.center_y + radius_y);
    constexpr auto exact_texel_limit = [] {
        if constexpr (std::same_as<Scalar, TransportScalar>) {
            return Scalar{8'388'608.0}; // 2^23
        } else {
            return Scalar{4'503'599'627'370'496.0}; // 2^52
        }
    }();
    const auto integer_minimum = static_cast<Scalar>(std::numeric_limits<std::int64_t>::min());
    const auto integer_maximum = static_cast<Scalar>(std::numeric_limits<std::int64_t>::max());
    if (!std::isfinite(radius_x) || !std::isfinite(radius_y) || !std::isfinite(minimum_x) ||
        !std::isfinite(maximum_x) || !std::isfinite(minimum_y) || !std::isfinite(maximum_y) ||
        minimum_x < integer_minimum || maximum_x > integer_maximum || minimum_y < integer_minimum ||
        maximum_y > integer_maximum || minimum_x <= -exact_texel_limit ||
        maximum_x >= exact_texel_limit || minimum_y <= -exact_texel_limit ||
        maximum_y >= exact_texel_limit) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "An EWA texture ellipse has unrepresentable integer bounds."));
    }
    ellipse.minimum_x = static_cast<std::int64_t>(minimum_x);
    ellipse.maximum_x = static_cast<std::int64_t>(maximum_x);
    ellipse.minimum_y = static_cast<std::int64_t>(minimum_y);
    ellipse.maximum_y = static_cast<std::int64_t>(maximum_y);
    if (ellipse.maximum_x < ellipse.minimum_x || ellipse.maximum_y < ellipse.minimum_y) {
        return std::unexpected(filter_error(core::StatusCode::internal_error,
                                            "An EWA texture ellipse has an empty bounding box."));
    }
    const auto width = static_cast<std::uint64_t>(ellipse.maximum_x - ellipse.minimum_x) + 1U;
    const auto height = static_cast<std::uint64_t>(ellipse.maximum_y - ellipse.minimum_y) + 1U;
    const auto visit_count = checked_visit_product(width, height);
    if (!visit_count) {
        return std::unexpected(visit_count.error());
    }
    ellipse.visit_count = *visit_count;
    return ellipse;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
sample_ewa_ellipse(const HostImage& image, const EwaEllipseT<Scalar> ellipse,
                   const std::uint32_t channel, const TextureWrapMode u_wrap,
                   const TextureWrapMode v_wrap) {
    constexpr auto gaussian_alpha = Scalar{2};
    const auto edge_weight = std::exp(-gaussian_alpha);
    auto filtered = Scalar{};
    auto accumulated_weight = Scalar{};
    for (auto y = ellipse.minimum_y; y <= ellipse.maximum_y; ++y) {
        const auto absolute_y = checked_add(y, static_cast<std::int64_t>(image.origin_y()));
        if (!absolute_y) {
            return std::unexpected(absolute_y.error());
        }
        const auto wrapped_y =
            wrap_texture_index(*absolute_y, image.origin_y(), image.height(), v_wrap);
        if (!wrapped_y) {
            return std::unexpected(wrapped_y.error());
        }
        const auto offset_y = static_cast<Scalar>(y) - ellipse.center_y;
        for (auto x = ellipse.minimum_x; x <= ellipse.maximum_x; ++x) {
            const auto offset_x = static_cast<Scalar>(x) - ellipse.center_x;
            const auto major_distance =
                std::fma(offset_x, ellipse.major_x, offset_y * ellipse.major_y) *
                ellipse.inverse_major_radius;
            const auto minor_distance =
                std::fma(offset_x, ellipse.minor_x, offset_y * ellipse.minor_y) *
                ellipse.inverse_minor_radius;
            const auto radius_squared =
                std::fma(major_distance, major_distance, minor_distance * minor_distance);
            if (!std::isfinite(radius_squared) || radius_squared < Scalar{}) {
                return std::unexpected(
                    filter_error(core::StatusCode::internal_error,
                                 "An EWA texture radius invariant is not representable."));
            }
            if (radius_squared >= Scalar{1}) {
                continue;
            }
            const auto weight =
                edge_weight * std::expm1(gaussian_alpha * (Scalar{1} - radius_squared));
            if (!std::isfinite(weight) || !(weight > Scalar{})) {
                return std::unexpected(
                    filter_error(core::StatusCode::internal_error,
                                 "An EWA texture weight invariant is not representable."));
            }
            const auto absolute_x = checked_add(x, static_cast<std::int64_t>(image.origin_x()));
            if (!absolute_x) {
                return std::unexpected(absolute_x.error());
            }
            const auto wrapped_x =
                wrap_texture_index(*absolute_x, image.origin_x(), image.width(), u_wrap);
            if (!wrapped_x) {
                return std::unexpected(wrapped_x.error());
            }
            const auto value = fetch_channel<Scalar>(image, *wrapped_x, *wrapped_y, channel);
            const auto next_weight = accumulated_weight + weight;
            if (!std::isfinite(next_weight)) {
                return std::unexpected(
                    filter_error(core::StatusCode::invalid_argument,
                                 "An EWA texture weight sum is not representable."));
            }
            if (accumulated_weight == Scalar{}) {
                filtered = value;
            } else {
                filtered = std::lerp(filtered, value, weight / next_weight);
            }
            accumulated_weight = next_weight;
        }
    }
    if (!(accumulated_weight > Scalar{}) || !std::isfinite(filtered)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "An EWA texture ellipse did not produce a finite weighted value."));
    }
    return filtered;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
filter_ewa(const HostImageMipChain& mip_chain, const Point2T<Scalar> uv,
           const TextureCoordinateDifferentialsT<Scalar> differentials, const std::uint32_t channel,
           const TextureWrapMode u_wrap, const TextureWrapMode v_wrap,
           const HostImageEwaLimits limits) {
    if (!is_valid_texture_wrap_mode(u_wrap) || !is_valid_texture_wrap_mode(v_wrap)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "An EWA texture wrap mode must be a supported explicit value."));
    }
    if (limits.maximum_anisotropy == 0U ||
        limits.maximum_anisotropy > HostImageEwaMaximumAnisotropy) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "EWA maximum anisotropy must be in the supported range [1, 64]."));
    }
    if (limits.maximum_texel_visits == 0U) {
        return std::unexpected(filter_error(core::StatusCode::invalid_argument,
                                            "An EWA texture visit budget must be non-zero."));
    }
    if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) {
        return std::unexpected(filter_error(core::StatusCode::invalid_argument,
                                            "EWA texture coordinates must both be finite."));
    }
    if (mip_chain.level_count() == 0U) {
        return std::unexpected(filter_error(core::StatusCode::incompatible,
                                            "EWA filtering requires a complete mip chain."));
    }
    const auto base = mip_chain.level(0U);
    if (!base) {
        return std::unexpected(base.error());
    }
    const auto base_status = validate_image(**base);
    if (!base_status) {
        return std::unexpected(base_status.error());
    }
    if (channel >= (*base)->channel_count()) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "An EWA texture channel must address the immutable image snapshot."));
    }

    auto lower_image = *base;
    auto lower_footprint = make_ewa_footprint(*lower_image, differentials, limits);
    if (!lower_footprint) {
        return std::unexpected(lower_footprint.error());
    }
    auto upper_image = HostImageHandle{};
    auto upper_footprint = EwaFootprintT<Scalar>{};
    auto fraction = Scalar{};
    if (lower_footprint->minor_length > Scalar{1}) {
        for (auto level_index = std::uint32_t{1U}; level_index < mip_chain.level_count();
             ++level_index) {
            const auto level = mip_chain.level(level_index);
            if (!level) {
                return std::unexpected(level.error());
            }
            if (const auto status = validate_image(**level); !status) {
                return std::unexpected(status.error());
            }
            const auto footprint = make_ewa_footprint(**level, differentials, limits);
            if (!footprint) {
                return std::unexpected(footprint.error());
            }
            if (footprint->minor_length <= Scalar{1}) {
                const auto lower_log = std::log(lower_footprint->minor_length);
                const auto upper_log = std::log(footprint->minor_length);
                const auto denominator = lower_log - upper_log;
                const auto blend = lower_log / denominator;
                if (!std::isfinite(blend) || denominator <= Scalar{} || blend < Scalar{} ||
                    blend > Scalar{1}) {
                    return std::unexpected(
                        filter_error(core::StatusCode::internal_error,
                                     "EWA mip interpolation is not representable."));
                }
                if (blend == Scalar{1}) {
                    lower_image = *level;
                    lower_footprint = *footprint;
                } else {
                    upper_image = *level;
                    upper_footprint = *footprint;
                    fraction = blend;
                }
                break;
            }
            lower_image = *level;
            lower_footprint = *footprint;
        }
    }
    if (!upper_image && lower_footprint->minor_length > Scalar{1}) {
        lower_footprint = EwaFootprintT<Scalar>{};
    }

    const auto lower_ellipse = make_ewa_ellipse(*lower_image, uv, *lower_footprint);
    if (!lower_ellipse) {
        return std::unexpected(lower_ellipse.error());
    }

    auto upper_ellipse = EwaEllipseT<Scalar>{};
    auto total_visits = lower_ellipse->visit_count;
    if (upper_image) {
        const auto ellipse = make_ewa_ellipse(*upper_image, uv, upper_footprint);
        if (!ellipse) {
            return std::unexpected(ellipse.error());
        }
        if (total_visits > std::numeric_limits<std::uint64_t>::max() - ellipse->visit_count) {
            return std::unexpected(
                filter_error(core::StatusCode::resource_exhausted,
                             "The combined EWA texture visit count is not representable."));
        }
        total_visits += ellipse->visit_count;
        upper_ellipse = *ellipse;
    }
    if (total_visits > limits.maximum_texel_visits) {
        return std::unexpected(
            filter_error(core::StatusCode::resource_exhausted,
                         "The EWA texture footprint exceeds its texel-visit budget."));
    }

    const auto lower = sample_ewa_ellipse(*lower_image, *lower_ellipse, channel, u_wrap, v_wrap);
    if (!lower) {
        return std::unexpected(lower.error());
    }
    if (!upper_image) {
        return *lower;
    }
    const auto upper = sample_ewa_ellipse(*upper_image, upper_ellipse, channel, u_wrap, v_wrap);
    if (!upper) {
        return std::unexpected(upper.error());
    }
    const auto filtered = std::lerp(*lower, *upper, fraction);
    if (!std::isfinite(filtered)) {
        return std::unexpected(
            filter_error(core::StatusCode::invalid_argument,
                         "An EWA texture value must remain finite in the requested precision."));
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

core::Result<TransportScalar>
filter_host_image_ewa_channel(const HostImageMipChain& mip_chain, const Point2 uv,
                              const TextureCoordinateDifferentials differentials,
                              const std::uint32_t channel, const TextureWrapMode u_wrap,
                              const TextureWrapMode v_wrap, const HostImageEwaLimits limits) {
    return filter_ewa(mip_chain, uv, differentials, channel, u_wrap, v_wrap, limits);
}

core::Result<ReferenceScalar>
filter_host_image_ewa_channel(const HostImageMipChain& mip_chain, const ReferencePoint2 uv,
                              const ReferenceTextureCoordinateDifferentials differentials,
                              const std::uint32_t channel, const TextureWrapMode u_wrap,
                              const TextureWrapMode v_wrap, const HostImageEwaLimits limits) {
    return filter_ewa(mip_chain, uv, differentials, channel, u_wrap, v_wrap, limits);
}

} // namespace blackframe::renderer
