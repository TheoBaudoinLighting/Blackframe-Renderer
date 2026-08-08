#include <Blackframe/Renderer/HostImageMipChain.hpp>
#include <Blackframe/Renderer/HostSurfaceMaps.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Error surface_map_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Vector3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar fused_dot(const Vector3T<Scalar> left, const Vector3T<Scalar> right) noexcept {
    return std::fma(left.x, right.x, std::fma(left.y, right.y, left.z * right.z));
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar fused_dot(const Normal3T<Scalar> left, const Vector3T<Scalar> right) noexcept {
    return std::fma(left.x, right.x, std::fma(left.y, right.y, left.z * right.z));
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar fused_dot(const Normal3T<Scalar> left, const Normal3T<Scalar> right) noexcept {
    return std::fma(left.x, right.x, std::fma(left.y, right.y, left.z * right.z));
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> normalized_vector(const Vector3T<Scalar> value,
                                                               const char* const message) {
    const auto scale = std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
    if (!finite(value) || !std::isfinite(scale) || scale == Scalar{0}) {
        return std::unexpected(surface_map_error(message));
    }
    const auto scaled = value / scale;
    const auto squared_length = fused_dot(scaled, scaled);
    if (!std::isfinite(squared_length) || squared_length <= Scalar{0}) {
        return std::unexpected(surface_map_error(message));
    }
    const auto length = std::sqrt(squared_length);
    if (!std::isfinite(length) || length <= Scalar{0}) {
        return std::unexpected(surface_map_error(message));
    }
    return scaled / length;
}

template <GeometryScalar Scalar>
[[nodiscard]] Normal3T<Scalar> as_normal(const Vector3T<Scalar> value) noexcept {
    return {.x = value.x, .y = value.y, .z = value.z};
}

template <GeometryScalar Scalar>
[[nodiscard]] Vector3T<Scalar> as_vector(const Normal3T<Scalar> value) noexcept {
    return {.x = value.x, .y = value.y, .z = value.z};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Status validate_ewa_options(const TextureWrapMode u_wrap,
                                                const TextureWrapMode v_wrap,
                                                const HostImageEwaLimits limits) {
    if (!is_valid_texture_wrap_mode(u_wrap) || !is_valid_texture_wrap_mode(v_wrap)) {
        return std::unexpected(surface_map_error(
            "A surface map texture wrap mode must be a supported explicit value."));
    }
    if (limits.maximum_anisotropy == 0U ||
        limits.maximum_anisotropy > HostImageEwaMaximumAnisotropy) {
        return std::unexpected(
            surface_map_error("A surface map EWA maximum anisotropy must lie in [1, 64]."));
    }
    if (limits.maximum_texel_visits == 0U) {
        return std::unexpected(
            surface_map_error("A surface map EWA texel-visit budget must be non-zero."));
    }
    return {};
}

[[nodiscard]] core::Result<HostImageHandle>
validate_data_source(const HostImageMipChain& mip_chain) {
    const auto source = mip_chain.source_image();
    if (!source) {
        return std::unexpected(
            surface_map_error("A surface map mip chain must own a source image."));
    }
    if (source->source_color_space() != TextureColorSpace::data ||
        source->storage_color_space() != TextureColorSpace::data) {
        return std::unexpected(surface_map_error(
            "Normal and bump maps must be explicitly tagged and stored as data."));
    }
    if (source->width() == 0U || source->height() == 0U || source->channel_count() == 0U) {
        return std::unexpected(surface_map_error("A surface map source image must be non-empty."));
    }
    return source;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Status
validate_footprint(const TextureCoordinateDifferentialsT<Scalar> differentials) {
    const auto values = std::array<Scalar, 4U>{differentials.dudx, differentials.dvdx,
                                               differentials.dudy, differentials.dvdy};
    if (!std::ranges::all_of(values, [](const Scalar value) { return std::isfinite(value); })) {
        return std::unexpected(surface_map_error("A surface map EWA footprint must be finite."));
    }
    const auto scale = std::max({std::abs(differentials.dudx), std::abs(differentials.dvdx),
                                 std::abs(differentials.dudy), std::abs(differentials.dvdy)});
    if (!std::isfinite(scale) || scale == Scalar{0}) {
        return std::unexpected(
            surface_map_error("A surface map EWA footprint must have full rank, not zero area."));
    }
    const auto dudx = differentials.dudx / scale;
    const auto dvdx = differentials.dvdx / scale;
    const auto dudy = differentials.dudy / scale;
    const auto dvdy = differentials.dvdy / scale;
    const auto determinant = std::fma(dudx, dvdy, -(dvdx * dudy));
    if (!std::isfinite(determinant) || determinant == Scalar{0}) {
        return std::unexpected(
            surface_map_error("A surface map EWA footprint must have full rank, not zero area."));
    }
    return {};
}

template <GeometryScalar Scalar> struct TangentFrame final {
    Vector3T<Scalar> tangent;
    Vector3T<Scalar> bitangent;
};

template <GeometryScalar Scalar> struct ProjectedSurfaceTangents final {
    Vector3T<Scalar> dpdu;
    Vector3T<Scalar> dpdv;
};

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> normalized_cross(Vector3T<Scalar> left,
                                                              Vector3T<Scalar> right);

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<TangentFrame<Scalar>>
surface_tangent_frame(const SurfaceInteractionT<Scalar>& interaction) {
    const auto shading_normal = interaction.shading_normal();
    const auto shading_vector = as_vector(shading_normal);
    const auto tangent_alignment = fused_dot(shading_normal, interaction.dpdu());
    const auto projected_tangent =
        Vector3T<Scalar>{.x = std::fma(-tangent_alignment, shading_normal.x, interaction.dpdu().x),
                         .y = std::fma(-tangent_alignment, shading_normal.y, interaction.dpdu().y),
                         .z = std::fma(-tangent_alignment, shading_normal.z, interaction.dpdu().z)};
    const auto tangent = normalized_vector(
        projected_tangent,
        "A surface map requires a finite non-zero dpdu projected onto the shading tangent plane.");
    if (!tangent) {
        return std::unexpected(tangent.error());
    }
    const auto canonical_bitangent =
        normalized_vector(cross(shading_vector, *tangent),
                          "A surface map requires a finite non-degenerate shading tangent frame.");
    if (!canonical_bitangent) {
        return std::unexpected(canonical_bitangent.error());
    }

    const auto dpdv_scale =
        std::max({std::abs(interaction.dpdv().x), std::abs(interaction.dpdv().y),
                  std::abs(interaction.dpdv().z)});
    if (!std::isfinite(dpdv_scale) || dpdv_scale == Scalar{0}) {
        return std::unexpected(surface_map_error(
            "A surface map requires finite non-zero dpdv to determine tangent handedness."));
    }
    const auto handedness_alignment =
        fused_dot(*canonical_bitangent, interaction.dpdv() / dpdv_scale);
    if (!std::isfinite(handedness_alignment) || handedness_alignment == Scalar{0}) {
        return std::unexpected(surface_map_error(
            "A surface map dpdv must determine an unambiguous tangent handedness."));
    }
    return TangentFrame<Scalar>{
        .tangent = *tangent,
        .bitangent =
            *canonical_bitangent * (handedness_alignment > Scalar{0} ? Scalar{1} : Scalar{-1}),
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<ProjectedSurfaceTangents<Scalar>>
projected_surface_tangents(const SurfaceInteractionT<Scalar>& interaction) {
    const auto shading = interaction.shading_normal();
    const auto project = [shading](const Vector3T<Scalar> derivative) {
        const auto alignment = fused_dot(shading, derivative);
        return Vector3T<Scalar>{
            .x = std::fma(-alignment, shading.x, derivative.x),
            .y = std::fma(-alignment, shading.y, derivative.y),
            .z = std::fma(-alignment, shading.z, derivative.z),
        };
    };
    const auto tangent_u = project(interaction.dpdu());
    const auto tangent_v = project(interaction.dpdv());
    const auto normalized_u = normalized_vector(
        tangent_u, "A filtered bump map requires finite non-zero dpdu projected onto Ns.");
    const auto normalized_v = normalized_vector(
        tangent_v, "A filtered bump map requires finite non-zero dpdv projected onto Ns.");
    if (!normalized_u) {
        return std::unexpected(normalized_u.error());
    }
    if (!normalized_v) {
        return std::unexpected(normalized_v.error());
    }
    const auto orientation = normalized_cross(tangent_u, tangent_v);
    if (!orientation) {
        return std::unexpected(surface_map_error(
            "A filtered bump map requires linearly independent projected surface tangents."));
    }
    const auto handedness = fused_dot(shading, *orientation);
    if (!std::isfinite(handedness) || handedness == Scalar{0}) {
        return std::unexpected(surface_map_error(
            "A filtered bump map requires unambiguous projected tangent handedness."));
    }
    return ProjectedSurfaceTangents<Scalar>{.dpdu = tangent_u, .dpdv = tangent_v};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Status validate_common(const HostImageMipChain& mip_chain,
                                           const TextureCoordinateDifferentialsT<Scalar> footprint,
                                           const TextureWrapMode u_wrap,
                                           const TextureWrapMode v_wrap,
                                           const HostImageEwaLimits limits) {
    const auto source = validate_data_source(mip_chain);
    if (!source) {
        return std::unexpected(source.error());
    }
    const auto option_status = validate_ewa_options<Scalar>(u_wrap, v_wrap, limits);
    if (!option_status) {
        return option_status;
    }
    return validate_footprint(footprint);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
sample(const HostImageMipChain& mip_chain, const Point2T<Scalar> uv,
       const TextureCoordinateDifferentialsT<Scalar> footprint, const std::uint32_t channel,
       const TextureWrapMode u_wrap, const TextureWrapMode v_wrap,
       const HostImageEwaLimits limits) {
    return filter_host_image_ewa_channel(mip_chain, uv, footprint, channel, u_wrap, v_wrap, limits);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Normal3T<Scalar>>
evaluate_normal_map(const HostImageMipChain& mip_chain,
                    const SurfaceInteractionT<Scalar>& interaction,
                    const TextureCoordinateDifferentialsT<Scalar> differentials,
                    const HostNormalMapOptions options) {
    const auto common_status = validate_common(mip_chain, differentials, options.u_wrap,
                                               options.v_wrap, options.ewa_limits);
    if (!common_status) {
        return std::unexpected(common_status.error());
    }
    if (!is_valid_tangent_space_normal_y_convention(options.y_convention)) {
        return std::unexpected(surface_map_error(
            "A tangent-space normal Y convention must be a supported explicit value."));
    }
    const auto source = mip_chain.source_image();
    if (options.channels.x >= source->channel_count() ||
        options.channels.y >= source->channel_count() ||
        options.channels.z >= source->channel_count()) {
        return std::unexpected(surface_map_error(
            "Every tangent-space normal channel must exist in the source image."));
    }
    if (options.channels.x == options.channels.y || options.channels.x == options.channels.z ||
        options.channels.y == options.channels.z) {
        return std::unexpected(
            surface_map_error("Tangent-space normal X, Y, and Z channels must be distinct."));
    }
    const auto frame = surface_tangent_frame(interaction);
    if (!frame) {
        return std::unexpected(frame.error());
    }

    const auto x = sample(mip_chain, interaction.uv(), differentials, options.channels.x,
                          options.u_wrap, options.v_wrap, options.ewa_limits);
    if (!x) {
        return std::unexpected(x.error());
    }
    const auto y = sample(mip_chain, interaction.uv(), differentials, options.channels.y,
                          options.u_wrap, options.v_wrap, options.ewa_limits);
    if (!y) {
        return std::unexpected(y.error());
    }
    const auto z = sample(mip_chain, interaction.uv(), differentials, options.channels.z,
                          options.u_wrap, options.v_wrap, options.ewa_limits);
    if (!z) {
        return std::unexpected(z.error());
    }
    const auto encoded = std::array<Scalar, 3U>{*x, *y, *z};
    if (!std::ranges::all_of(encoded, [](const Scalar component) {
            return std::isfinite(component) && component >= Scalar{0} && component <= Scalar{1};
        })) {
        return std::unexpected(surface_map_error(
            "Filtered tangent-space normal components must be finite and lie in [0, 1]."));
    }
    auto decoded = Vector3T<Scalar>{.x = std::fma(Scalar{2}, *x, Scalar{-1}),
                                    .y = std::fma(Scalar{2}, *y, Scalar{-1}),
                                    .z = std::fma(Scalar{2}, *z, Scalar{-1})};
    if (decoded.z <= Scalar{0}) {
        return std::unexpected(surface_map_error(
            "A tangent-space normal must lie in the open upper tangent hemisphere."));
    }
    const auto normalized_decoded = normalized_vector(
        decoded, "A filtered tangent-space normal must have non-zero finite length.");
    if (!normalized_decoded) {
        return std::unexpected(normalized_decoded.error());
    }
    decoded = *normalized_decoded;
    if (options.y_convention == TangentSpaceNormalYConvention::negative_v) {
        decoded.y = -decoded.y;
    }

    const auto shading_vector = as_vector(interaction.shading_normal());
    const auto world = Vector3T<Scalar>{
        .x = std::fma(frame->tangent.x, decoded.x,
                      std::fma(frame->bitangent.x, decoded.y, shading_vector.x * decoded.z)),
        .y = std::fma(frame->tangent.y, decoded.x,
                      std::fma(frame->bitangent.y, decoded.y, shading_vector.y * decoded.z)),
        .z = std::fma(frame->tangent.z, decoded.x,
                      std::fma(frame->bitangent.z, decoded.y, shading_vector.z * decoded.z)),
    };
    const auto normalized_world = normalized_vector(
        world, "A transformed tangent-space normal must have non-zero finite length.");
    if (!normalized_world) {
        return std::unexpected(normalized_world.error());
    }
    const auto result = as_normal(*normalized_world);
    if (!std::isfinite(fused_dot(interaction.geometric_normal(), result)) ||
        fused_dot(interaction.geometric_normal(), result) <= Scalar{0}) {
        return std::unexpected(surface_map_error(
            "A tangent-space normal map result must remain above the geometric surface."));
    }
    return result;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> normalized_cross(const Vector3T<Scalar> left,
                                                              const Vector3T<Scalar> right) {
    const auto left_scale = std::max({std::abs(left.x), std::abs(left.y), std::abs(left.z)});
    const auto right_scale = std::max({std::abs(right.x), std::abs(right.y), std::abs(right.z)});
    if (!finite(left) || !finite(right) || !std::isfinite(left_scale) ||
        !std::isfinite(right_scale) || left_scale == Scalar{0} || right_scale == Scalar{0}) {
        return std::unexpected(
            surface_map_error("Filtered bump tangents must be finite and non-zero."));
    }
    return normalized_vector(cross(left / left_scale, right / right_scale),
                             "Filtered bump tangents must be linearly independent.");
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Normal3T<Scalar>>
evaluate_bump_map(const HostImageMipChain& mip_chain,
                  const SurfaceInteractionT<Scalar>& interaction,
                  const TextureCoordinateDifferentialsT<Scalar> differentials,
                  const HostBumpMapOptionsT<Scalar> options) {
    const auto common_status = validate_common(mip_chain, differentials, options.u_wrap,
                                               options.v_wrap, options.ewa_limits);
    if (!common_status) {
        return std::unexpected(common_status.error());
    }
    const auto source = mip_chain.source_image();
    if (options.height_channel >= source->channel_count()) {
        return std::unexpected(
            surface_map_error("The bump height channel must exist in the source image."));
    }
    if (!std::isfinite(options.displacement_scale)) {
        return std::unexpected(surface_map_error("A bump displacement scale must be finite."));
    }
    const auto projected_tangents = projected_surface_tangents(interaction);
    if (!projected_tangents) {
        return std::unexpected(projected_tangents.error());
    }

    const auto du = std::max(Scalar{1} / static_cast<Scalar>(source->width()),
                             std::hypot(differentials.dudx, differentials.dudy));
    const auto dv = std::max(Scalar{1} / static_cast<Scalar>(source->height()),
                             std::hypot(differentials.dvdx, differentials.dvdy));
    if (!std::isfinite(du) || !std::isfinite(dv) || du <= Scalar{0} || dv <= Scalar{0}) {
        return std::unexpected(surface_map_error(
            "Filtered bump finite-difference steps must be finite and positive."));
    }
    const auto uv = interaction.uv();
    const auto u_minus = Point2T<Scalar>{.x = uv.x - du, .y = uv.y};
    const auto u_plus = Point2T<Scalar>{.x = uv.x + du, .y = uv.y};
    const auto v_minus = Point2T<Scalar>{.x = uv.x, .y = uv.y - dv};
    const auto v_plus = Point2T<Scalar>{.x = uv.x, .y = uv.y + dv};
    if (!std::isfinite(u_minus.x) || !std::isfinite(u_plus.x) || !std::isfinite(v_minus.y) ||
        !std::isfinite(v_plus.y)) {
        return std::unexpected(surface_map_error(
            "Filtered bump finite-difference coordinates must be representable."));
    }
    const auto height_u_minus = sample(mip_chain, u_minus, differentials, options.height_channel,
                                       options.u_wrap, options.v_wrap, options.ewa_limits);
    if (!height_u_minus) {
        return std::unexpected(height_u_minus.error());
    }
    const auto height_u_plus = sample(mip_chain, u_plus, differentials, options.height_channel,
                                      options.u_wrap, options.v_wrap, options.ewa_limits);
    if (!height_u_plus) {
        return std::unexpected(height_u_plus.error());
    }
    const auto height_v_minus = sample(mip_chain, v_minus, differentials, options.height_channel,
                                       options.u_wrap, options.v_wrap, options.ewa_limits);
    if (!height_v_minus) {
        return std::unexpected(height_v_minus.error());
    }
    const auto height_v_plus = sample(mip_chain, v_plus, differentials, options.height_channel,
                                      options.u_wrap, options.v_wrap, options.ewa_limits);
    if (!height_v_plus) {
        return std::unexpected(height_v_plus.error());
    }
    const auto heights =
        std::array<Scalar, 4U>{*height_u_minus, *height_u_plus, *height_v_minus, *height_v_plus};
    if (!std::ranges::all_of(heights, [](const Scalar height) { return std::isfinite(height); })) {
        return std::unexpected(surface_map_error("Filtered bump heights must be finite."));
    }
    const auto dhdu = (*height_u_plus - *height_u_minus) / (Scalar{2} * du);
    const auto dhdv = (*height_v_plus - *height_v_minus) / (Scalar{2} * dv);
    const auto scaled_dhdu = options.displacement_scale * dhdu;
    const auto scaled_dhdv = options.displacement_scale * dhdv;
    if (!std::isfinite(dhdu) || !std::isfinite(dhdv) || !std::isfinite(scaled_dhdu) ||
        !std::isfinite(scaled_dhdv)) {
        return std::unexpected(
            surface_map_error("Filtered bump derivatives must be finite and representable."));
    }
    const auto shading = interaction.shading_normal();
    const auto tangent_u = Vector3T<Scalar>{
        .x = std::fma(shading.x, scaled_dhdu, projected_tangents->dpdu.x),
        .y = std::fma(shading.y, scaled_dhdu, projected_tangents->dpdu.y),
        .z = std::fma(shading.z, scaled_dhdu, projected_tangents->dpdu.z),
    };
    const auto tangent_v = Vector3T<Scalar>{
        .x = std::fma(shading.x, scaled_dhdv, projected_tangents->dpdv.x),
        .y = std::fma(shading.y, scaled_dhdv, projected_tangents->dpdv.y),
        .z = std::fma(shading.z, scaled_dhdv, projected_tangents->dpdv.z),
    };
    auto normal = normalized_cross(tangent_u, tangent_v);
    if (!normal) {
        return std::unexpected(normal.error());
    }
    auto shading_alignment = fused_dot(interaction.shading_normal(), *normal);
    if (!std::isfinite(shading_alignment)) {
        return std::unexpected(
            surface_map_error("A filtered bump normal must have finite shading alignment."));
    }
    if (shading_alignment < Scalar{0}) {
        *normal = -*normal;
        shading_alignment = -shading_alignment;
    }
    const auto result = as_normal(*normal);
    if (!(shading_alignment > Scalar{0})) {
        return std::unexpected(surface_map_error(
            "A filtered bump normal must remain strictly above the original shading surface."));
    }
    const auto geometric_alignment = fused_dot(interaction.geometric_normal(), result);
    if (!std::isfinite(geometric_alignment) || geometric_alignment <= Scalar{0}) {
        return std::unexpected(
            surface_map_error("A filtered bump normal must remain above the geometric surface."));
    }
    return result;
}

} // namespace

core::Result<Normal3> evaluate_host_tangent_space_normal_map(
    const HostImageMipChain& mip_chain, const SurfaceInteraction& interaction,
    const TextureCoordinateDifferentials differentials, const HostNormalMapOptions options) {
    return evaluate_normal_map(mip_chain, interaction, differentials, options);
}

core::Result<ReferenceNormal3>
evaluate_host_tangent_space_normal_map(const HostImageMipChain& mip_chain,
                                       const ReferenceSurfaceInteraction& interaction,
                                       const ReferenceTextureCoordinateDifferentials differentials,
                                       const HostNormalMapOptions options) {
    return evaluate_normal_map(mip_chain, interaction, differentials, options);
}

core::Result<Normal3> evaluate_host_filtered_bump_map(
    const HostImageMipChain& mip_chain, const SurfaceInteraction& interaction,
    const TextureCoordinateDifferentials differentials, const HostBumpMapOptions options) {
    return evaluate_bump_map(mip_chain, interaction, differentials, options);
}

core::Result<ReferenceNormal3>
evaluate_host_filtered_bump_map(const HostImageMipChain& mip_chain,
                                const ReferenceSurfaceInteraction& interaction,
                                const ReferenceTextureCoordinateDifferentials differentials,
                                const ReferenceHostBumpMapOptions options) {
    return evaluate_bump_map(mip_chain, interaction, differentials, options);
}

} // namespace blackframe::renderer
