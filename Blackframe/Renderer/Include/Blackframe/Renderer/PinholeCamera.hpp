#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/PixelJitter.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/RayCone.hpp>
#include <Blackframe/Renderer/RayDifferential.hpp>
#include <Blackframe/Renderer/RenderConfiguration.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <utility>

namespace blackframe::renderer {

namespace pinhole_camera_detail {

[[nodiscard]] inline core::Error invalid_camera_input(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>>
robust_unit_direction(const Vector3T<Scalar> direction) {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return std::unexpected(invalid_camera_input("A pinhole ray direction must remain finite."));
    }
    const auto maximum_component =
        std::max({std::abs(direction.x), std::abs(direction.y), std::abs(direction.z)});
    if (maximum_component == Scalar{0}) {
        return std::unexpected(invalid_camera_input("A pinhole ray direction must be non-zero."));
    }

    const auto scaled = direction / maximum_component;
    const auto magnitude = std::sqrt(length_squared(scaled));
    const auto result = scaled / magnitude;
    if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
        return std::unexpected(
            invalid_camera_input("Pinhole ray normalization produced a non-finite direction."));
    }
    return result;
}

} // namespace pinhole_camera_detail

// Camera space is right-handed: +X is right, +Y is up, and the camera looks
// along -Z. The supplied frame maps those axes to tangent, bitangent, and
// normal in world space. The field of view is the full vertical angle. Raster
// coordinates cover the complete image extent from the upper-left corner, +X
// goes right, +Y goes down, and pixel (x, y) has its center at
// (x + 0.5, y + 0.5).
template <GeometryScalar Scalar> class PinholeCameraT final {
  public:
    [[nodiscard]] static core::Result<PinholeCameraT>
    create(const Point3T<Scalar> origin, const OrthonormalFrameT<Scalar> orientation,
           const RenderExtent extent, const Scalar vertical_field_of_view_radians,
           const Scalar t_min, const Scalar t_max, const RayMask mask,
           const MediumId current_medium) {
        const auto extent_status = validate_render_extent(extent);
        if (!extent_status.has_value()) {
            return std::unexpected(extent_status.error());
        }
        if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z)) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "A pinhole camera origin must be finite."));
        }
        if (!std::isfinite(vertical_field_of_view_radians) ||
            vertical_field_of_view_radians <= Scalar{0} ||
            vertical_field_of_view_radians >= std::numbers::pi_v<Scalar>) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "A pinhole vertical field of view must be finite and inside (0, pi)."));
        }
        if (!std::isfinite(t_min) || t_min < Scalar{0} || std::isnan(t_max) || t_max < t_min) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "Pinhole ray bounds must form an ordered non-negative interval."));
        }

        const auto half_height = std::tan(vertical_field_of_view_radians / Scalar{2});
        const auto aspect_ratio =
            static_cast<Scalar>(extent.width) / static_cast<Scalar>(extent.height);
        const auto half_width = half_height * aspect_ratio;
        if (!std::isfinite(half_height) || !std::isfinite(half_width) || half_height <= Scalar{0} ||
            half_width <= Scalar{0}) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "Pinhole projection dimensions are not representable."));
        }

        return PinholeCameraT{origin,     orientation,   extent, vertical_field_of_view_radians,
                              half_width, half_height,   t_min,  t_max,
                              mask,       current_medium};
    }

    [[nodiscard]] core::Result<RayT<Scalar>>
    generate_primary_ray(const Point2T<Scalar> raster_sample, const Scalar time) const {
        const auto raster_width = static_cast<Scalar>(extent_.width);
        const auto raster_height = static_cast<Scalar>(extent_.height);
        if (!std::isfinite(raster_sample.x) || !std::isfinite(raster_sample.y) ||
            raster_sample.x < Scalar{0} || raster_sample.x >= raster_width ||
            raster_sample.y < Scalar{0} || raster_sample.y >= raster_height) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "A pinhole raster sample must be finite and inside the half-open image extent."));
        }

        return generate_primary_ray_from_normalized_raster(
            Scalar{2} * raster_sample.x / raster_width - Scalar{1},
            Scalar{1} - Scalar{2} * raster_sample.y / raster_height, time);
    }

    [[nodiscard]] core::Result<RayDifferentialT<Scalar>>
    generate_primary_ray_differential(const Point2T<Scalar> raster_sample,
                                      const Scalar time) const {
        const auto primary = generate_primary_ray(raster_sample, time);
        if (!primary) {
            return std::unexpected(primary.error());
        }
        return generate_primary_ray_differential_from_validated_raster(*primary, raster_sample.x,
                                                                       raster_sample.y, time);
    }

    [[nodiscard]] core::Result<RayConeT<Scalar>>
    generate_primary_ray_cone(const Point2T<Scalar> raster_sample, const Scalar time) const {
        const auto differential = generate_primary_ray_differential(raster_sample, time);
        if (!differential) {
            return std::unexpected(differential.error());
        }
        return ray_cone_from_differential(*differential);
    }

    [[nodiscard]] core::Result<RayT<Scalar>> generate_primary_ray(const PixelSampleT<Scalar> sample,
                                                                  const Scalar time) const {
        if (sample.pixel_x >= extent_.width || sample.pixel_y >= extent_.height) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "A pinhole pixel sample must address a pixel inside the image extent."));
        }
        if (!std::isfinite(sample.offset_x) || !std::isfinite(sample.offset_y) ||
            sample.offset_x < Scalar{0} || sample.offset_x >= Scalar{1} ||
            sample.offset_y < Scalar{0} || sample.offset_y >= Scalar{1}) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "Pinhole subpixel offsets must be finite and inside [0, 1)."));
        }

        const auto raster_width = static_cast<Scalar>(extent_.width);
        const auto raster_height = static_cast<Scalar>(extent_.height);
        const auto twice_pixel_x = Scalar{2} * static_cast<Scalar>(sample.pixel_x);
        const auto twice_pixel_y = Scalar{2} * static_cast<Scalar>(sample.pixel_y);
        return generate_primary_ray_from_normalized_raster(
            (twice_pixel_x + Scalar{2} * sample.offset_x) / raster_width - Scalar{1},
            Scalar{1} - (twice_pixel_y + Scalar{2} * sample.offset_y) / raster_height, time);
    }

    [[nodiscard]] core::Result<RayDifferentialT<Scalar>>
    generate_primary_ray_differential(const PixelSampleT<Scalar> sample, const Scalar time) const {
        const auto primary = generate_primary_ray(sample, time);
        if (!primary) {
            return std::unexpected(primary.error());
        }
        return generate_primary_ray_differential_from_validated_raster(
            *primary, static_cast<Scalar>(sample.pixel_x) + sample.offset_x,
            static_cast<Scalar>(sample.pixel_y) + sample.offset_y, time);
    }

    [[nodiscard]] core::Result<RayConeT<Scalar>>
    generate_primary_ray_cone(const PixelSampleT<Scalar> sample, const Scalar time) const {
        const auto differential = generate_primary_ray_differential(sample, time);
        if (!differential) {
            return std::unexpected(differential.error());
        }
        return ray_cone_from_differential(*differential);
    }

    [[nodiscard]] core::Result<RayT<Scalar>> generate_primary_ray(const PixelSampleIndex index,
                                                                  const PixelJitterMode mode,
                                                                  const Scalar time) const {
        const auto sample = generate_pixel_sample<Scalar>(index, mode);
        if (!sample.has_value()) {
            return std::unexpected(sample.error());
        }
        return generate_primary_ray(*sample, time);
    }

    [[nodiscard]] core::Result<RayDifferentialT<Scalar>>
    generate_primary_ray_differential(const PixelSampleIndex index, const PixelJitterMode mode,
                                      const Scalar time) const {
        const auto sample = generate_pixel_sample<Scalar>(index, mode);
        if (!sample.has_value()) {
            return std::unexpected(sample.error());
        }
        return generate_primary_ray_differential(*sample, time);
    }

    [[nodiscard]] core::Result<RayConeT<Scalar>>
    generate_primary_ray_cone(const PixelSampleIndex index, const PixelJitterMode mode,
                              const Scalar time) const {
        const auto differential = generate_primary_ray_differential(index, mode, time);
        if (!differential) {
            return std::unexpected(differential.error());
        }
        return ray_cone_from_differential(*differential);
    }

    [[nodiscard]] constexpr const Point3T<Scalar>& origin() const noexcept {
        return origin_;
    }

    [[nodiscard]] constexpr const OrthonormalFrameT<Scalar>& orientation() const noexcept {
        return orientation_;
    }

    [[nodiscard]] constexpr RenderExtent extent() const noexcept {
        return extent_;
    }

    [[nodiscard]] constexpr Scalar vertical_field_of_view_radians() const noexcept {
        return vertical_field_of_view_radians_;
    }

  private:
    [[nodiscard]] core::Result<RayDifferentialT<Scalar>>
    generate_primary_ray_differential_from_validated_raster(const RayT<Scalar> primary,
                                                            const Scalar raster_x,
                                                            const Scalar raster_y,
                                                            const Scalar time) const {
        const auto raster_width = static_cast<Scalar>(extent_.width);
        const auto raster_height = static_cast<Scalar>(extent_.height);
        // Differential samples deliberately extrapolate one pixel beyond the right and bottom
        // edges. Clamping or switching to a backward difference would collapse or flip the
        // footprint at the film boundary.
        const auto rx = generate_primary_ray_from_normalized_raster(
            Scalar{2} * (raster_x + Scalar{1}) / raster_width - Scalar{1},
            Scalar{1} - Scalar{2} * raster_y / raster_height, time);
        if (!rx) {
            return std::unexpected(rx.error());
        }
        const auto ry = generate_primary_ray_from_normalized_raster(
            Scalar{2} * raster_x / raster_width - Scalar{1},
            Scalar{1} - Scalar{2} * (raster_y + Scalar{1}) / raster_height, time);
        if (!ry) {
            return std::unexpected(ry.error());
        }
        if (rx->direction() == primary.direction() || ry->direction() == primary.direction()) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "A one-pixel pinhole differential is not representable at this extent and field "
                "of view."));
        }
        return RayDifferentialT<Scalar>::create(primary, origin_, rx->direction(), origin_,
                                                ry->direction());
    }

    [[nodiscard]] core::Result<RayT<Scalar>> generate_primary_ray_from_normalized_raster(
        const Scalar normalized_x, const Scalar normalized_y, const Scalar time) const {
        if (!std::isfinite(time)) {
            return std::unexpected(pinhole_camera_detail::invalid_camera_input(
                "A pinhole primary-ray time must be finite."));
        }

        const auto camera_direction = Vector3T<Scalar>{
            .x = normalized_x * half_width_,
            .y = normalized_y * half_height_,
            .z = Scalar{-1},
        };
        const auto unit_camera_direction =
            pinhole_camera_detail::robust_unit_direction(camera_direction);
        if (!unit_camera_direction.has_value()) {
            return std::unexpected(unit_camera_direction.error());
        }
        const auto world_direction = pinhole_camera_detail::robust_unit_direction(
            orientation_.to_world(*unit_camera_direction));
        if (!world_direction.has_value()) {
            return std::unexpected(world_direction.error());
        }

        return RayT<Scalar>::create(origin_, *world_direction, t_min_, t_max_, time, mask_,
                                    current_medium_);
    }

    constexpr PinholeCameraT(const Point3T<Scalar> origin,
                             const OrthonormalFrameT<Scalar> orientation, const RenderExtent extent,
                             const Scalar vertical_field_of_view_radians, const Scalar half_width,
                             const Scalar half_height, const Scalar t_min, const Scalar t_max,
                             const RayMask mask, const MediumId current_medium) noexcept
        : origin_{origin}, orientation_{orientation}, extent_{extent},
          vertical_field_of_view_radians_{vertical_field_of_view_radians}, half_width_{half_width},
          half_height_{half_height}, t_min_{t_min}, t_max_{t_max}, mask_{mask},
          current_medium_{current_medium} {}

    Point3T<Scalar> origin_;
    OrthonormalFrameT<Scalar> orientation_;
    RenderExtent extent_;
    Scalar vertical_field_of_view_radians_;
    Scalar half_width_;
    Scalar half_height_;
    Scalar t_min_;
    Scalar t_max_;
    RayMask mask_;
    MediumId current_medium_;
};

using PinholeCamera = PinholeCameraT<TransportScalar>;
using ReferencePinholeCamera = PinholeCameraT<ReferenceScalar>;

} // namespace blackframe::renderer
