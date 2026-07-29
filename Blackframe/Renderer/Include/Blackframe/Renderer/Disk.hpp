#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Plane.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <algorithm>
#include <cmath>
#include <optional>

namespace blackframe::renderer {

template <GeometryScalar Scalar> struct DiskHitT final {
    Scalar parameter{};
    Point3T<Scalar> position{};
    Normal3T<Scalar> geometric_normal{};

    [[nodiscard]] constexpr bool operator==(const DiskHitT&) const noexcept = default;
};

using DiskHit = DiskHitT<TransportScalar>;
using ReferenceDiskHit = DiskHitT<ReferenceScalar>;

// A disk inherits its two-sided orientation and exact ray clipping from its
// supporting plane. Points exactly on the circular boundary are included.
template <GeometryScalar Scalar> class DiskT final {
  public:
    [[nodiscard]] static core::Result<DiskT>
    create(const Point3T<Scalar> center, const Normal3T<Scalar> normal, const Scalar radius) {
        if (!std::isfinite(radius) || radius <= Scalar{0}) {
            return std::unexpected(
                plane_detail::plane_error("A disk radius must be finite and positive."));
        }
        auto plane = PlaneT<Scalar>::create(center, normal);
        if (!plane.has_value()) {
            return std::unexpected(plane.error());
        }
        return DiskT{*plane, radius};
    }

    [[nodiscard]] core::Result<std::optional<DiskHitT<Scalar>>>
    intersect(const RayT<Scalar>& ray) const {
        const auto plane_intersection = plane_.intersect(ray);
        if (!plane_intersection.has_value()) {
            return std::unexpected(plane_intersection.error());
        }
        if (!plane_intersection->has_value()) {
            return std::optional<DiskHitT<Scalar>>{};
        }

        const auto radial = (**plane_intersection).position - plane_.point();
        if (!std::isfinite(radial.x) || !std::isfinite(radial.y) || !std::isfinite(radial.z)) {
            return std::unexpected(
                plane_detail::plane_error("Disk intersection cannot represent its radial offset."));
        }

        const auto radial_maximum =
            std::max({std::abs(radial.x), std::abs(radial.y), std::abs(radial.z), radius_});
        int radial_exponent = 0;
        static_cast<void>(std::frexp(radial_maximum, &radial_exponent));
        const auto scaled_radial = Vector3T<Scalar>{
            .x = std::ldexp(radial.x, -radial_exponent),
            .y = std::ldexp(radial.y, -radial_exponent),
            .z = std::ldexp(radial.z, -radial_exponent),
        };
        const auto scaled_radius = std::ldexp(radius_, -radial_exponent);
        if (!plane_detail::scaling_preserves(radial.x, scaled_radial.x) ||
            !plane_detail::scaling_preserves(radial.y, scaled_radial.y) ||
            !plane_detail::scaling_preserves(radial.z, scaled_radial.z) ||
            !plane_detail::scaling_preserves(radius_, scaled_radius)) {
            return std::unexpected(
                plane_detail::plane_error("Disk radius scaling is not representable."));
        }

        const auto radius_difference =
            plane_detail::exact_squared_radius_difference(scaled_radial, scaled_radius);
        if (!radius_difference.has_value()) {
            return std::unexpected(radius_difference.error());
        }
        if (radius_difference->sign > 0) {
            return std::optional<DiskHitT<Scalar>>{};
        }

        const auto& plane_hit = **plane_intersection;
        return std::optional<DiskHitT<Scalar>>{DiskHitT<Scalar>{
            .parameter = plane_hit.parameter,
            .position = plane_hit.position,
            .geometric_normal = plane_hit.geometric_normal,
        }};
    }

    [[nodiscard]] constexpr const Point3T<Scalar>& center() const noexcept {
        return plane_.point();
    }

    [[nodiscard]] constexpr const OrthonormalFrameT<Scalar>& orientation() const noexcept {
        return plane_.orientation();
    }

    [[nodiscard]] constexpr Scalar radius() const noexcept {
        return radius_;
    }

  private:
    constexpr DiskT(const PlaneT<Scalar> plane, const Scalar radius) noexcept
        : plane_{plane}, radius_{radius} {}

    PlaneT<Scalar> plane_;
    Scalar radius_;
};

using Disk = DiskT<TransportScalar>;
using ReferenceDisk = DiskT<ReferenceScalar>;

} // namespace blackframe::renderer
