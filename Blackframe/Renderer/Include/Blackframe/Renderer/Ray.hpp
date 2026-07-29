#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {

struct MediumId final {
    std::uint32_t value{};

    [[nodiscard]] constexpr bool operator==(const MediumId&) const noexcept = default;
};

inline constexpr auto VacuumMedium = MediumId{.value = 0};

using RayMask = std::uint32_t;
inline constexpr auto AllRayVisibility = std::numeric_limits<RayMask>::max();

template <GeometryScalar Scalar> class RayT final {
  public:
    [[nodiscard]] static core::Result<RayT> create(const Point3T<Scalar> origin,
                                                   const Vector3T<Scalar> direction,
                                                   const Scalar t_min, const Scalar t_max,
                                                   const Scalar time, const RayMask mask,
                                                   const MediumId current_medium) {
        if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z) ||
            !std::isfinite(direction.x) || !std::isfinite(direction.y) ||
            !std::isfinite(direction.z) ||
            (direction.x == Scalar{0} && direction.y == Scalar{0} && direction.z == Scalar{0})) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "A ray requires a finite origin and a finite non-zero direction.",
            });
        }
        if (!std::isfinite(t_min) || t_min < Scalar{0} || std::isnan(t_max) || t_max < t_min) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "A ray requires an ordered non-negative parameter interval.",
            });
        }
        if (!std::isfinite(time)) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "A ray time must be finite.",
            });
        }

        return RayT{origin, direction, t_min, t_max, time, mask, current_medium};
    }

    [[nodiscard]] constexpr const Point3T<Scalar>& origin() const noexcept {
        return origin_;
    }
    [[nodiscard]] constexpr const Vector3T<Scalar>& direction() const noexcept {
        return direction_;
    }
    [[nodiscard]] constexpr Scalar t_min() const noexcept {
        return t_min_;
    }
    [[nodiscard]] constexpr Scalar t_max() const noexcept {
        return t_max_;
    }
    [[nodiscard]] constexpr Scalar time() const noexcept {
        return time_;
    }
    [[nodiscard]] constexpr RayMask mask() const noexcept {
        return mask_;
    }
    [[nodiscard]] constexpr MediumId current_medium() const noexcept {
        return current_medium_;
    }

    [[nodiscard]] constexpr bool contains_parameter(const Scalar parameter) const noexcept {
        return !std::isnan(parameter) && parameter >= t_min_ && parameter <= t_max_;
    }

    [[nodiscard]] core::Result<Point3T<Scalar>> at(const Scalar parameter) const {
        if (!std::isfinite(parameter) || !contains_parameter(parameter)) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "A ray parameter must be finite and inside [tMin, tMax].",
            });
        }
        const auto point = origin_ + direction_ * parameter;
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Ray evaluation produced a non-finite point.",
            });
        }
        return point;
    }

  private:
    constexpr RayT(const Point3T<Scalar> origin, const Vector3T<Scalar> direction,
                   const Scalar t_min, const Scalar t_max, const Scalar time, const RayMask mask,
                   const MediumId current_medium) noexcept
        : origin_{origin}, direction_{direction}, t_min_{t_min}, t_max_{t_max}, time_{time},
          mask_{mask}, current_medium_{current_medium} {}

    Point3T<Scalar> origin_;
    Vector3T<Scalar> direction_;
    Scalar t_min_;
    Scalar t_max_;
    Scalar time_;
    RayMask mask_;
    MediumId current_medium_;
};

using Ray = RayT<TransportScalar>;
using ReferenceRay = RayT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<MediumId>);
static_assert(std::is_trivially_copyable_v<MediumId>);
static_assert(sizeof(MediumId) == sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<Ray>);
static_assert(std::is_trivially_copyable_v<Ray>);
static_assert(std::is_standard_layout_v<ReferenceRay>);
static_assert(std::is_trivially_copyable_v<ReferenceRay>);

} // namespace blackframe::renderer
