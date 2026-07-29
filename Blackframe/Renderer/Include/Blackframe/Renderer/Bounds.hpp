#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Interval.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace blackframe::renderer {
namespace bounds_detail {

template <GeometryScalar Scalar> [[nodiscard]] bool valid(const Point2T<Scalar> point) noexcept {
    return !std::isnan(point.x) && !std::isnan(point.y);
}

template <GeometryScalar Scalar> [[nodiscard]] bool valid(const Point3T<Scalar> point) noexcept {
    return !std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z);
}

template <typename Bounds> [[nodiscard]] core::Result<Bounds> bounds_error() {
    return std::unexpected(core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = "Bounds require ordered coordinates without NaN.",
    });
}

} // namespace bounds_detail

template <GeometryScalar Scalar> class Bounds2T final {
  public:
    [[nodiscard]] static core::Result<Bounds2T>
    from_minimum_maximum(const Point2T<Scalar> minimum, const Point2T<Scalar> maximum) {
        if (!bounds_detail::valid(minimum) || !bounds_detail::valid(maximum) ||
            minimum.x > maximum.x || minimum.y > maximum.y) {
            return bounds_detail::bounds_error<Bounds2T>();
        }
        return Bounds2T{minimum, maximum, false};
    }

    [[nodiscard]] static core::Result<Bounds2T> from_points(const Point2T<Scalar> first,
                                                            const Point2T<Scalar> second) {
        if (!bounds_detail::valid(first) || !bounds_detail::valid(second)) {
            return bounds_detail::bounds_error<Bounds2T>();
        }
        return Bounds2T{
            Point2T<Scalar>{.x = std::min(first.x, second.x), .y = std::min(first.y, second.y)},
            Point2T<Scalar>{.x = std::max(first.x, second.x), .y = std::max(first.y, second.y)},
            false,
        };
    }

    [[nodiscard]] static constexpr Bounds2T empty() noexcept {
        return Bounds2T{Point2T<Scalar>{}, Point2T<Scalar>{}, true};
    }

    [[nodiscard]] static constexpr Bounds2T unbounded() noexcept {
        constexpr auto infinity = std::numeric_limits<Scalar>::infinity();
        return Bounds2T{Point2T<Scalar>{.x = -infinity, .y = -infinity},
                        Point2T<Scalar>{.x = infinity, .y = infinity}, false};
    }

    [[nodiscard]] constexpr bool is_empty() const noexcept {
        return empty_;
    }
    [[nodiscard]] constexpr const Point2T<Scalar>& minimum() const noexcept {
        return minimum_;
    }
    [[nodiscard]] constexpr const Point2T<Scalar>& maximum() const noexcept {
        return maximum_;
    }

    [[nodiscard]] constexpr bool contains(const Point2T<Scalar> point) const noexcept {
        return !empty_ && bounds_detail::valid(point) && point.x >= minimum_.x &&
               point.x <= maximum_.x && point.y >= minimum_.y && point.y <= maximum_.y;
    }

  private:
    constexpr Bounds2T(Point2T<Scalar> minimum, Point2T<Scalar> maximum, const bool empty) noexcept
        : minimum_{minimum}, maximum_{maximum}, empty_{empty} {}

    Point2T<Scalar> minimum_;
    Point2T<Scalar> maximum_;
    bool empty_;
};

template <GeometryScalar Scalar> class Bounds3T final {
  public:
    [[nodiscard]] static core::Result<Bounds3T>
    from_minimum_maximum(const Point3T<Scalar> minimum, const Point3T<Scalar> maximum) {
        if (!bounds_detail::valid(minimum) || !bounds_detail::valid(maximum) ||
            minimum.x > maximum.x || minimum.y > maximum.y || minimum.z > maximum.z) {
            return bounds_detail::bounds_error<Bounds3T>();
        }
        return Bounds3T{minimum, maximum, false};
    }

    [[nodiscard]] static core::Result<Bounds3T> from_points(const Point3T<Scalar> first,
                                                            const Point3T<Scalar> second) {
        if (!bounds_detail::valid(first) || !bounds_detail::valid(second)) {
            return bounds_detail::bounds_error<Bounds3T>();
        }
        return Bounds3T{
            Point3T<Scalar>{
                .x = std::min(first.x, second.x),
                .y = std::min(first.y, second.y),
                .z = std::min(first.z, second.z),
            },
            Point3T<Scalar>{
                .x = std::max(first.x, second.x),
                .y = std::max(first.y, second.y),
                .z = std::max(first.z, second.z),
            },
            false,
        };
    }

    [[nodiscard]] static constexpr Bounds3T empty() noexcept {
        return Bounds3T{Point3T<Scalar>{}, Point3T<Scalar>{}, true};
    }

    [[nodiscard]] static constexpr Bounds3T unbounded() noexcept {
        constexpr auto infinity = std::numeric_limits<Scalar>::infinity();
        return Bounds3T{
            Point3T<Scalar>{.x = -infinity, .y = -infinity, .z = -infinity},
            Point3T<Scalar>{.x = infinity, .y = infinity, .z = infinity},
            false,
        };
    }

    [[nodiscard]] constexpr bool is_empty() const noexcept {
        return empty_;
    }
    [[nodiscard]] constexpr const Point3T<Scalar>& minimum() const noexcept {
        return minimum_;
    }
    [[nodiscard]] constexpr const Point3T<Scalar>& maximum() const noexcept {
        return maximum_;
    }

    [[nodiscard]] constexpr bool contains(const Point3T<Scalar> point) const noexcept {
        return !empty_ && bounds_detail::valid(point) && point.x >= minimum_.x &&
               point.x <= maximum_.x && point.y >= minimum_.y && point.y <= maximum_.y &&
               point.z >= minimum_.z && point.z <= maximum_.z;
    }

  private:
    constexpr Bounds3T(Point3T<Scalar> minimum, Point3T<Scalar> maximum, const bool empty) noexcept
        : minimum_{minimum}, maximum_{maximum}, empty_{empty} {}

    Point3T<Scalar> minimum_;
    Point3T<Scalar> maximum_;
    bool empty_;
};

using Bounds2 = Bounds2T<TransportScalar>;
using Bounds3 = Bounds3T<TransportScalar>;
using ReferenceBounds2 = Bounds2T<ReferenceScalar>;
using ReferenceBounds3 = Bounds3T<ReferenceScalar>;

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<IntervalT<Scalar>>
intersect_aabb(const Bounds3T<Scalar>& bounds, const Point3T<Scalar> origin,
               const Vector3T<Scalar> direction, const IntervalT<Scalar>& parameter_range) {
    if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z) ||
        !std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z) ||
        (direction.x == Scalar{0} && direction.y == Scalar{0} && direction.z == Scalar{0})) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "AABB intersection requires a finite ray with a non-zero direction.",
        });
    }

    if (bounds.is_empty() || parameter_range.is_empty()) {
        return IntervalT<Scalar>::empty();
    }

    const auto origins = std::array{origin.x, origin.y, origin.z};
    const auto directions = std::array{direction.x, direction.y, direction.z};
    const auto minima = std::array{bounds.minimum().x, bounds.minimum().y, bounds.minimum().z};
    const auto maxima = std::array{bounds.maximum().x, bounds.maximum().y, bounds.maximum().z};
    auto lower = parameter_range.lower();
    auto upper = parameter_range.upper();

    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (directions[axis] == Scalar{0}) {
            if (origins[axis] < minima[axis] || origins[axis] > maxima[axis]) {
                return IntervalT<Scalar>::empty();
            }
            continue;
        }

        auto near_parameter = (minima[axis] - origins[axis]) / directions[axis];
        auto far_parameter = (maxima[axis] - origins[axis]) / directions[axis];
        if (near_parameter > far_parameter) {
            std::swap(near_parameter, far_parameter);
        }
        lower = std::max(lower, near_parameter);
        upper = std::min(upper, far_parameter);
        if (lower > upper) {
            return IntervalT<Scalar>::empty();
        }
    }

    return IntervalT<Scalar>::closed(lower, upper);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<IntervalT<Scalar>> intersect_aabb(const Bounds3T<Scalar>& bounds,
                                                             const Point3T<Scalar> origin,
                                                             const Vector3T<Scalar> direction) {
    return intersect_aabb(bounds, origin, direction, IntervalT<Scalar>::nonnegative());
}

} // namespace blackframe::renderer
