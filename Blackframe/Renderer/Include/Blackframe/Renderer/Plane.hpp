#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {

template <GeometryScalar Scalar> struct PlaneHitT final {
    Scalar parameter{};
    Point3T<Scalar> position{};
    Normal3T<Scalar> geometric_normal{};

    [[nodiscard]] constexpr bool operator==(const PlaneHitT&) const noexcept = default;
};

using PlaneHit = PlaneHitT<TransportScalar>;
using ReferencePlaneHit = PlaneHitT<ReferenceScalar>;

namespace plane_detail {

[[nodiscard]] inline core::Error plane_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar, std::size_t Capacity = 8> struct FloatingExpansion final {
    static constexpr std::size_t capacity = Capacity;

    std::array<Scalar, capacity> components{};
    std::size_t size{};
};

template <GeometryScalar Scalar, std::size_t Capacity>
[[nodiscard]] bool add_component(FloatingExpansion<Scalar, Capacity>& expansion,
                                 const Scalar component) noexcept {
    auto accumulated = component;
    auto output_index = std::size_t{0};
    const auto input_size = expansion.size;

    for (auto input_index = std::size_t{0}; input_index < input_size; ++input_index) {
        const auto existing = expansion.components[input_index];
        const auto rounded_sum = accumulated + existing;
        const auto existing_from_sum = rounded_sum - accumulated;
        const auto accumulated_from_sum = rounded_sum - existing_from_sum;
        const auto existing_remainder = existing - existing_from_sum;
        const auto accumulated_remainder = accumulated - accumulated_from_sum;
        const auto error = existing_remainder + accumulated_remainder;
        if (!std::isfinite(rounded_sum) || !std::isfinite(error)) {
            return false;
        }
        if (error != Scalar{0}) {
            if (output_index == FloatingExpansion<Scalar, Capacity>::capacity) {
                return false;
            }
            expansion.components[output_index++] = error;
        }
        accumulated = rounded_sum;
    }

    if (accumulated != Scalar{0} || output_index == 0) {
        if (output_index == FloatingExpansion<Scalar, Capacity>::capacity) {
            return false;
        }
        expansion.components[output_index++] = accumulated;
    }
    expansion.size = output_index;
    return true;
}

template <GeometryScalar Scalar, std::size_t Capacity>
[[nodiscard]] bool add_product(FloatingExpansion<Scalar, Capacity>& expansion, const Scalar left,
                               const Scalar right, const Scalar sign = Scalar{1}) noexcept {
    const auto rounded_product = left * right;
    const auto product_error = std::fma(left, right, -rounded_product);
    if (!std::isfinite(rounded_product) || !std::isfinite(product_error) ||
        (left != Scalar{0} && right != Scalar{0} && !std::isnormal(rounded_product)) ||
        (product_error != Scalar{0} && !std::isnormal(product_error))) {
        return false;
    }
    return add_component(expansion, sign * product_error) &&
           add_component(expansion, sign * rounded_product);
}

template <GeometryScalar Scalar> struct ExactValue final {
    int sign{};
    Scalar value{};
};

template <GeometryScalar Scalar, std::size_t Capacity>
[[nodiscard]] int expansion_sign(const FloatingExpansion<Scalar, Capacity>& expansion) noexcept {
    for (auto index = expansion.size; index > 0; --index) {
        const auto component = expansion.components[index - 1];
        if (component != Scalar{0}) {
            return component > Scalar{0} ? 1 : -1;
        }
    }
    return 0;
}

template <GeometryScalar Scalar, std::size_t Capacity>
[[nodiscard]] core::Result<ExactValue<Scalar>>
resolve_expansion(const FloatingExpansion<Scalar, Capacity>& expansion,
                  const char* const error_message) {
    const auto exact_sign = expansion_sign(expansion);
    if (exact_sign == 0) {
        return ExactValue<Scalar>{.sign = 0, .value = Scalar{0}};
    }

    auto value = Scalar{0};
    for (auto index = std::size_t{0}; index < expansion.size; ++index) {
        value += expansion.components[index];
    }
    if (!std::isfinite(value) || value == Scalar{0} || (value > Scalar{0} ? 1 : -1) != exact_sign) {
        return std::unexpected(plane_error(error_message));
    }
    return ExactValue<Scalar>{.sign = exact_sign, .value = value};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<FloatingExpansion<Scalar>>
exact_dot_expansion(const Vector3T<Scalar> vector, const Normal3T<Scalar> normal) {
    auto expansion = FloatingExpansion<Scalar>{};
    if (!add_product(expansion, vector.x, normal.x) ||
        !add_product(expansion, vector.y, normal.y) ||
        !add_product(expansion, vector.z, normal.z)) {
        return std::unexpected(plane_error("A planar dot product is not exactly representable."));
    }
    return expansion;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<FloatingExpansion<Scalar, 16>>
exact_plane_numerator_expansion(const Vector3T<Scalar> point, const Vector3T<Scalar> ray_origin,
                                const Normal3T<Scalar> normal) {
    auto expansion = FloatingExpansion<Scalar, 16>{};
    if (!add_product(expansion, point.x, normal.x) || !add_product(expansion, point.y, normal.y) ||
        !add_product(expansion, point.z, normal.z) ||
        !add_product(expansion, ray_origin.x, normal.x, Scalar{-1}) ||
        !add_product(expansion, ray_origin.y, normal.y, Scalar{-1}) ||
        !add_product(expansion, ray_origin.z, normal.z, Scalar{-1})) {
        return std::unexpected(plane_error("A plane numerator is not exactly representable."));
    }
    return expansion;
}

template <GeometryScalar Scalar, std::size_t DestinationCapacity, std::size_t SourceCapacity>
[[nodiscard]] bool add_scaled_expansion(FloatingExpansion<Scalar, DestinationCapacity>& destination,
                                        const FloatingExpansion<Scalar, SourceCapacity>& source,
                                        const Scalar scale,
                                        const Scalar sign = Scalar{1}) noexcept {
    for (auto index = std::size_t{0}; index < source.size; ++index) {
        if (!add_product(destination, source.components[index], scale, sign)) {
            return false;
        }
    }
    return true;
}

template <GeometryScalar Scalar, std::size_t NumeratorCapacity, std::size_t DenominatorCapacity>
[[nodiscard]] core::Result<int>
quotient_residual_sign(const FloatingExpansion<Scalar, NumeratorCapacity>& numerator,
                       const FloatingExpansion<Scalar, DenominatorCapacity>& denominator,
                       const Scalar candidate) {
    auto residual = FloatingExpansion<Scalar, 24>{};
    for (auto index = std::size_t{0}; index < numerator.size; ++index) {
        if (!add_component(residual, numerator.components[index])) {
            return std::unexpected(plane_error("Plane quotient residual is not representable."));
        }
    }
    if (!add_scaled_expansion(residual, denominator, candidate, Scalar{-1})) {
        return std::unexpected(plane_error("Plane quotient residual is not representable."));
    }
    return expansion_sign(residual);
}

template <GeometryScalar Scalar, std::size_t NumeratorCapacity, std::size_t DenominatorCapacity>
[[nodiscard]] core::Result<int>
quotient_midpoint_sign(const FloatingExpansion<Scalar, NumeratorCapacity>& numerator,
                       const FloatingExpansion<Scalar, DenominatorCapacity>& denominator,
                       const Scalar lower, const Scalar upper) {
    auto comparison = FloatingExpansion<Scalar, 48>{};
    if (!add_scaled_expansion(comparison, numerator, Scalar{2}) ||
        !add_scaled_expansion(comparison, denominator, lower, Scalar{-1}) ||
        !add_scaled_expansion(comparison, denominator, upper, Scalar{-1})) {
        return std::unexpected(
            plane_error("Plane quotient midpoint comparison is not representable."));
    }
    return expansion_sign(comparison);
}

template <GeometryScalar Scalar>
[[nodiscard]] bool has_even_significand(const Scalar value) noexcept {
    using Bits =
        std::conditional_t<sizeof(Scalar) == sizeof(std::uint32_t), std::uint32_t, std::uint64_t>;
    static_assert(sizeof(Bits) == sizeof(Scalar));
    return (std::bit_cast<Bits>(value) & Bits{1}) == Bits{0};
}

template <GeometryScalar Scalar, std::size_t NumeratorCapacity, std::size_t DenominatorCapacity>
[[nodiscard]] core::Result<Scalar> correctly_rounded_quotient(
    const FloatingExpansion<Scalar, NumeratorCapacity>& numerator_expansion,
    const FloatingExpansion<Scalar, DenominatorCapacity>& denominator_expansion) {
    const auto numerator =
        resolve_expansion(numerator_expansion, "Plane numerator cannot be rounded faithfully.");
    const auto denominator =
        resolve_expansion(denominator_expansion, "Plane denominator cannot be rounded faithfully.");
    if (!numerator.has_value()) {
        return std::unexpected(numerator.error());
    }
    if (!denominator.has_value()) {
        return std::unexpected(denominator.error());
    }
    if (denominator->sign == 0) {
        return std::unexpected(plane_error("A plane quotient requires a non-zero denominator."));
    }
    if (numerator->sign == 0) {
        return Scalar{0};
    }

    auto candidate = numerator->value / denominator->value;
    if (!std::isfinite(candidate) || candidate == Scalar{0} || !std::isnormal(candidate)) {
        return std::unexpected(plane_error("Plane intersection parameter is not representable."));
    }

    constexpr auto maximum_corrections = std::numeric_limits<Scalar>::digits + 2;
    for (auto correction = 0; correction < maximum_corrections; ++correction) {
        const auto candidate_residual =
            quotient_residual_sign(numerator_expansion, denominator_expansion, candidate);
        if (!candidate_residual.has_value()) {
            return std::unexpected(candidate_residual.error());
        }
        if (*candidate_residual == 0) {
            return candidate;
        }

        const auto direction = *candidate_residual * denominator->sign;
        const auto destination = direction > 0 ? std::numeric_limits<Scalar>::infinity()
                                               : -std::numeric_limits<Scalar>::infinity();
        const auto neighbor = std::nextafter(candidate, destination);
        if (!std::isfinite(neighbor) || neighbor == Scalar{0} || !std::isnormal(neighbor)) {
            return std::unexpected(
                plane_error("Plane intersection parameter is not representable."));
        }

        const auto neighbor_residual =
            quotient_residual_sign(numerator_expansion, denominator_expansion, neighbor);
        if (!neighbor_residual.has_value()) {
            return std::unexpected(neighbor_residual.error());
        }
        if (*neighbor_residual == 0) {
            return neighbor;
        }
        if (*neighbor_residual * denominator->sign == direction) {
            candidate = neighbor;
            continue;
        }

        const auto lower = std::min(candidate, neighbor);
        const auto upper = std::max(candidate, neighbor);
        const auto midpoint =
            quotient_midpoint_sign(numerator_expansion, denominator_expansion, lower, upper);
        if (!midpoint.has_value()) {
            return std::unexpected(midpoint.error());
        }
        const auto quotient_midpoint_direction = *midpoint * denominator->sign;
        if (quotient_midpoint_direction < 0) {
            return lower;
        }
        if (quotient_midpoint_direction > 0) {
            return upper;
        }
        return has_even_significand(lower) ? lower : upper;
    }

    return std::unexpected(
        plane_error("Plane quotient correction exceeded its representable bound."));
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<ExactValue<Scalar>>
exact_squared_radius_difference(const Vector3T<Scalar> radial, const Scalar radius) {
    auto expansion = FloatingExpansion<Scalar>{};
    if (!add_product(expansion, radial.x, radial.x) ||
        !add_product(expansion, radial.y, radial.y) ||
        !add_product(expansion, radial.z, radial.z) ||
        !add_product(expansion, radius, radius, Scalar{-1})) {
        return std::unexpected(
            plane_error("A disk radius comparison is not exactly representable."));
    }
    return resolve_expansion(expansion, "A disk radius comparison cannot be rounded faithfully.");
}

template <GeometryScalar Scalar>
[[nodiscard]] bool scaling_preserves(const Scalar original, const Scalar scaled) noexcept {
    return original == Scalar{0} || std::isnormal(scaled);
}

} // namespace plane_detail

// A plane is two-sided. Its geometric normal always follows the declared
// orientation and is never face-forwarded to the intersecting ray.
template <GeometryScalar Scalar> class PlaneT final {
  public:
    [[nodiscard]] static core::Result<PlaneT> create(const Point3T<Scalar> point,
                                                     const Normal3T<Scalar> normal) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            return std::unexpected(plane_detail::plane_error("A plane point must be finite."));
        }
        auto orientation = OrthonormalFrameT<Scalar>::from_normal(normal);
        if (!orientation.has_value()) {
            return std::unexpected(
                plane_detail::plane_error("A plane normal must be finite and non-zero."));
        }
        return PlaneT{point, *orientation};
    }

    // A successful null optional is a geometric miss or clipped intersection.
    // A coplanar ray has no unique hit and is reported as an explicit error.
    [[nodiscard]] core::Result<std::optional<PlaneHitT<Scalar>>>
    intersect(const RayT<Scalar>& ray) const {
        const auto& normal = orientation_.normal();
        const auto relevant_component = [](const Scalar component,
                                           const Scalar normal_component) noexcept {
            return normal_component == Scalar{0} ? Scalar{0} : component;
        };
        const auto plane_coordinates = Vector3T<Scalar>{
            .x = relevant_component(point_.x, normal.x),
            .y = relevant_component(point_.y, normal.y),
            .z = relevant_component(point_.z, normal.z),
        };
        const auto origin_coordinates = Vector3T<Scalar>{
            .x = relevant_component(ray.origin().x, normal.x),
            .y = relevant_component(ray.origin().y, normal.y),
            .z = relevant_component(ray.origin().z, normal.z),
        };
        const auto direction_coordinates = Vector3T<Scalar>{
            .x = relevant_component(ray.direction().x, normal.x),
            .y = relevant_component(ray.direction().y, normal.y),
            .z = relevant_component(ray.direction().z, normal.z),
        };

        const auto spatial_maximum =
            std::max({std::abs(plane_coordinates.x), std::abs(plane_coordinates.y),
                      std::abs(plane_coordinates.z), std::abs(origin_coordinates.x),
                      std::abs(origin_coordinates.y), std::abs(origin_coordinates.z)});
        const auto direction_maximum =
            std::max({std::abs(direction_coordinates.x), std::abs(direction_coordinates.y),
                      std::abs(direction_coordinates.z)});
        int spatial_exponent = 0;
        int direction_exponent = 0;
        static_cast<void>(std::frexp(spatial_maximum, &spatial_exponent));
        static_cast<void>(std::frexp(direction_maximum, &direction_exponent));
        const auto scaled_plane_coordinates = Vector3T<Scalar>{
            .x = std::ldexp(plane_coordinates.x, -spatial_exponent),
            .y = std::ldexp(plane_coordinates.y, -spatial_exponent),
            .z = std::ldexp(plane_coordinates.z, -spatial_exponent),
        };
        const auto scaled_origin_coordinates = Vector3T<Scalar>{
            .x = std::ldexp(origin_coordinates.x, -spatial_exponent),
            .y = std::ldexp(origin_coordinates.y, -spatial_exponent),
            .z = std::ldexp(origin_coordinates.z, -spatial_exponent),
        };
        const auto scaled_direction = Vector3T<Scalar>{
            .x = std::ldexp(direction_coordinates.x, -direction_exponent),
            .y = std::ldexp(direction_coordinates.y, -direction_exponent),
            .z = std::ldexp(direction_coordinates.z, -direction_exponent),
        };
        if (!plane_detail::scaling_preserves(plane_coordinates.x, scaled_plane_coordinates.x) ||
            !plane_detail::scaling_preserves(plane_coordinates.y, scaled_plane_coordinates.y) ||
            !plane_detail::scaling_preserves(plane_coordinates.z, scaled_plane_coordinates.z) ||
            !plane_detail::scaling_preserves(origin_coordinates.x, scaled_origin_coordinates.x) ||
            !plane_detail::scaling_preserves(origin_coordinates.y, scaled_origin_coordinates.y) ||
            !plane_detail::scaling_preserves(origin_coordinates.z, scaled_origin_coordinates.z) ||
            !plane_detail::scaling_preserves(direction_coordinates.x, scaled_direction.x) ||
            !plane_detail::scaling_preserves(direction_coordinates.y, scaled_direction.y) ||
            !plane_detail::scaling_preserves(direction_coordinates.z, scaled_direction.z)) {
            return std::unexpected(
                plane_detail::plane_error("Plane coefficient scaling is not representable."));
        }

        const auto numerator_expansion = plane_detail::exact_plane_numerator_expansion(
            scaled_plane_coordinates, scaled_origin_coordinates, normal);
        const auto denominator_expansion =
            plane_detail::exact_dot_expansion(scaled_direction, normal);
        if (!numerator_expansion.has_value()) {
            return std::unexpected(numerator_expansion.error());
        }
        if (!denominator_expansion.has_value()) {
            return std::unexpected(denominator_expansion.error());
        }
        const auto numerator_sign = plane_detail::expansion_sign(*numerator_expansion);
        const auto denominator_sign = plane_detail::expansion_sign(*denominator_expansion);
        if (denominator_sign == 0) {
            if (numerator_sign == 0) {
                return std::unexpected(plane_detail::plane_error(
                    "A coplanar ray does not define a unique plane intersection."));
            }
            return std::optional<PlaneHitT<Scalar>>{};
        }

        const auto scaled_parameter =
            plane_detail::correctly_rounded_quotient(*numerator_expansion, *denominator_expansion);
        if (!scaled_parameter.has_value()) {
            return std::unexpected(scaled_parameter.error());
        }
        const auto parameter = std::ldexp(*scaled_parameter, spatial_exponent - direction_exponent);
        if (!std::isfinite(parameter) ||
            (*scaled_parameter != Scalar{0} && parameter == Scalar{0})) {
            return std::unexpected(
                plane_detail::plane_error("Plane intersection parameter is not representable."));
        }
        if (!ray.contains_parameter(parameter)) {
            return std::optional<PlaneHitT<Scalar>>{};
        }

        const auto position = ray.at(parameter);
        if (!position.has_value()) {
            return std::unexpected(position.error());
        }
        return std::optional<PlaneHitT<Scalar>>{PlaneHitT<Scalar>{
            .parameter = parameter,
            .position = *position,
            .geometric_normal = orientation_.normal(),
        }};
    }

    [[nodiscard]] constexpr const Point3T<Scalar>& point() const noexcept {
        return point_;
    }

    [[nodiscard]] constexpr const OrthonormalFrameT<Scalar>& orientation() const noexcept {
        return orientation_;
    }

  private:
    constexpr PlaneT(const Point3T<Scalar> point,
                     const OrthonormalFrameT<Scalar> orientation) noexcept
        : point_{point}, orientation_{orientation} {}

    Point3T<Scalar> point_;
    OrthonormalFrameT<Scalar> orientation_;
};

using Plane = PlaneT<TransportScalar>;
using ReferencePlane = PlaneT<ReferenceScalar>;

} // namespace blackframe::renderer
