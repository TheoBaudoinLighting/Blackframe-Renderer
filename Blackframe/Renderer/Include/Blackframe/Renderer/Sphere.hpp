#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>

namespace blackframe::renderer {

template <GeometryScalar Scalar> struct SphereHitT final {
    Scalar parameter{};
    Point3T<Scalar> position{};
    Normal3T<Scalar> geometric_normal{};
    Point2T<Scalar> uv{};

    [[nodiscard]] constexpr bool operator==(const SphereHitT&) const noexcept = default;
};

using SphereHit = SphereHitT<TransportScalar>;
using ReferenceSphereHit = SphereHitT<ReferenceScalar>;

namespace sphere_detail {

[[nodiscard]] inline core::Error sphere_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar fused_dot(const Vector3T<Scalar> left, const Vector3T<Scalar> right) noexcept {
    return std::fma(left.x, right.x, std::fma(left.y, right.y, left.z * right.z));
}

template <GeometryScalar Scalar> struct FloatingExpansion final {
    static constexpr std::size_t capacity = 128;

    std::array<Scalar, capacity> components{};
    std::size_t size{};
};

template <GeometryScalar Scalar>
[[nodiscard]] bool add_component(FloatingExpansion<Scalar>& expansion,
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
            if (output_index == FloatingExpansion<Scalar>::capacity) {
                return false;
            }
            expansion.components[output_index++] = error;
        }
        accumulated = rounded_sum;
    }

    if (accumulated != Scalar{0} || output_index == 0) {
        if (output_index == FloatingExpansion<Scalar>::capacity) {
            return false;
        }
        expansion.components[output_index++] = accumulated;
    }
    expansion.size = output_index;
    return true;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool add_product(FloatingExpansion<Scalar>& expansion, const Scalar left,
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

template <GeometryScalar Scalar>
[[nodiscard]] bool add_expansion(FloatingExpansion<Scalar>& destination,
                                 const FloatingExpansion<Scalar>& source,
                                 const Scalar sign = Scalar{1}) noexcept {
    for (auto index = std::size_t{0}; index < source.size; ++index) {
        if (!add_component(destination, sign * source.components[index])) {
            return false;
        }
    }
    return true;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool multiply_expansions(const FloatingExpansion<Scalar>& left,
                                       const FloatingExpansion<Scalar>& right,
                                       FloatingExpansion<Scalar>& product) noexcept {
    product = {};
    for (auto left_index = std::size_t{0}; left_index < left.size; ++left_index) {
        for (auto right_index = std::size_t{0}; right_index < right.size; ++right_index) {
            if (!add_product(product, left.components[left_index], right.components[right_index])) {
                return false;
            }
        }
    }
    return true;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool squared_norm_expansion(const Vector3T<Scalar> vector,
                                          FloatingExpansion<Scalar>& result) noexcept {
    result = {};
    return add_product(result, vector.x, vector.x) && add_product(result, vector.y, vector.y) &&
           add_product(result, vector.z, vector.z);
}

template <GeometryScalar Scalar>
[[nodiscard]] bool cross_component_expansion(const Scalar first_left, const Scalar first_right,
                                             const Scalar second_left, const Scalar second_right,
                                             FloatingExpansion<Scalar>& result) noexcept {
    result = {};
    return add_product(result, first_left, first_right) &&
           add_product(result, second_left, second_right, Scalar{-1});
}

template <GeometryScalar Scalar> struct ExactDiscriminant final {
    int sign{};
    Scalar value{};
};

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> exact_quadratic_constant(const Vector3T<Scalar> relative_origin,
                                                            const Scalar radius) {
    auto origin_squared = FloatingExpansion<Scalar>{};
    auto radius_squared = FloatingExpansion<Scalar>{};
    if (!squared_norm_expansion(relative_origin, origin_squared) ||
        !add_product(radius_squared, radius, radius) ||
        !add_expansion(origin_squared, radius_squared, Scalar{-1})) {
        return std::unexpected(sphere_error("Sphere quadratic constant is not representable."));
    }

    auto exact_sign = 0;
    for (auto index = origin_squared.size; index > 0; --index) {
        const auto component = origin_squared.components[index - 1];
        if (component != Scalar{0}) {
            exact_sign = component > Scalar{0} ? 1 : -1;
            break;
        }
    }
    if (exact_sign == 0) {
        return Scalar{0};
    }

    auto value = Scalar{0};
    for (auto index = std::size_t{0}; index < origin_squared.size; ++index) {
        value += origin_squared.components[index];
    }
    if (!std::isfinite(value) || value == Scalar{0} || (value > Scalar{0} ? 1 : -1) != exact_sign) {
        return std::unexpected(
            sphere_error("Sphere quadratic constant cannot be rounded faithfully."));
    }
    return value;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<ExactDiscriminant<Scalar>>
exact_discriminant(const Vector3T<Scalar> relative_origin, const Vector3T<Scalar> direction,
                   const Scalar radius) {
    auto direction_squared = FloatingExpansion<Scalar>{};
    auto radius_squared = FloatingExpansion<Scalar>{};
    if (!squared_norm_expansion(direction, direction_squared) ||
        !add_product(radius_squared, radius, radius)) {
        return std::unexpected(sphere_error("Sphere discriminant products are not representable."));
    }

    auto radial_term = FloatingExpansion<Scalar>{};
    if (!multiply_expansions(direction_squared, radius_squared, radial_term)) {
        return std::unexpected(
            sphere_error("Sphere discriminant radial term is not representable."));
    }

    const auto first_terms = std::array{
        std::array{relative_origin.y, direction.z, relative_origin.z, direction.y},
        std::array{relative_origin.z, direction.x, relative_origin.x, direction.z},
        std::array{relative_origin.x, direction.y, relative_origin.y, direction.x},
    };
    auto perpendicular_squared = FloatingExpansion<Scalar>{};
    for (const auto& terms : first_terms) {
        auto cross_component = FloatingExpansion<Scalar>{};
        auto cross_component_squared = FloatingExpansion<Scalar>{};
        if (!cross_component_expansion(terms[0], terms[1], terms[2], terms[3], cross_component) ||
            !multiply_expansions(cross_component, cross_component, cross_component_squared) ||
            !add_expansion(perpendicular_squared, cross_component_squared)) {
            return std::unexpected(
                sphere_error("Sphere discriminant cross product is not representable."));
        }
    }

    if (!add_expansion(radial_term, perpendicular_squared, Scalar{-1})) {
        return std::unexpected(
            sphere_error("Sphere discriminant difference is not representable."));
    }

    auto sign = 0;
    for (auto index = radial_term.size; index > 0; --index) {
        const auto component = radial_term.components[index - 1];
        if (component != Scalar{0}) {
            sign = component > Scalar{0} ? 1 : -1;
            break;
        }
    }
    if (sign <= 0) {
        return ExactDiscriminant<Scalar>{.sign = sign, .value = Scalar{0}};
    }

    auto value = Scalar{0};
    for (auto index = std::size_t{0}; index < radial_term.size; ++index) {
        value += radial_term.components[index];
    }
    if (!std::isfinite(value) || value <= Scalar{0}) {
        return std::unexpected(
            sphere_error("A positive sphere discriminant is not representable."));
    }
    return ExactDiscriminant<Scalar>{.sign = sign, .value = value};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Normal3T<Scalar>> robust_outward_normal(const Vector3T<Scalar> outward) {
    if (!std::isfinite(outward.x) || !std::isfinite(outward.y) || !std::isfinite(outward.z)) {
        return std::unexpected(
            sphere_error("Sphere intersection produced a non-finite outward normal."));
    }
    const auto maximum_component =
        std::max({std::abs(outward.x), std::abs(outward.y), std::abs(outward.z)});
    if (maximum_component == Scalar{0}) {
        return std::unexpected(sphere_error("Sphere intersection produced a zero outward normal."));
    }

    const auto scaled = outward / maximum_component;
    const auto magnitude = std::sqrt(fused_dot(scaled, scaled));
    if (!std::isfinite(magnitude) || magnitude <= Scalar{0}) {
        return std::unexpected(sphere_error("Sphere normal normalization is not representable."));
    }
    return Normal3T<Scalar>{
        .x = scaled.x / magnitude,
        .y = scaled.y / magnitude,
        .z = scaled.z / magnitude,
    };
}

// +Y is the north pole. The seam lies on +X, u increases toward +Z, and
// v increases from the north pole to the south pole. Both poles use u = 0.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Point2T<Scalar>> sphere_uv(const Normal3T<Scalar> normal) {
    const auto pi = std::numbers::pi_v<Scalar>;
    const auto two_pi = Scalar{2} * pi;
    const auto equatorial_radius = std::hypot(normal.x, normal.z);

    auto u = Scalar{0};
    if (equatorial_radius != Scalar{0}) {
        auto azimuth = std::atan2(normal.z, normal.x);
        if (azimuth < Scalar{0}) {
            azimuth += two_pi;
        }
        u = azimuth / two_pi;
        if (u >= Scalar{1}) {
            u = Scalar{0};
        } else if (u == Scalar{0}) {
            u = Scalar{0};
        }
    }
    auto v = std::atan2(equatorial_radius, normal.y) / pi;
    if (v == Scalar{0}) {
        v = Scalar{0};
    }

    if (!std::isfinite(u) || !std::isfinite(v) || u < Scalar{0} || u >= Scalar{1} ||
        v < Scalar{0} || v > Scalar{1}) {
        return std::unexpected(sphere_error("Sphere UV computation is not representable."));
    }
    return Point2T<Scalar>{.x = u, .y = v};
}

} // namespace sphere_detail

// A sphere is expressed directly in world space. Its geometric normal always
// points outward, including when a ray starts inside the sphere.
template <GeometryScalar Scalar> class SphereT final {
  public:
    [[nodiscard]] static core::Result<SphereT> create(const Point3T<Scalar> center,
                                                      const Scalar radius) {
        if (!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(center.z)) {
            return std::unexpected(sphere_detail::sphere_error("A sphere center must be finite."));
        }
        if (!std::isfinite(radius) || radius <= Scalar{0}) {
            return std::unexpected(
                sphere_detail::sphere_error("A sphere radius must be finite and positive."));
        }
        return SphereT{center, radius};
    }

    // A successful null optional is a geometric miss. Arithmetic that cannot
    // be represented is an explicit error and is never converted into a miss.
    [[nodiscard]] core::Result<std::optional<SphereHitT<Scalar>>>
    intersect(const RayT<Scalar>& ray) const {
        const auto relative_origin = ray.origin() - center_;
        if (!std::isfinite(relative_origin.x) || !std::isfinite(relative_origin.y) ||
            !std::isfinite(relative_origin.z)) {
            return std::unexpected(sphere_detail::sphere_error(
                "Sphere intersection cannot represent the relative ray origin."));
        }

        const auto spatial_maximum =
            std::max({std::abs(relative_origin.x), std::abs(relative_origin.y),
                      std::abs(relative_origin.z), radius_});
        const auto direction_maximum =
            std::max({std::abs(ray.direction().x), std::abs(ray.direction().y),
                      std::abs(ray.direction().z)});
        int spatial_exponent = 0;
        int direction_exponent = 0;
        static_cast<void>(std::frexp(spatial_maximum, &spatial_exponent));
        static_cast<void>(std::frexp(direction_maximum, &direction_exponent));
        const auto scaled_relative_origin = Vector3T<Scalar>{
            .x = std::ldexp(relative_origin.x, -spatial_exponent),
            .y = std::ldexp(relative_origin.y, -spatial_exponent),
            .z = std::ldexp(relative_origin.z, -spatial_exponent),
        };
        const auto scaled_direction = Vector3T<Scalar>{
            .x = std::ldexp(ray.direction().x, -direction_exponent),
            .y = std::ldexp(ray.direction().y, -direction_exponent),
            .z = std::ldexp(ray.direction().z, -direction_exponent),
        };
        const auto scaled_radius = std::ldexp(radius_, -spatial_exponent);
        const auto scaling_preserved = [](const Scalar original, const Scalar scaled) noexcept {
            return original == Scalar{0} || std::isnormal(scaled);
        };
        if (!std::isfinite(direction_maximum) || direction_maximum <= Scalar{0} ||
            !scaling_preserved(relative_origin.x, scaled_relative_origin.x) ||
            !scaling_preserved(relative_origin.y, scaled_relative_origin.y) ||
            !scaling_preserved(relative_origin.z, scaled_relative_origin.z) ||
            !scaling_preserved(ray.direction().x, scaled_direction.x) ||
            !scaling_preserved(ray.direction().y, scaled_direction.y) ||
            !scaling_preserved(ray.direction().z, scaled_direction.z) ||
            !scaling_preserved(radius_, scaled_radius)) {
            return std::unexpected(
                sphere_detail::sphere_error("Sphere coefficient scaling is not representable."));
        }

        const auto a = sphere_detail::fused_dot(scaled_direction, scaled_direction);
        const auto half_b = sphere_detail::fused_dot(scaled_relative_origin, scaled_direction);
        const auto exact_c =
            sphere_detail::exact_quadratic_constant(scaled_relative_origin, scaled_radius);
        if (!exact_c.has_value()) {
            return std::unexpected(exact_c.error());
        }
        const auto c = *exact_c;
        const auto discriminant = sphere_detail::exact_discriminant(
            scaled_relative_origin, scaled_direction, scaled_radius);
        if (!discriminant.has_value()) {
            return std::unexpected(discriminant.error());
        }
        if (!std::isfinite(a) || !std::isfinite(half_b) || !std::isfinite(c) || a <= Scalar{0} ||
            scaled_radius <= Scalar{0}) {
            return std::unexpected(sphere_detail::sphere_error(
                "Sphere quadratic coefficients are not representable."));
        }
        if (discriminant->sign < 0) {
            return std::optional<SphereHitT<Scalar>>{};
        }

        Scalar first_scaled_root{};
        Scalar second_scaled_root{};
        if (discriminant->sign == 0) {
            first_scaled_root = -half_b / a;
            second_scaled_root = first_scaled_root;
        } else {
            const auto square_root = std::sqrt(discriminant->value);
            const auto q = -half_b - std::copysign(square_root, half_b);
            if (!std::isfinite(square_root) || !std::isfinite(q) || q == Scalar{0}) {
                return std::unexpected(
                    sphere_detail::sphere_error("Sphere roots are not representable."));
            }
            first_scaled_root = q / a;
            second_scaled_root = c / q;
            if (first_scaled_root > second_scaled_root) {
                std::swap(first_scaled_root, second_scaled_root);
            }
        }
        const auto parameter_exponent = spatial_exponent - direction_exponent;
        const auto first_root = std::ldexp(first_scaled_root, parameter_exponent);
        const auto second_root = std::ldexp(second_scaled_root, parameter_exponent);
        if (!std::isfinite(first_root) || !std::isfinite(second_root) ||
            (first_scaled_root != Scalar{0} && first_root == Scalar{0}) ||
            (second_scaled_root != Scalar{0} && second_root == Scalar{0})) {
            return std::unexpected(
                sphere_detail::sphere_error("Sphere roots are not representable."));
        }

        auto parameter = std::optional<Scalar>{};
        if (ray.contains_parameter(first_root)) {
            parameter = first_root;
        } else if (ray.contains_parameter(second_root)) {
            parameter = second_root;
        } else {
            return std::optional<SphereHitT<Scalar>>{};
        }

        const auto position = ray.at(*parameter);
        if (!position.has_value()) {
            return std::unexpected(position.error());
        }
        const auto geometric_normal = sphere_detail::robust_outward_normal(*position - center_);
        if (!geometric_normal.has_value()) {
            return std::unexpected(geometric_normal.error());
        }
        const auto uv = sphere_detail::sphere_uv(*geometric_normal);
        if (!uv.has_value()) {
            return std::unexpected(uv.error());
        }

        return std::optional<SphereHitT<Scalar>>{SphereHitT<Scalar>{
            .parameter = *parameter,
            .position = *position,
            .geometric_normal = *geometric_normal,
            .uv = *uv,
        }};
    }

    [[nodiscard]] constexpr const Point3T<Scalar>& center() const noexcept {
        return center_;
    }

    [[nodiscard]] constexpr Scalar radius() const noexcept {
        return radius_;
    }

  private:
    constexpr SphereT(const Point3T<Scalar> center, const Scalar radius) noexcept
        : center_{center}, radius_{radius} {}

    Point3T<Scalar> center_;
    Scalar radius_;
};

using Sphere = SphereT<TransportScalar>;
using ReferenceSphere = SphereT<ReferenceScalar>;

} // namespace blackframe::renderer
