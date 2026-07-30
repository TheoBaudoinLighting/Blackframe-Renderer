#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {

template <GeometryScalar Scalar> struct TriangleBarycentricsT final {
    Scalar vertex0{};
    Scalar vertex1{};
    Scalar vertex2{};

    [[nodiscard]] constexpr bool operator==(const TriangleBarycentricsT&) const noexcept = default;
};

using TriangleBarycentrics = TriangleBarycentricsT<TransportScalar>;
using ReferenceTriangleBarycentrics = TriangleBarycentricsT<ReferenceScalar>;

template <GeometryScalar Scalar> struct TriangleHitT final {
    Scalar parameter{};
    Point3T<Scalar> position{};
    Normal3T<Scalar> geometric_normal{};
    TriangleBarycentricsT<Scalar> barycentrics{};

    [[nodiscard]] constexpr bool operator==(const TriangleHitT&) const noexcept = default;
};

using TriangleHit = TriangleHitT<TransportScalar>;
using ReferenceTriangleHit = TriangleHitT<ReferenceScalar>;

enum class TriangleIntersectionErrorKind : std::uint8_t {
    numerical_failure,
    coplanar_ambiguity,
};

struct TriangleIntersectionError final {
    TriangleIntersectionErrorKind kind{TriangleIntersectionErrorKind::numerical_failure};
    core::Error diagnostic;
};

template <GeometryScalar Scalar>
using TriangleIntersectionResultT =
    std::expected<std::optional<TriangleHitT<Scalar>>, TriangleIntersectionError>;

namespace triangle_detail {

inline constexpr char CoplanarIntersectionMessage[] =
    "A coplanar ray does not define a unique triangle intersection.";

[[nodiscard]] inline core::Error triangle_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

[[nodiscard]] inline TriangleIntersectionError triangle_intersection_error(
    core::Error diagnostic,
    const TriangleIntersectionErrorKind kind = TriangleIntersectionErrorKind::numerical_failure) {
    return TriangleIntersectionError{
        .kind = kind,
        .diagnostic = std::move(diagnostic),
    };
}

template <GeometryScalar Scalar, std::size_t Capacity> struct FloatingExpansion final {
    std::array<Scalar, Capacity> components{};
    std::size_t size{};
};

template <GeometryScalar Scalar, std::size_t Capacity>
[[nodiscard]] bool add_component(FloatingExpansion<Scalar, Capacity>& expansion,
                                 const Scalar component) noexcept {
    if (!std::isfinite(component)) {
        return false;
    }

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
            if (output_index == Capacity) {
                return false;
            }
            expansion.components[output_index++] = error;
        }
        accumulated = rounded_sum;
    }

    if (accumulated != Scalar{0} || output_index == 0) {
        if (output_index == Capacity) {
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

template <GeometryScalar Scalar, std::size_t DestinationCapacity, std::size_t SourceCapacity>
[[nodiscard]] bool add_expansion(FloatingExpansion<Scalar, DestinationCapacity>& destination,
                                 const FloatingExpansion<Scalar, SourceCapacity>& source,
                                 const Scalar sign = Scalar{1}) noexcept {
    for (auto index = std::size_t{0}; index < source.size; ++index) {
        if (!add_component(destination, sign * source.components[index])) {
            return false;
        }
    }
    return true;
}

template <GeometryScalar Scalar, std::size_t OutputCapacity, std::size_t LeftCapacity,
          std::size_t RightCapacity>
[[nodiscard]] bool
multiply_expansions(const FloatingExpansion<Scalar, LeftCapacity>& left,
                    const FloatingExpansion<Scalar, RightCapacity>& right,
                    FloatingExpansion<Scalar, OutputCapacity>& product) noexcept {
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

template <GeometryScalar Scalar> struct ExactValue final {
    int sign{};
    Scalar value{};
};

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
        return std::unexpected(triangle_error(error_message));
    }
    return ExactValue<Scalar>{.sign = exact_sign, .value = value};
}

template <GeometryScalar Scalar> using CoordinateExpansion = FloatingExpansion<Scalar, 2>;

template <GeometryScalar Scalar> struct ExpansionVector3 final {
    CoordinateExpansion<Scalar> x{};
    CoordinateExpansion<Scalar> y{};
    CoordinateExpansion<Scalar> z{};
};

template <GeometryScalar Scalar>
[[nodiscard]] bool exact_difference(const Scalar left, const Scalar right,
                                    CoordinateExpansion<Scalar>& difference) noexcept {
    difference = {};
    return add_component(difference, left) && add_component(difference, -right);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<ExpansionVector3<Scalar>>
exact_relative_vector(const Point3T<Scalar> point, const Point3T<Scalar> origin) {
    auto result = ExpansionVector3<Scalar>{};
    if (!exact_difference(point.x, origin.x, result.x) ||
        !exact_difference(point.y, origin.y, result.y) ||
        !exact_difference(point.z, origin.z, result.z)) {
        return std::unexpected(
            triangle_error("A triangle relative coordinate is not exactly representable."));
    }
    return result;
}

template <GeometryScalar Scalar>
[[nodiscard]] ExpansionVector3<Scalar> exact_vector(const Vector3T<Scalar> vector) noexcept {
    auto result = ExpansionVector3<Scalar>{};
    result.x.components[0] = vector.x;
    result.x.size = 1;
    result.y.components[0] = vector.y;
    result.y.size = 1;
    result.z.components[0] = vector.z;
    result.z.size = 1;
    return result;
}

template <GeometryScalar Scalar> using PairProductExpansion = FloatingExpansion<Scalar, 16>;

template <GeometryScalar Scalar> using TripleProductExpansion = FloatingExpansion<Scalar, 64>;

template <GeometryScalar Scalar> using DeterminantExpansion = FloatingExpansion<Scalar, 512>;

template <GeometryScalar Scalar>
[[nodiscard]] bool add_triple_product(DeterminantExpansion<Scalar>& determinant,
                                      const CoordinateExpansion<Scalar>& first,
                                      const CoordinateExpansion<Scalar>& second,
                                      const CoordinateExpansion<Scalar>& third,
                                      const Scalar sign) noexcept {
    auto pair_product = PairProductExpansion<Scalar>{};
    auto triple_product = TripleProductExpansion<Scalar>{};
    return multiply_expansions(first, second, pair_product) &&
           multiply_expansions(pair_product, third, triple_product) &&
           add_expansion(determinant, triple_product, sign);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<DeterminantExpansion<Scalar>>
exact_determinant(const ExpansionVector3<Scalar>& first, const ExpansionVector3<Scalar>& second,
                  const ExpansionVector3<Scalar>& third) {
    auto determinant = DeterminantExpansion<Scalar>{};
    if (!add_triple_product(determinant, first.x, second.y, third.z, Scalar{1}) ||
        !add_triple_product(determinant, first.y, second.z, third.x, Scalar{1}) ||
        !add_triple_product(determinant, first.z, second.x, third.y, Scalar{1}) ||
        !add_triple_product(determinant, first.z, second.y, third.x, Scalar{-1}) ||
        !add_triple_product(determinant, first.y, second.x, third.z, Scalar{-1}) ||
        !add_triple_product(determinant, first.x, second.z, third.y, Scalar{-1})) {
        return std::unexpected(
            triangle_error("A triangle determinant is not exactly representable."));
    }
    return determinant;
}

template <GeometryScalar Scalar> using CrossComponentExpansion = FloatingExpansion<Scalar, 32>;

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<CrossComponentExpansion<Scalar>>
exact_difference_of_products(const CoordinateExpansion<Scalar>& first_left,
                             const CoordinateExpansion<Scalar>& first_right,
                             const CoordinateExpansion<Scalar>& second_left,
                             const CoordinateExpansion<Scalar>& second_right) {
    auto first_product = PairProductExpansion<Scalar>{};
    auto second_product = PairProductExpansion<Scalar>{};
    auto difference = CrossComponentExpansion<Scalar>{};
    if (!multiply_expansions(first_left, first_right, first_product) ||
        !multiply_expansions(second_left, second_right, second_product) ||
        !add_expansion(difference, first_product) ||
        !add_expansion(difference, second_product, Scalar{-1})) {
        return std::unexpected(
            triangle_error("A triangle cross product is not exactly representable."));
    }
    return difference;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool scaling_preserves(const Scalar original, const Scalar scaled) noexcept {
    return original == Scalar{0} || std::isnormal(scaled);
}

struct AxisExponents final {
    int x{};
    int y{};
    int z{};
};

template <GeometryScalar Scalar, std::size_t Size>
[[nodiscard]] int scaling_exponent(const std::array<Scalar, Size>& values) noexcept {
    auto maximum = Scalar{0};
    for (const auto value : values) {
        maximum = std::max(maximum, std::abs(value));
    }
    auto exponent = 0;
    static_cast<void>(std::frexp(maximum, &exponent));
    return exponent;
}

template <GeometryScalar Scalar>
[[nodiscard]] Point3T<Scalar> scaled_point(const Point3T<Scalar> point,
                                           const AxisExponents exponents) noexcept {
    return {
        .x = std::ldexp(point.x, -exponents.x),
        .y = std::ldexp(point.y, -exponents.y),
        .z = std::ldexp(point.z, -exponents.z),
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] Vector3T<Scalar> scaled_vector(const Vector3T<Scalar> vector,
                                             const AxisExponents spatial_exponents,
                                             const int direction_exponent) noexcept {
    return {
        .x = std::ldexp(vector.x, -spatial_exponents.x - direction_exponent),
        .y = std::ldexp(vector.y, -spatial_exponents.y - direction_exponent),
        .z = std::ldexp(vector.z, -spatial_exponents.z - direction_exponent),
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] int transformed_direction_exponent(const Vector3T<Scalar> direction,
                                                 const AxisExponents spatial_exponents) noexcept {
    auto result = std::numeric_limits<int>::min();
    const auto include_component = [&result](const Scalar component,
                                             const int spatial_exponent) noexcept {
        if (component == Scalar{0}) {
            return;
        }
        auto component_exponent = 0;
        static_cast<void>(std::frexp(std::abs(component), &component_exponent));
        result = std::max(result, component_exponent - spatial_exponent);
    };
    include_component(direction.x, spatial_exponents.x);
    include_component(direction.y, spatial_exponents.y);
    include_component(direction.z, spatial_exponents.z);
    return result;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool scaling_preserves(const Point3T<Scalar> original,
                                     const Point3T<Scalar> scaled) noexcept {
    return scaling_preserves(original.x, scaled.x) && scaling_preserves(original.y, scaled.y) &&
           scaling_preserves(original.z, scaled.z);
}

template <GeometryScalar Scalar>
[[nodiscard]] bool scaling_preserves(const Vector3T<Scalar> original,
                                     const Vector3T<Scalar> scaled) noexcept {
    return scaling_preserves(original.x, scaled.x) && scaling_preserves(original.y, scaled.y) &&
           scaling_preserves(original.z, scaled.z);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Normal3T<Scalar>> triangle_normal(const Point3T<Scalar> vertex0,
                                                             const Point3T<Scalar> vertex1,
                                                             const Point3T<Scalar> vertex2) {
    const auto spatial_exponents = AxisExponents{
        .x = scaling_exponent(std::array{vertex0.x, vertex1.x, vertex2.x}),
        .y = scaling_exponent(std::array{vertex0.y, vertex1.y, vertex2.y}),
        .z = scaling_exponent(std::array{vertex0.z, vertex1.z, vertex2.z}),
    };
    const auto scaled_vertex0 = scaled_point(vertex0, spatial_exponents);
    const auto scaled_vertex1 = scaled_point(vertex1, spatial_exponents);
    const auto scaled_vertex2 = scaled_point(vertex2, spatial_exponents);
    if (!scaling_preserves(vertex0, scaled_vertex0) ||
        !scaling_preserves(vertex1, scaled_vertex1) ||
        !scaling_preserves(vertex2, scaled_vertex2)) {
        return std::unexpected(triangle_error("Triangle vertex scaling is not representable."));
    }

    const auto edge1 = exact_relative_vector(scaled_vertex1, scaled_vertex0);
    const auto edge2 = exact_relative_vector(scaled_vertex2, scaled_vertex0);
    if (!edge1.has_value()) {
        return std::unexpected(edge1.error());
    }
    if (!edge2.has_value()) {
        return std::unexpected(edge2.error());
    }

    const auto cross_x = exact_difference_of_products(edge1->y, edge2->z, edge1->z, edge2->y);
    const auto cross_y = exact_difference_of_products(edge1->z, edge2->x, edge1->x, edge2->z);
    const auto cross_z = exact_difference_of_products(edge1->x, edge2->y, edge1->y, edge2->x);
    if (!cross_x.has_value()) {
        return std::unexpected(cross_x.error());
    }
    if (!cross_y.has_value()) {
        return std::unexpected(cross_y.error());
    }
    if (!cross_z.has_value()) {
        return std::unexpected(cross_z.error());
    }

    const auto x = resolve_expansion(*cross_x, "Triangle normal x cannot be rounded faithfully.");
    const auto y = resolve_expansion(*cross_y, "Triangle normal y cannot be rounded faithfully.");
    const auto z = resolve_expansion(*cross_z, "Triangle normal z cannot be rounded faithfully.");
    if (!x.has_value()) {
        return std::unexpected(x.error());
    }
    if (!y.has_value()) {
        return std::unexpected(y.error());
    }
    if (!z.has_value()) {
        return std::unexpected(z.error());
    }
    if (x->sign == 0 && y->sign == 0 && z->sign == 0) {
        return std::unexpected(triangle_error("A triangle requires three non-collinear vertices."));
    }

    const auto cross_exponents = AxisExponents{
        .x = spatial_exponents.y + spatial_exponents.z,
        .y = spatial_exponents.z + spatial_exponents.x,
        .z = spatial_exponents.x + spatial_exponents.y,
    };
    auto normal_exponent = std::numeric_limits<int>::min();
    const auto include_cross_component = [&normal_exponent](const ExactValue<Scalar> component,
                                                            const int cross_exponent) noexcept {
        if (component.sign == 0) {
            return;
        }
        auto component_exponent = 0;
        static_cast<void>(std::frexp(std::abs(component.value), &component_exponent));
        normal_exponent = std::max(normal_exponent, component_exponent + cross_exponent);
    };
    include_cross_component(*x, cross_exponents.x);
    include_cross_component(*y, cross_exponents.y);
    include_cross_component(*z, cross_exponents.z);

    const auto adjusted_x =
        x->sign == 0 ? Scalar{0} : std::ldexp(x->value, cross_exponents.x - normal_exponent);
    const auto adjusted_y =
        y->sign == 0 ? Scalar{0} : std::ldexp(y->value, cross_exponents.y - normal_exponent);
    const auto adjusted_z =
        z->sign == 0 ? Scalar{0} : std::ldexp(z->value, cross_exponents.z - normal_exponent);
    if (!scaling_preserves(x->value, adjusted_x) || !scaling_preserves(y->value, adjusted_y) ||
        !scaling_preserves(z->value, adjusted_z)) {
        return std::unexpected(triangle_error("Triangle normal scaling is not representable."));
    }

    const auto maximum_component =
        std::max({std::abs(adjusted_x), std::abs(adjusted_y), std::abs(adjusted_z)});
    if (!std::isfinite(maximum_component) || maximum_component <= Scalar{0}) {
        return std::unexpected(
            triangle_error("Triangle normal normalization is not representable."));
    }
    const auto scaled_x = adjusted_x / maximum_component;
    const auto scaled_y = adjusted_y / maximum_component;
    const auto scaled_z = adjusted_z / maximum_component;
    const auto squared_length =
        std::fma(scaled_x, scaled_x, std::fma(scaled_y, scaled_y, scaled_z * scaled_z));
    const auto length = std::sqrt(squared_length);
    if (!std::isfinite(length) || length <= Scalar{0}) {
        return std::unexpected(
            triangle_error("Triangle normal normalization is not representable."));
    }
    return Normal3T<Scalar>{
        .x = scaled_x / length,
        .y = scaled_y / length,
        .z = scaled_z / length,
    };
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
    auto residual = FloatingExpansion<Scalar, NumeratorCapacity + 2 * DenominatorCapacity + 8>{};
    if (!add_expansion(residual, numerator) ||
        !add_scaled_expansion(residual, denominator, candidate, Scalar{-1})) {
        return std::unexpected(triangle_error("Triangle quotient residual is not representable."));
    }
    return expansion_sign(residual);
}

template <GeometryScalar Scalar, std::size_t NumeratorCapacity, std::size_t DenominatorCapacity>
[[nodiscard]] core::Result<int>
quotient_midpoint_sign(const FloatingExpansion<Scalar, NumeratorCapacity>& numerator,
                       const FloatingExpansion<Scalar, DenominatorCapacity>& denominator,
                       const Scalar lower, const Scalar upper) {
    auto comparison =
        FloatingExpansion<Scalar, 2 * NumeratorCapacity + 4 * DenominatorCapacity + 8>{};
    if (!add_scaled_expansion(comparison, numerator, Scalar{2}) ||
        !add_scaled_expansion(comparison, denominator, lower, Scalar{-1}) ||
        !add_scaled_expansion(comparison, denominator, upper, Scalar{-1})) {
        return std::unexpected(
            triangle_error("Triangle quotient midpoint comparison is not representable."));
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
    const FloatingExpansion<Scalar, DenominatorCapacity>& denominator_expansion,
    const char* const unrepresentable_message) {
    const auto numerator = resolve_expansion(
        numerator_expansion, "Triangle quotient numerator cannot be rounded faithfully.");
    const auto denominator = resolve_expansion(
        denominator_expansion, "Triangle quotient denominator cannot be rounded faithfully.");
    if (!numerator.has_value()) {
        return std::unexpected(numerator.error());
    }
    if (!denominator.has_value()) {
        return std::unexpected(denominator.error());
    }
    if (denominator->sign == 0) {
        return std::unexpected(
            triangle_error("A triangle quotient requires a non-zero denominator."));
    }
    if (numerator->sign == 0) {
        return Scalar{0};
    }

    auto candidate = numerator->value / denominator->value;
    if (!std::isfinite(candidate) || candidate == Scalar{0} || !std::isnormal(candidate)) {
        return std::unexpected(triangle_error(unrepresentable_message));
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
            return std::unexpected(triangle_error(unrepresentable_message));
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
        const auto midpoint_direction = *midpoint * denominator->sign;
        if (midpoint_direction < 0) {
            return lower;
        }
        if (midpoint_direction > 0) {
            return upper;
        }
        return has_even_significand(lower) ? lower : upper;
    }

    return std::unexpected(
        triangle_error("Triangle quotient correction exceeded its representable bound."));
}

} // namespace triangle_detail

// A triangle is two-sided. Its winding fixes the geometric normal, which is
// never face-forwarded. All three edges are closed.
template <GeometryScalar Scalar> class TriangleT final {
  public:
    [[nodiscard]] static core::Result<TriangleT> create(const Point3T<Scalar> vertex0,
                                                        const Point3T<Scalar> vertex1,
                                                        const Point3T<Scalar> vertex2) {
        const auto finite_point = [](const Point3T<Scalar> point) noexcept {
            return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
        };
        if (!finite_point(vertex0) || !finite_point(vertex1) || !finite_point(vertex2)) {
            return std::unexpected(
                triangle_detail::triangle_error("Triangle vertices must be finite."));
        }

        const auto geometric_normal = triangle_detail::triangle_normal(vertex0, vertex1, vertex2);
        if (!geometric_normal.has_value()) {
            return std::unexpected(geometric_normal.error());
        }
        return TriangleT{{vertex0, vertex1, vertex2}, *geometric_normal};
    }

    [[nodiscard]] core::Result<std::optional<TriangleHitT<Scalar>>>
    intersect(const RayT<Scalar>& ray) const {
        auto intersection = intersect_classified(ray);
        if (!intersection) {
            return std::unexpected(std::move(intersection.error().diagnostic));
        }
        return std::move(*intersection);
    }

    // A successful null optional is a geometric miss or clipped intersection.
    // A coplanar ray has no unique hit and has a typed classification so
    // acceleration structures can define their crossing policy explicitly.
    [[nodiscard]] TriangleIntersectionResultT<Scalar>
    intersect_classified(const RayT<Scalar>& ray) const {
        const auto spatial_exponents = triangle_detail::AxisExponents{
            .x = triangle_detail::scaling_exponent(
                std::array{vertices_[0].x, vertices_[1].x, vertices_[2].x, ray.origin().x}),
            .y = triangle_detail::scaling_exponent(
                std::array{vertices_[0].y, vertices_[1].y, vertices_[2].y, ray.origin().y}),
            .z = triangle_detail::scaling_exponent(
                std::array{vertices_[0].z, vertices_[1].z, vertices_[2].z, ray.origin().z}),
        };
        const auto direction_exponent =
            triangle_detail::transformed_direction_exponent(ray.direction(), spatial_exponents);
        const auto scaled_vertex0 = triangle_detail::scaled_point(vertices_[0], spatial_exponents);
        const auto scaled_vertex1 = triangle_detail::scaled_point(vertices_[1], spatial_exponents);
        const auto scaled_vertex2 = triangle_detail::scaled_point(vertices_[2], spatial_exponents);
        const auto scaled_origin = triangle_detail::scaled_point(ray.origin(), spatial_exponents);
        const auto scaled_direction =
            triangle_detail::scaled_vector(ray.direction(), spatial_exponents, direction_exponent);
        if (!triangle_detail::scaling_preserves(vertices_[0], scaled_vertex0) ||
            !triangle_detail::scaling_preserves(vertices_[1], scaled_vertex1) ||
            !triangle_detail::scaling_preserves(vertices_[2], scaled_vertex2) ||
            !triangle_detail::scaling_preserves(ray.origin(), scaled_origin) ||
            !triangle_detail::scaling_preserves(ray.direction(), scaled_direction)) {
            return std::unexpected(
                triangle_detail::triangle_intersection_error(triangle_detail::triangle_error(
                    "Triangle intersection scaling is not representable.")));
        }

        const auto point0 = triangle_detail::exact_relative_vector(scaled_vertex0, scaled_origin);
        const auto point1 = triangle_detail::exact_relative_vector(scaled_vertex1, scaled_origin);
        const auto point2 = triangle_detail::exact_relative_vector(scaled_vertex2, scaled_origin);
        if (!point0.has_value()) {
            return std::unexpected(triangle_detail::triangle_intersection_error(point0.error()));
        }
        if (!point1.has_value()) {
            return std::unexpected(triangle_detail::triangle_intersection_error(point1.error()));
        }
        if (!point2.has_value()) {
            return std::unexpected(triangle_detail::triangle_intersection_error(point2.error()));
        }
        const auto direction = triangle_detail::exact_vector(scaled_direction);

        const auto edge0 = triangle_detail::exact_determinant(*point1, *point2, direction);
        const auto edge1 = triangle_detail::exact_determinant(*point2, *point0, direction);
        const auto edge2 = triangle_detail::exact_determinant(*point0, *point1, direction);
        const auto volume = triangle_detail::exact_determinant(*point0, *point1, *point2);
        if (!edge0.has_value()) {
            return std::unexpected(triangle_detail::triangle_intersection_error(edge0.error()));
        }
        if (!edge1.has_value()) {
            return std::unexpected(triangle_detail::triangle_intersection_error(edge1.error()));
        }
        if (!edge2.has_value()) {
            return std::unexpected(triangle_detail::triangle_intersection_error(edge2.error()));
        }
        if (!volume.has_value()) {
            return std::unexpected(triangle_detail::triangle_intersection_error(volume.error()));
        }

        auto determinant = triangle_detail::DeterminantExpansion<Scalar>{};
        if (!triangle_detail::add_expansion(determinant, *edge0) ||
            !triangle_detail::add_expansion(determinant, *edge1) ||
            !triangle_detail::add_expansion(determinant, *edge2)) {
            return std::unexpected(
                triangle_detail::triangle_intersection_error(triangle_detail::triangle_error(
                    "Triangle edge determinant sum is not representable.")));
        }

        const auto determinant_sign = triangle_detail::expansion_sign(determinant);
        const auto volume_sign = triangle_detail::expansion_sign(*volume);
        if (determinant_sign == 0) {
            if (volume_sign == 0) {
                return std::unexpected(triangle_detail::triangle_intersection_error(
                    triangle_detail::triangle_error(triangle_detail::CoplanarIntersectionMessage),
                    TriangleIntersectionErrorKind::coplanar_ambiguity));
            }
            return std::optional<TriangleHitT<Scalar>>{};
        }

        const auto edge0_sign = triangle_detail::expansion_sign(*edge0);
        const auto edge1_sign = triangle_detail::expansion_sign(*edge1);
        const auto edge2_sign = triangle_detail::expansion_sign(*edge2);
        const auto all_non_negative = edge0_sign >= 0 && edge1_sign >= 0 && edge2_sign >= 0;
        const auto all_non_positive = edge0_sign <= 0 && edge1_sign <= 0 && edge2_sign <= 0;
        if (!all_non_negative && !all_non_positive) {
            return std::optional<TriangleHitT<Scalar>>{};
        }

        const auto scaled_parameter = triangle_detail::correctly_rounded_quotient(
            *volume, determinant, "Triangle intersection parameter is not representable.");
        if (!scaled_parameter.has_value()) {
            return std::unexpected(
                triangle_detail::triangle_intersection_error(scaled_parameter.error()));
        }
        const auto parameter = std::ldexp(*scaled_parameter, -direction_exponent);
        if (!std::isfinite(parameter) ||
            (*scaled_parameter != Scalar{0} && parameter == Scalar{0})) {
            return std::unexpected(
                triangle_detail::triangle_intersection_error(triangle_detail::triangle_error(
                    "Triangle intersection parameter is not representable.")));
        }
        if (!ray.contains_parameter(parameter)) {
            return std::optional<TriangleHitT<Scalar>>{};
        }

        const auto barycentric0 = triangle_detail::correctly_rounded_quotient(
            *edge0, determinant, "Triangle barycentric coordinate is not representable.");
        const auto barycentric1 = triangle_detail::correctly_rounded_quotient(
            *edge1, determinant, "Triangle barycentric coordinate is not representable.");
        const auto barycentric2 = triangle_detail::correctly_rounded_quotient(
            *edge2, determinant, "Triangle barycentric coordinate is not representable.");
        if (!barycentric0.has_value()) {
            return std::unexpected(
                triangle_detail::triangle_intersection_error(barycentric0.error()));
        }
        if (!barycentric1.has_value()) {
            return std::unexpected(
                triangle_detail::triangle_intersection_error(barycentric1.error()));
        }
        if (!barycentric2.has_value()) {
            return std::unexpected(
                triangle_detail::triangle_intersection_error(barycentric2.error()));
        }

        const auto position = ray.at(parameter);
        if (!position.has_value()) {
            return std::unexpected(triangle_detail::triangle_intersection_error(position.error()));
        }
        return std::optional<TriangleHitT<Scalar>>{TriangleHitT<Scalar>{
            .parameter = parameter,
            .position = *position,
            .geometric_normal = geometric_normal_,
            .barycentrics =
                {
                    .vertex0 = *barycentric0,
                    .vertex1 = *barycentric1,
                    .vertex2 = *barycentric2,
                },
        }};
    }

    [[nodiscard]] constexpr const std::array<Point3T<Scalar>, 3>& vertices() const noexcept {
        return vertices_;
    }

    [[nodiscard]] constexpr const Normal3T<Scalar>& geometric_normal() const noexcept {
        return geometric_normal_;
    }

  private:
    constexpr TriangleT(const std::array<Point3T<Scalar>, 3> vertices,
                        const Normal3T<Scalar> geometric_normal) noexcept
        : vertices_{vertices}, geometric_normal_{geometric_normal} {}

    std::array<Point3T<Scalar>, 3> vertices_;
    Normal3T<Scalar> geometric_normal_;
};

using Triangle = TriangleT<TransportScalar>;
using ReferenceTriangle = TriangleT<ReferenceScalar>;

} // namespace blackframe::renderer
