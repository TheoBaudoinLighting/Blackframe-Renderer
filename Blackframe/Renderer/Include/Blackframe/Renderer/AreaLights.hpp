#pragma once

#include <Blackframe/Renderer/Detail/PacketLightSpectrum.hpp>
#include <Blackframe/Renderer/Disk.hpp>
#include <Blackframe/Renderer/Emission.hpp>
#include <Blackframe/Renderer/Light.hpp>
#include <Blackframe/Renderer/Plane.hpp>
#include <Blackframe/Renderer/SamplingMappings.hpp>
#include <Blackframe/Renderer/Sphere.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::renderer {

enum class AreaLightSidedness : std::uint8_t {
    one_sided,
    two_sided,
};

struct AreaLightTriangleIndices final {
    std::uint32_t vertex0{};
    std::uint32_t vertex1{};
    std::uint32_t vertex2{};

    [[nodiscard]] constexpr bool
    operator==(const AreaLightTriangleIndices&) const noexcept = default;
};

namespace area_light_detail {

[[nodiscard]] inline core::Error invalid_area_light(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

[[nodiscard]] inline core::Error exhausted_area_light(const char* const message) {
    return core::Error{
        .code = core::StatusCode::resource_exhausted,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_canonical_sample(const Point2T<Scalar> sample) {
    if (!std::isfinite(sample.x) || !std::isfinite(sample.y) || sample.x < Scalar{0} ||
        sample.x >= Scalar{1} || sample.y < Scalar{0} || sample.y >= Scalar{1}) {
        return std::unexpected(invalid_area_light(
            "Area-light sampling requires a finite canonical sample in [0, 1)^2."));
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_finite_scene_bounds(const Bounds3T<Scalar>& bounds) {
    if (bounds.is_empty() || !light_detail::finite(bounds.minimum()) ||
        !light_detail::finite(bounds.maximum())) {
        return std::unexpected(invalid_area_light(
            "Area-light power requires explicit non-empty finite scene bounds."));
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_position_error(const Vector3T<Scalar> absolute_position_error) {
    if (!light_detail::finite_non_negative(absolute_position_error)) {
        return std::unexpected(
            invalid_area_light("Area lights require finite non-negative absolute position error."));
    }
    return {};
}

[[nodiscard]] inline core::Status validate_sidedness(const AreaLightSidedness sidedness) {
    if (sidedness != AreaLightSidedness::one_sided && sidedness != AreaLightSidedness::two_sided) {
        return std::unexpected(
            invalid_area_light("Area-light sidedness is not a recognized value."));
    }
    return {};
}

template <SpectrumScalar Scalar, std::size_t Count>
[[nodiscard]] core::Result<Scalar> positive_product(const std::array<Scalar, Count>& factors,
                                                    const char* const error_message) {
    auto fraction = Scalar{1};
    auto exponent = 0LL;
    for (const auto factor : factors) {
        if (!std::isfinite(factor) || !(factor > Scalar{0})) {
            return std::unexpected(invalid_area_light(error_message));
        }
        auto factor_exponent = 0;
        fraction *= std::frexp(factor, &factor_exponent);
        exponent += factor_exponent;

        auto normalization_exponent = 0;
        fraction = std::frexp(fraction, &normalization_exponent);
        exponent += normalization_exponent;
    }

    if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max()) {
        return std::unexpected(invalid_area_light(error_message));
    }
    const auto product = std::scalbn(fraction, static_cast<int>(exponent));
    if (!std::isfinite(product) || !(product > Scalar{0})) {
        return std::unexpected(invalid_area_light(error_message));
    }
    return product;
}

template <SpectrumScalar Scalar> struct AreaMeasureT final {
    Scalar area;
    Scalar inverse_area;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<AreaMeasureT<Scalar>> area_measure(const Scalar area) {
    if (!std::isfinite(area) || !(area > Scalar{0})) {
        return std::unexpected(
            invalid_area_light("Area-light surface area must be finite and positive."));
    }
    const auto inverse_area = Scalar{1} / area;
    if (!std::isfinite(inverse_area) || !(inverse_area > Scalar{0})) {
        return std::unexpected(
            invalid_area_light("Area-light reciprocal surface area is not representable."));
    }
    return AreaMeasureT<Scalar>{.area = area, .inverse_area = inverse_area};
}

template <SpectrumScalar Scalar, std::size_t Count>
[[nodiscard]] core::Result<AreaMeasureT<Scalar>>
area_measure(const std::array<Scalar, Count>& factors) {
    const auto area = positive_product(
        factors, "Area-light surface area is not representable in the requested precision.");
    if (!area) {
        return std::unexpected(area.error());
    }
    return area_measure(*area);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Bounds3T<Scalar>>
expanded_bounds(const Point3T<Scalar> minimum, const Point3T<Scalar> maximum,
                const Vector3T<Scalar> absolute_position_error) {
    const auto expanded_minimum = Point3T<Scalar>{
        .x = std::nextafter(minimum.x - absolute_position_error.x,
                            -std::numeric_limits<Scalar>::infinity()),
        .y = std::nextafter(minimum.y - absolute_position_error.y,
                            -std::numeric_limits<Scalar>::infinity()),
        .z = std::nextafter(minimum.z - absolute_position_error.z,
                            -std::numeric_limits<Scalar>::infinity()),
    };
    const auto expanded_maximum = Point3T<Scalar>{
        .x = std::nextafter(maximum.x + absolute_position_error.x,
                            std::numeric_limits<Scalar>::infinity()),
        .y = std::nextafter(maximum.y + absolute_position_error.y,
                            std::numeric_limits<Scalar>::infinity()),
        .z = std::nextafter(maximum.z + absolute_position_error.z,
                            std::numeric_limits<Scalar>::infinity()),
    };
    if (!light_detail::finite(expanded_minimum) || !light_detail::finite(expanded_maximum)) {
        return std::unexpected(invalid_area_light("Area-light bounds are not representable."));
    }
    return Bounds3T<Scalar>::from_minimum_maximum(expanded_minimum, expanded_maximum);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Bounds3T<Scalar>>
center_extent_bounds(const Point3T<Scalar> center, const Vector3T<Scalar> extent,
                     const Vector3T<Scalar> absolute_position_error) {
    if (!light_detail::finite(center) || !light_detail::finite_non_negative(extent)) {
        return std::unexpected(
            invalid_area_light("Area-light support bounds require finite geometry."));
    }
    const auto outward_extent = Vector3T<Scalar>{
        .x = std::nextafter(extent.x, std::numeric_limits<Scalar>::infinity()),
        .y = std::nextafter(extent.y, std::numeric_limits<Scalar>::infinity()),
        .z = std::nextafter(extent.z, std::numeric_limits<Scalar>::infinity()),
    };
    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
    constexpr auto gamma16 = (Scalar{16} * epsilon) / (Scalar{1} - Scalar{16} * epsilon);
    constexpr auto underflow_allowance = Scalar{16} * std::numeric_limits<Scalar>::denorm_min();
    const auto arithmetic_error = Vector3T<Scalar>{
        .x = std::fma(gamma16, std::abs(center.x) + outward_extent.x, underflow_allowance),
        .y = std::fma(gamma16, std::abs(center.y) + outward_extent.y, underflow_allowance),
        .z = std::fma(gamma16, std::abs(center.z) + outward_extent.z, underflow_allowance),
    };
    const auto total_error = Vector3T<Scalar>{
        .x = std::nextafter(absolute_position_error.x + arithmetic_error.x,
                            std::numeric_limits<Scalar>::infinity()),
        .y = std::nextafter(absolute_position_error.y + arithmetic_error.y,
                            std::numeric_limits<Scalar>::infinity()),
        .z = std::nextafter(absolute_position_error.z + arithmetic_error.z,
                            std::numeric_limits<Scalar>::infinity()),
    };
    if (!light_detail::finite_non_negative(outward_extent) ||
        !light_detail::finite_non_negative(arithmetic_error) ||
        !light_detail::finite_non_negative(total_error)) {
        return std::unexpected(
            invalid_area_light("Area-light support error bounds are not representable."));
    }
    return expanded_bounds(center - outward_extent, center + outward_extent, total_error);
}

template <SpectrumScalar Scalar> struct SurfacePointT final {
    Point3T<Scalar> position;
    Vector3T<Scalar> absolute_position_error;
    Normal3T<Scalar> geometric_normal;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::pair<Point3T<Scalar>, Vector3T<Scalar>>>
position_from_terms(const Point3T<Scalar> center, const Vector3T<Scalar> first,
                    const Vector3T<Scalar> second,
                    const Vector3T<Scalar> base_absolute_position_error) {
    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
    constexpr auto gamma16 = (Scalar{16} * epsilon) / (Scalar{1} - Scalar{16} * epsilon);
    constexpr auto underflow_allowance = Scalar{16} * std::numeric_limits<Scalar>::denorm_min();
    const auto center_values = std::array{center.x, center.y, center.z};
    const auto first_values = std::array{first.x, first.y, first.z};
    const auto second_values = std::array{second.x, second.y, second.z};
    const auto base_error_values =
        std::array{base_absolute_position_error.x, base_absolute_position_error.y,
                   base_absolute_position_error.z};

    auto point = std::array<Scalar, 3>{};
    auto error = std::array<Scalar, 3>{};
    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
        point[axis] = std::fma(Scalar{1}, first_values[axis],
                               std::fma(Scalar{1}, second_values[axis], center_values[axis]));
        const auto magnitude = std::abs(center_values[axis]) + std::abs(first_values[axis]) +
                               std::abs(second_values[axis]);
        const auto arithmetic_error = std::fma(gamma16, magnitude, underflow_allowance);
        error[axis] = std::nextafter(base_error_values[axis] + arithmetic_error,
                                     std::numeric_limits<Scalar>::infinity());
        if (!std::isfinite(point[axis]) || !std::isfinite(magnitude) ||
            !std::isfinite(arithmetic_error) || !std::isfinite(error[axis]) ||
            !(error[axis] > Scalar{0})) {
            return std::unexpected(
                invalid_area_light("Area-light sampled position is not representable."));
        }
    }
    return std::pair{
        Point3T<Scalar>{.x = point[0], .y = point[1], .z = point[2]},
        Vector3T<Scalar>{.x = error[0], .y = error[1], .z = error[2]},
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::pair<Point3T<Scalar>, Vector3T<Scalar>>>
triangle_position_with_error(const std::array<Point3T<Scalar>, 3>& vertices,
                             const std::array<Scalar, 3>& barycentrics,
                             const Vector3T<Scalar> base_absolute_position_error) {
    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
    constexpr auto gamma12 = (Scalar{12} * epsilon) / (Scalar{1} - Scalar{12} * epsilon);
    constexpr auto underflow_allowance = Scalar{12} * std::numeric_limits<Scalar>::denorm_min();
    const auto coordinates = std::array{
        std::array{vertices[0].x, vertices[0].y, vertices[0].z},
        std::array{vertices[1].x, vertices[1].y, vertices[1].z},
        std::array{vertices[2].x, vertices[2].y, vertices[2].z},
    };
    const auto base_error_values =
        std::array{base_absolute_position_error.x, base_absolute_position_error.y,
                   base_absolute_position_error.z};

    auto point = std::array<Scalar, 3>{};
    auto error = std::array<Scalar, 3>{};
    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
        const auto products = std::array{barycentrics[0] * coordinates[0][axis],
                                         barycentrics[1] * coordinates[1][axis],
                                         barycentrics[2] * coordinates[2][axis]};
        point[axis] = std::fma(barycentrics[0], coordinates[0][axis],
                               std::fma(barycentrics[1], coordinates[1][axis], products[2]));
        const auto magnitude =
            std::abs(products[0]) + std::abs(products[1]) + std::abs(products[2]);
        const auto arithmetic_error = std::fma(gamma12, magnitude, underflow_allowance);
        error[axis] = std::nextafter(base_error_values[axis] + arithmetic_error,
                                     std::numeric_limits<Scalar>::infinity());
        if (!std::isfinite(point[axis]) || !std::isfinite(magnitude) ||
            !std::isfinite(arithmetic_error) || !std::isfinite(error[axis]) ||
            !(error[axis] > Scalar{0})) {
            return std::unexpected(
                invalid_area_light("Mesh area-light sampled position is not representable."));
        }
    }
    return std::pair{
        Point3T<Scalar>{.x = point[0], .y = point[1], .z = point[2]},
        Vector3T<Scalar>{.x = error[0], .y = error[1], .z = error[2]},
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<bool> one_sided_support(const Normal3T<Scalar> geometric_normal,
                                                   const Vector3T<Scalar> outgoing_direction) {
    auto unit_radiance = LightSpectrumT<Scalar>{};
    unit_radiance.values.fill(Scalar{1});
    const auto emission = OneSidedSurfaceEmissionT<Scalar>::create(unit_radiance);
    if (!emission) {
        return std::unexpected(emission.error());
    }
    const auto evaluated = emission->eval(geometric_normal, outgoing_direction);
    if (!evaluated) {
        return std::unexpected(evaluated.error());
    }
    return (*evaluated)[0] > Scalar{0};
}

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr bool is_black(const LightSpectrumT<Scalar>& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (value != Scalar{0}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
finish_sample(const LightSampleContextT<Scalar>& context, const SurfacePointT<Scalar>& surface,
              const Scalar inverse_area, const AreaLightSidedness sidedness,
              const LightSpectrumT<Scalar>& radiance) {
    const auto solid_angle_probability = convert_area_pdf_to_solid_angle(
        LightProbabilityDensityT<Scalar>{
            .value = inverse_area,
            .measure = ProbabilityMeasure::area,
        },
        context.position(), surface.position, surface.geometric_normal);
    if (!solid_angle_probability) {
        return std::unexpected(solid_angle_probability.error());
    }
    if (sidedness == AreaLightSidedness::one_sided) {
        const auto supported =
            one_sided_support(surface.geometric_normal, context.position() - surface.position);
        if (!supported) {
            return std::unexpected(supported.error());
        }
        if (!*supported) {
            return std::optional<IncidentLightSampleT<Scalar>>{};
        }
    }

    const auto endpoint = LightSampleEndpointT<Scalar>::create_surface(
        surface.position, surface.absolute_position_error, surface.geometric_normal);
    if (!endpoint) {
        return std::unexpected(endpoint.error());
    }
    const auto sampled = IncidentLightSampleT<Scalar>::create_finite(context, *endpoint, radiance,
                                                                     *solid_angle_probability);
    if (!sampled) {
        return std::unexpected(sampled.error());
    }
    if (is_black(radiance)) {
        return std::optional<IncidentLightSampleT<Scalar>>{};
    }
    return std::optional<IncidentLightSampleT<Scalar>>{*sampled};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
finish_pdf(const LightSampleContextT<Scalar>& context, const Point3T<Scalar> surface_position,
           const Normal3T<Scalar> geometric_normal, const Scalar inverse_area,
           const AreaLightSidedness sidedness, const LightSpectrumT<Scalar>& radiance) {
    const auto solid_angle_probability = convert_area_pdf_to_solid_angle(
        LightProbabilityDensityT<Scalar>{
            .value = inverse_area,
            .measure = ProbabilityMeasure::area,
        },
        context.position(), surface_position, geometric_normal);
    if (!solid_angle_probability) {
        return std::unexpected(solid_angle_probability.error());
    }
    if (sidedness == AreaLightSidedness::one_sided) {
        const auto supported =
            one_sided_support(geometric_normal, context.position() - surface_position);
        if (!supported) {
            return std::unexpected(supported.error());
        }
        if (!*supported) {
            return DirectionalLightPdfT<Scalar>::create(Scalar{0});
        }
    }

    if (is_black(radiance)) {
        return DirectionalLightPdfT<Scalar>::create(Scalar{0});
    }
    return DirectionalLightPdfT<Scalar>::create(solid_angle_probability->value);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>> query_ray(const LightSampleContextT<Scalar>& context,
                                                   const Vector3T<Scalar> direction_to_light) {
    if (!light_detail::unit_direction(direction_to_light)) {
        return std::unexpected(
            invalid_area_light("Area-light PDF evaluation requires a finite unit direction."));
    }
    return RayT<Scalar>::create(context.position(), direction_to_light, Scalar{0},
                                std::numeric_limits<Scalar>::infinity(), context.time(),
                                AllRayVisibility, VacuumMedium);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<LightSpectrumT<Scalar>>
power_spectrum(const LightSpectrumT<Scalar>& radiance, const Scalar area,
               const AreaLightSidedness sidedness) {
    const auto side_factor = sidedness == AreaLightSidedness::one_sided ? Scalar{1} : Scalar{2};
    auto power = LightSpectrumT<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        if (radiance[lane] == Scalar{0}) {
            power[lane] = Scalar{0};
            continue;
        }
        const auto value = positive_product(
            std::array{radiance[lane], area, std::numbers::pi_v<Scalar>, side_factor},
            "Area-light radiant power is not representable in the requested precision.");
        if (!value) {
            return std::unexpected(value.error());
        }
        power[lane] = *value;
    }
    return power;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar> triangle_area(const TriangleT<Scalar>& triangle) {
    const auto& vertices = triangle.vertices();
    const auto spatial_exponents = triangle_detail::AxisExponents{
        .x = triangle_detail::scaling_exponent(
            std::array{vertices[0].x, vertices[1].x, vertices[2].x}),
        .y = triangle_detail::scaling_exponent(
            std::array{vertices[0].y, vertices[1].y, vertices[2].y}),
        .z = triangle_detail::scaling_exponent(
            std::array{vertices[0].z, vertices[1].z, vertices[2].z}),
    };
    const auto scaled_vertex0 = triangle_detail::scaled_point(vertices[0], spatial_exponents);
    const auto scaled_vertex1 = triangle_detail::scaled_point(vertices[1], spatial_exponents);
    const auto scaled_vertex2 = triangle_detail::scaled_point(vertices[2], spatial_exponents);
    if (!triangle_detail::scaling_preserves(vertices[0], scaled_vertex0) ||
        !triangle_detail::scaling_preserves(vertices[1], scaled_vertex1) ||
        !triangle_detail::scaling_preserves(vertices[2], scaled_vertex2)) {
        return std::unexpected(
            invalid_area_light("Mesh area-light vertex scaling is not representable."));
    }

    const auto first = triangle_detail::exact_relative_vector(scaled_vertex1, scaled_vertex0);
    const auto second = triangle_detail::exact_relative_vector(scaled_vertex2, scaled_vertex0);
    if (!first) {
        return std::unexpected(first.error());
    }
    if (!second) {
        return std::unexpected(second.error());
    }
    const auto cross_x =
        triangle_detail::exact_difference_of_products(first->y, second->z, first->z, second->y);
    const auto cross_y =
        triangle_detail::exact_difference_of_products(first->z, second->x, first->x, second->z);
    const auto cross_z =
        triangle_detail::exact_difference_of_products(first->x, second->y, first->y, second->x);
    if (!cross_x) {
        return std::unexpected(cross_x.error());
    }
    if (!cross_y) {
        return std::unexpected(cross_y.error());
    }
    if (!cross_z) {
        return std::unexpected(cross_z.error());
    }
    const auto x = triangle_detail::resolve_expansion(
        *cross_x, "Mesh area-light cross product x cannot be rounded faithfully.");
    const auto y = triangle_detail::resolve_expansion(
        *cross_y, "Mesh area-light cross product y cannot be rounded faithfully.");
    const auto z = triangle_detail::resolve_expansion(
        *cross_z, "Mesh area-light cross product z cannot be rounded faithfully.");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (!y) {
        return std::unexpected(y.error());
    }
    if (!z) {
        return std::unexpected(z.error());
    }

    const auto cross_exponents = triangle_detail::AxisExponents{
        .x = spatial_exponents.y + spatial_exponents.z,
        .y = spatial_exponents.z + spatial_exponents.x,
        .z = spatial_exponents.x + spatial_exponents.y,
    };
    auto magnitude_exponent = std::numeric_limits<int>::min();
    const auto include_component =
        [&magnitude_exponent](const triangle_detail::ExactValue<Scalar> component,
                              const int component_exponent) noexcept {
            if (component.sign == 0) {
                return;
            }
            auto value_exponent = 0;
            static_cast<void>(std::frexp(std::abs(component.value), &value_exponent));
            magnitude_exponent = std::max(magnitude_exponent, value_exponent + component_exponent);
        };
    include_component(*x, cross_exponents.x);
    include_component(*y, cross_exponents.y);
    include_component(*z, cross_exponents.z);
    if (magnitude_exponent == std::numeric_limits<int>::min()) {
        return std::unexpected(
            invalid_area_light("Mesh area-light triangles must be non-degenerate."));
    }

    const auto adjusted_x =
        x->sign == 0 ? Scalar{0} : std::ldexp(x->value, cross_exponents.x - magnitude_exponent);
    const auto adjusted_y =
        y->sign == 0 ? Scalar{0} : std::ldexp(y->value, cross_exponents.y - magnitude_exponent);
    const auto adjusted_z =
        z->sign == 0 ? Scalar{0} : std::ldexp(z->value, cross_exponents.z - magnitude_exponent);
    if ((x->sign != 0 && adjusted_x == Scalar{0}) || (y->sign != 0 && adjusted_y == Scalar{0}) ||
        (z->sign != 0 && adjusted_z == Scalar{0}) || !std::isfinite(adjusted_x) ||
        !std::isfinite(adjusted_y) || !std::isfinite(adjusted_z)) {
        return std::unexpected(
            invalid_area_light("Mesh area-light cross product scaling is not representable."));
    }
    const auto adjusted_length = std::hypot(adjusted_x, adjusted_y, adjusted_z);
    const auto area = std::scalbn(adjusted_length * Scalar{0.5}, magnitude_exponent);
    if (!std::isfinite(adjusted_length) || !(adjusted_length > Scalar{0}) || !std::isfinite(area) ||
        !(area > Scalar{0})) {
        return std::unexpected(invalid_area_light(
            "Mesh area-light triangle area is not representable in the requested precision."));
    }
    return area;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<TriangleT<Scalar>>
indexed_triangle(const std::vector<Point3T<Scalar>>& positions,
                 const AreaLightTriangleIndices indices) {
    const auto vertex_count = positions.size();
    if (indices.vertex0 >= vertex_count || indices.vertex1 >= vertex_count ||
        indices.vertex2 >= vertex_count) {
        return std::unexpected(
            invalid_area_light("Mesh area-light triangle index is out of range."));
    }
    return TriangleT<Scalar>::create(positions[indices.vertex0], positions[indices.vertex1],
                                     positions[indices.vertex2]);
}

} // namespace area_light_detail

// A world-space rectangle centered on an oriented local frame. half_width is
// measured along tangent and half_height along bitangent.
template <SpectrumScalar Scalar> class RectangleAreaLightT final {
  public:
    [[nodiscard]] static core::Result<RectangleAreaLightT>
    create(const Point3T<Scalar> center, const Normal3T<Scalar> normal,
           const Vector3T<Scalar> tangent, const Scalar half_width, const Scalar half_height,
           const Vector3T<Scalar> absolute_position_error, const AreaLightSidedness sidedness,
           const SampledWavelengthsT<Scalar>& wavelengths,
           const LightSpectrumT<Scalar>& emitted_radiance) {
        if (!std::isfinite(half_width) || !(half_width > Scalar{0}) ||
            !std::isfinite(half_height) || !(half_height > Scalar{0})) {
            return std::unexpected(area_light_detail::invalid_area_light(
                "Rectangle area-light half extents must be finite and positive."));
        }
        const auto error_status =
            area_light_detail::validate_position_error(absolute_position_error);
        if (!error_status) {
            return std::unexpected(error_status.error());
        }
        const auto sidedness_status = area_light_detail::validate_sidedness(sidedness);
        if (!sidedness_status) {
            return std::unexpected(sidedness_status.error());
        }
        const auto orientation =
            OrthonormalFrameT<Scalar>::from_normal_and_tangent(normal, tangent);
        if (!orientation) {
            return std::unexpected(orientation.error());
        }
        const auto plane = PlaneT<Scalar>::create(center, orientation->normal());
        if (!plane) {
            return std::unexpected(plane.error());
        }
        const auto measure =
            area_light_detail::area_measure(std::array{Scalar{4}, half_width, half_height});
        if (!measure) {
            return std::unexpected(measure.error());
        }
        const auto extent = Vector3T<Scalar>{
            .x = std::fma(std::abs(orientation->tangent().x), half_width,
                          std::abs(orientation->bitangent().x) * half_height),
            .y = std::fma(std::abs(orientation->tangent().y), half_width,
                          std::abs(orientation->bitangent().y) * half_height),
            .z = std::fma(std::abs(orientation->tangent().z), half_width,
                          std::abs(orientation->bitangent().z) * half_height),
        };
        const auto bounds =
            area_light_detail::center_extent_bounds(center, extent, absolute_position_error);
        if (!bounds) {
            return std::unexpected(bounds.error());
        }
        const auto radiance =
            light_detail::PacketLightSpectrumT<Scalar>::create(wavelengths, emitted_radiance);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return RectangleAreaLightT{
            *plane,    *orientation, half_width, half_height, absolute_position_error,
            sidedness, *measure,     *bounds,    *radiance};
    }

    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>& context, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto sample_status = area_light_detail::validate_canonical_sample(canonical_sample);
        if (!sample_status) {
            return std::unexpected(sample_status.error());
        }
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        const auto local_x = (Scalar{2} * canonical_sample.x - Scalar{1}) * half_width_;
        const auto local_y = (Scalar{2} * canonical_sample.y - Scalar{1}) * half_height_;
        const auto first = orientation_.tangent() * local_x;
        const auto second = orientation_.bitangent() * local_y;
        const auto positioned = area_light_detail::position_from_terms(
            plane_.point(), first, second, absolute_position_error_);
        if (!positioned) {
            return std::unexpected(positioned.error());
        }
        return area_light_detail::finish_sample(context,
                                                area_light_detail::SurfacePointT<Scalar>{
                                                    .position = positioned->first,
                                                    .absolute_position_error = positioned->second,
                                                    .geometric_normal = orientation_.normal(),
                                                },
                                                measure_.inverse_area, sidedness_, *radiance);
    }

    [[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>& context, const Vector3T<Scalar> direction_to_light,
           const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        const auto ray = area_light_detail::query_ray(context, direction_to_light);
        if (!ray) {
            return std::unexpected(ray.error());
        }
        const auto hit = plane_.intersect(*ray);
        if (!hit) {
            return std::unexpected(hit.error());
        }
        if (!hit->has_value()) {
            return DirectionalLightPdfT<Scalar>::create(Scalar{0});
        }
        const auto local = orientation_.to_local((**hit).position - plane_.point());
        if (!std::isfinite(local.x) || !std::isfinite(local.y) || std::abs(local.x) > half_width_ ||
            std::abs(local.y) > half_height_) {
            return DirectionalLightPdfT<Scalar>::create(Scalar{0});
        }
        return area_light_detail::finish_pdf(context, (**hit).position, orientation_.normal(),
                                             measure_.inverse_area, sidedness_, *radiance);
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    le(const RayT<Scalar>&, const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return LightSpectrumT<Scalar>{};
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    power(const Bounds3T<Scalar>& scene_bounds,
          const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto bounds_status = area_light_detail::validate_finite_scene_bounds(scene_bounds);
        if (!bounds_status) {
            return std::unexpected(bounds_status.error());
        }
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return area_light_detail::power_spectrum(*radiance, measure_.area, sidedness_);
    }

    [[nodiscard]] constexpr Bounds3T<Scalar> bounds() const noexcept {
        return bounds_;
    }
    [[nodiscard]] constexpr Scalar surface_area() const noexcept {
        return measure_.area;
    }
    [[nodiscard]] constexpr AreaLightSidedness sidedness() const noexcept {
        return sidedness_;
    }

  private:
    constexpr RectangleAreaLightT(
        const PlaneT<Scalar> plane, const OrthonormalFrameT<Scalar> orientation,
        const Scalar half_width, const Scalar half_height,
        const Vector3T<Scalar> absolute_position_error, const AreaLightSidedness sidedness,
        const area_light_detail::AreaMeasureT<Scalar> measure, const Bounds3T<Scalar> bounds,
        const light_detail::PacketLightSpectrumT<Scalar> radiance) noexcept
        : plane_{plane}, orientation_{orientation}, half_width_{half_width},
          half_height_{half_height}, absolute_position_error_{absolute_position_error},
          sidedness_{sidedness}, measure_{measure}, bounds_{bounds}, radiance_{radiance} {}

    PlaneT<Scalar> plane_;
    OrthonormalFrameT<Scalar> orientation_;
    Scalar half_width_;
    Scalar half_height_;
    Vector3T<Scalar> absolute_position_error_;
    AreaLightSidedness sidedness_;
    area_light_detail::AreaMeasureT<Scalar> measure_;
    Bounds3T<Scalar> bounds_;
    light_detail::PacketLightSpectrumT<Scalar> radiance_;
};

template <SpectrumScalar Scalar> class DiskAreaLightT final {
  public:
    [[nodiscard]] static core::Result<DiskAreaLightT>
    create(const Point3T<Scalar> center, const Normal3T<Scalar> normal, const Scalar radius,
           const Vector3T<Scalar> absolute_position_error, const AreaLightSidedness sidedness,
           const SampledWavelengthsT<Scalar>& wavelengths,
           const LightSpectrumT<Scalar>& emitted_radiance) {
        const auto error_status =
            area_light_detail::validate_position_error(absolute_position_error);
        if (!error_status) {
            return std::unexpected(error_status.error());
        }
        const auto sidedness_status = area_light_detail::validate_sidedness(sidedness);
        if (!sidedness_status) {
            return std::unexpected(sidedness_status.error());
        }
        const auto disk = DiskT<Scalar>::create(center, normal, radius);
        if (!disk) {
            return std::unexpected(disk.error());
        }
        const auto measure =
            area_light_detail::area_measure(std::array{std::numbers::pi_v<Scalar>, radius, radius});
        if (!measure) {
            return std::unexpected(measure.error());
        }
        const auto bounds = area_light_detail::center_extent_bounds(
            center, Vector3T<Scalar>{.x = radius, .y = radius, .z = radius},
            absolute_position_error);
        if (!bounds) {
            return std::unexpected(bounds.error());
        }
        const auto radiance =
            light_detail::PacketLightSpectrumT<Scalar>::create(wavelengths, emitted_radiance);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return DiskAreaLightT{*disk,    absolute_position_error, sidedness, *measure, *bounds,
                              *radiance};
    }

    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>& context, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto disk_sample = map_concentric_disk(canonical_sample);
        if (!disk_sample) {
            return std::unexpected(disk_sample.error());
        }
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        const auto first = disk_.orientation().tangent() * (disk_sample->x * disk_.radius());
        const auto second = disk_.orientation().bitangent() * (disk_sample->y * disk_.radius());
        const auto positioned = area_light_detail::position_from_terms(
            disk_.center(), first, second, absolute_position_error_);
        if (!positioned) {
            return std::unexpected(positioned.error());
        }
        const auto sampled =
            area_light_detail::finish_sample(context,
                                             area_light_detail::SurfacePointT<Scalar>{
                                                 .position = positioned->first,
                                                 .absolute_position_error = positioned->second,
                                                 .geometric_normal = disk_.orientation().normal(),
                                             },
                                             measure_.inverse_area, sidedness_, *radiance);
        return sampled;
    }

    [[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>& context, const Vector3T<Scalar> direction_to_light,
           const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        const auto ray = area_light_detail::query_ray(context, direction_to_light);
        if (!ray) {
            return std::unexpected(ray.error());
        }
        const auto hit = disk_.intersect(*ray);
        if (!hit) {
            return std::unexpected(hit.error());
        }
        if (!hit->has_value()) {
            return DirectionalLightPdfT<Scalar>::create(Scalar{0});
        }
        return area_light_detail::finish_pdf(context, (**hit).position, (**hit).geometric_normal,
                                             measure_.inverse_area, sidedness_, *radiance);
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    le(const RayT<Scalar>&, const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return LightSpectrumT<Scalar>{};
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    power(const Bounds3T<Scalar>& scene_bounds,
          const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto bounds_status = area_light_detail::validate_finite_scene_bounds(scene_bounds);
        if (!bounds_status) {
            return std::unexpected(bounds_status.error());
        }
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return area_light_detail::power_spectrum(*radiance, measure_.area, sidedness_);
    }

    [[nodiscard]] constexpr Bounds3T<Scalar> bounds() const noexcept {
        return bounds_;
    }
    [[nodiscard]] constexpr Scalar surface_area() const noexcept {
        return measure_.area;
    }
    [[nodiscard]] constexpr AreaLightSidedness sidedness() const noexcept {
        return sidedness_;
    }

  private:
    constexpr DiskAreaLightT(const DiskT<Scalar> disk,
                             const Vector3T<Scalar> absolute_position_error,
                             const AreaLightSidedness sidedness,
                             const area_light_detail::AreaMeasureT<Scalar> measure,
                             const Bounds3T<Scalar> bounds,
                             const light_detail::PacketLightSpectrumT<Scalar> radiance) noexcept
        : disk_{disk}, absolute_position_error_{absolute_position_error}, sidedness_{sidedness},
          measure_{measure}, bounds_{bounds}, radiance_{radiance} {}

    DiskT<Scalar> disk_;
    Vector3T<Scalar> absolute_position_error_;
    AreaLightSidedness sidedness_;
    area_light_detail::AreaMeasureT<Scalar> measure_;
    Bounds3T<Scalar> bounds_;
    light_detail::PacketLightSpectrumT<Scalar> radiance_;
};

template <SpectrumScalar Scalar> class SphereAreaLightT final {
  public:
    [[nodiscard]] static core::Result<SphereAreaLightT>
    create(const Point3T<Scalar> center, const Scalar radius,
           const Vector3T<Scalar> absolute_position_error, const AreaLightSidedness sidedness,
           const SampledWavelengthsT<Scalar>& wavelengths,
           const LightSpectrumT<Scalar>& emitted_radiance) {
        const auto error_status =
            area_light_detail::validate_position_error(absolute_position_error);
        if (!error_status) {
            return std::unexpected(error_status.error());
        }
        const auto sidedness_status = area_light_detail::validate_sidedness(sidedness);
        if (!sidedness_status) {
            return std::unexpected(sidedness_status.error());
        }
        const auto sphere = SphereT<Scalar>::create(center, radius);
        if (!sphere) {
            return std::unexpected(sphere.error());
        }
        const auto measure = area_light_detail::area_measure(
            std::array{Scalar{4}, std::numbers::pi_v<Scalar>, radius, radius});
        if (!measure) {
            return std::unexpected(measure.error());
        }
        const auto bounds = area_light_detail::center_extent_bounds(
            center, Vector3T<Scalar>{.x = radius, .y = radius, .z = radius},
            absolute_position_error);
        if (!bounds) {
            return std::unexpected(bounds.error());
        }
        const auto radiance =
            light_detail::PacketLightSpectrumT<Scalar>::create(wavelengths, emitted_radiance);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return SphereAreaLightT{*sphere,  absolute_position_error, sidedness, *measure, *bounds,
                                *radiance};
    }

    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>& context, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto sphere_sample = map_uniform_sphere(canonical_sample);
        if (!sphere_sample) {
            return std::unexpected(sphere_sample.error());
        }
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        const auto positioned = area_light_detail::position_from_terms(
            sphere_.center(), *sphere_sample * sphere_.radius(), Vector3T<Scalar>{},
            absolute_position_error_);
        if (!positioned) {
            return std::unexpected(positioned.error());
        }
        const auto sampled =
            area_light_detail::finish_sample(context,
                                             area_light_detail::SurfacePointT<Scalar>{
                                                 .position = positioned->first,
                                                 .absolute_position_error = positioned->second,
                                                 .geometric_normal =
                                                     Normal3T<Scalar>{
                                                         .x = sphere_sample->x,
                                                         .y = sphere_sample->y,
                                                         .z = sphere_sample->z,
                                                     },
                                             },
                                             measure_.inverse_area, sidedness_, *radiance);
        return sampled;
    }

    [[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>& context, const Vector3T<Scalar> direction_to_light,
           const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        const auto ray = area_light_detail::query_ray(context, direction_to_light);
        if (!ray) {
            return std::unexpected(ray.error());
        }
        const auto hit = sphere_.intersect(*ray);
        if (!hit) {
            return std::unexpected(hit.error());
        }
        if (!hit->has_value()) {
            return DirectionalLightPdfT<Scalar>::create(Scalar{0});
        }
        return area_light_detail::finish_pdf(context, (**hit).position, (**hit).geometric_normal,
                                             measure_.inverse_area, sidedness_, *radiance);
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    le(const RayT<Scalar>&, const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return LightSpectrumT<Scalar>{};
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    power(const Bounds3T<Scalar>& scene_bounds,
          const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto bounds_status = area_light_detail::validate_finite_scene_bounds(scene_bounds);
        if (!bounds_status) {
            return std::unexpected(bounds_status.error());
        }
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return area_light_detail::power_spectrum(*radiance, measure_.area, sidedness_);
    }

    [[nodiscard]] constexpr Bounds3T<Scalar> bounds() const noexcept {
        return bounds_;
    }
    [[nodiscard]] constexpr Scalar surface_area() const noexcept {
        return measure_.area;
    }
    [[nodiscard]] constexpr AreaLightSidedness sidedness() const noexcept {
        return sidedness_;
    }

  private:
    constexpr SphereAreaLightT(const SphereT<Scalar> sphere,
                               const Vector3T<Scalar> absolute_position_error,
                               const AreaLightSidedness sidedness,
                               const area_light_detail::AreaMeasureT<Scalar> measure,
                               const Bounds3T<Scalar> bounds,
                               const light_detail::PacketLightSpectrumT<Scalar> radiance) noexcept
        : sphere_{sphere}, absolute_position_error_{absolute_position_error}, sidedness_{sidedness},
          measure_{measure}, bounds_{bounds}, radiance_{radiance} {}

    SphereT<Scalar> sphere_;
    Vector3T<Scalar> absolute_position_error_;
    AreaLightSidedness sidedness_;
    area_light_detail::AreaMeasureT<Scalar> measure_;
    Bounds3T<Scalar> bounds_;
    light_detail::PacketLightSpectrumT<Scalar> radiance_;
};

// The mesh owns one compact indexed position buffer. Per-triangle cumulative
// areas are derived data used for deterministic area-proportional sampling.
template <SpectrumScalar Scalar> class MeshAreaLightT final {
  public:
    [[nodiscard]] static core::Result<MeshAreaLightT>
    create(std::vector<Point3T<Scalar>> positions, std::vector<AreaLightTriangleIndices> triangles,
           const Vector3T<Scalar> absolute_position_error, const AreaLightSidedness sidedness,
           const SampledWavelengthsT<Scalar>& wavelengths,
           const LightSpectrumT<Scalar>& emitted_radiance) {
        const auto error_status =
            area_light_detail::validate_position_error(absolute_position_error);
        if (!error_status) {
            return std::unexpected(error_status.error());
        }
        const auto sidedness_status = area_light_detail::validate_sidedness(sidedness);
        if (!sidedness_status) {
            return std::unexpected(sidedness_status.error());
        }
        if (positions.empty() || triangles.empty()) {
            return std::unexpected(area_light_detail::invalid_area_light(
                "A mesh area light requires positions and triangles."));
        }
        if (positions.size() > std::numeric_limits<std::uint32_t>::max() ||
            triangles.size() > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(area_light_detail::exhausted_area_light(
                "Mesh area-light counts exceed the supported 32-bit index domain."));
        }
        for (const auto position : positions) {
            if (!light_detail::finite(position)) {
                return std::unexpected(area_light_detail::invalid_area_light(
                    "Mesh area-light positions must be finite."));
            }
        }

        auto triangle_areas = std::vector<Scalar>{};
        auto area_sum = Scalar{0};
        auto area_compensation = Scalar{0};
        try {
            triangle_areas.reserve(triangles.size());
            for (const auto indices : triangles) {
                const auto triangle = area_light_detail::indexed_triangle(positions, indices);
                if (!triangle) {
                    return std::unexpected(triangle.error());
                }
                const auto triangle_area = area_light_detail::triangle_area(*triangle);
                if (!triangle_area) {
                    return std::unexpected(triangle_area.error());
                }
                const auto next_area = area_sum + *triangle_area;
                if (!std::isfinite(next_area)) {
                    return std::unexpected(area_light_detail::invalid_area_light(
                        "Mesh area-light total area is not representable."));
                }
                if (std::abs(area_sum) >= std::abs(*triangle_area)) {
                    area_compensation += (area_sum - next_area) + *triangle_area;
                } else {
                    area_compensation += (*triangle_area - next_area) + area_sum;
                }
                if (!std::isfinite(area_compensation)) {
                    return std::unexpected(area_light_detail::invalid_area_light(
                        "Mesh area-light compensated area sum is not representable."));
                }
                area_sum = next_area;
                triangle_areas.push_back(*triangle_area);
            }
        } catch (const std::bad_alloc&) {
            return std::unexpected(area_light_detail::exhausted_area_light(
                "Mesh area-light distribution exhausted host memory."));
        } catch (const std::length_error&) {
            return std::unexpected(area_light_detail::exhausted_area_light(
                "Mesh area-light distribution exceeded host container limits."));
        }
        const auto total_area = area_sum + area_compensation;
        const auto measure = area_light_detail::area_measure(total_area);
        if (!measure) {
            return std::unexpected(measure.error());
        }

        auto cumulative_probabilities = std::vector<Scalar>{};
        auto cumulative_probability = Scalar{0};
        try {
            cumulative_probabilities.reserve(triangle_areas.size());
            for (auto triangle_index = std::size_t{0}; triangle_index < triangle_areas.size();
                 ++triangle_index) {
                auto next_probability = Scalar{1};
                if (triangle_index + 1U != triangle_areas.size()) {
                    const auto selection_probability =
                        triangle_areas[triangle_index] / measure->area;
                    next_probability = cumulative_probability + selection_probability;
                    if (!std::isfinite(selection_probability) ||
                        !(selection_probability > Scalar{0}) || !std::isfinite(next_probability) ||
                        !(next_probability > cumulative_probability) ||
                        !(next_probability < Scalar{1})) {
                        return std::unexpected(area_light_detail::invalid_area_light(
                            "Mesh area-light selection probabilities are not distinguishable in "
                            "the requested precision."));
                    }
                }
                cumulative_probability = next_probability;
                cumulative_probabilities.push_back(cumulative_probability);
            }
        } catch (const std::bad_alloc&) {
            return std::unexpected(area_light_detail::exhausted_area_light(
                "Mesh area-light CDF exhausted host memory."));
        } catch (const std::length_error&) {
            return std::unexpected(area_light_detail::exhausted_area_light(
                "Mesh area-light CDF exceeded host container limits."));
        }

        auto minimum = positions.front();
        auto maximum = positions.front();
        for (const auto position : positions) {
            minimum.x = std::min(minimum.x, position.x);
            minimum.y = std::min(minimum.y, position.y);
            minimum.z = std::min(minimum.z, position.z);
            maximum.x = std::max(maximum.x, position.x);
            maximum.y = std::max(maximum.y, position.y);
            maximum.z = std::max(maximum.z, position.z);
        }
        constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
        constexpr auto gamma12 = (Scalar{12} * epsilon) / (Scalar{1} - Scalar{12} * epsilon);
        constexpr auto underflow_allowance = Scalar{12} * std::numeric_limits<Scalar>::denorm_min();
        const auto interpolation_error = Vector3T<Scalar>{
            .x = std::fma(gamma12, Scalar{3} * std::max(std::abs(minimum.x), std::abs(maximum.x)),
                          underflow_allowance),
            .y = std::fma(gamma12, Scalar{3} * std::max(std::abs(minimum.y), std::abs(maximum.y)),
                          underflow_allowance),
            .z = std::fma(gamma12, Scalar{3} * std::max(std::abs(minimum.z), std::abs(maximum.z)),
                          underflow_allowance),
        };
        const auto bounds_error = Vector3T<Scalar>{
            .x = std::nextafter(absolute_position_error.x + interpolation_error.x,
                                std::numeric_limits<Scalar>::infinity()),
            .y = std::nextafter(absolute_position_error.y + interpolation_error.y,
                                std::numeric_limits<Scalar>::infinity()),
            .z = std::nextafter(absolute_position_error.z + interpolation_error.z,
                                std::numeric_limits<Scalar>::infinity()),
        };
        if (!light_detail::finite_non_negative(interpolation_error) ||
            !light_detail::finite_non_negative(bounds_error)) {
            return std::unexpected(area_light_detail::invalid_area_light(
                "Mesh area-light interpolation bounds are not representable."));
        }
        const auto bounds = area_light_detail::expanded_bounds(minimum, maximum, bounds_error);
        if (!bounds) {
            return std::unexpected(bounds.error());
        }
        const auto radiance =
            light_detail::PacketLightSpectrumT<Scalar>::create(wavelengths, emitted_radiance);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return MeshAreaLightT{std::move(positions),
                              std::move(triangles),
                              std::move(triangle_areas),
                              std::move(cumulative_probabilities),
                              absolute_position_error,
                              sidedness,
                              *measure,
                              *bounds,
                              *radiance};
    }

    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>& context, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto sample_status = area_light_detail::validate_canonical_sample(canonical_sample);
        if (!sample_status) {
            return std::unexpected(sample_status.error());
        }
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }

        const auto selected = std::upper_bound(cumulative_probabilities_.begin(),
                                               cumulative_probabilities_.end(), canonical_sample.x);
        if (selected == cumulative_probabilities_.end()) {
            return std::unexpected(area_light_detail::invalid_area_light(
                "Mesh area-light selection escaped its cumulative distribution."));
        }
        const auto triangle_index =
            static_cast<std::size_t>(selected - cumulative_probabilities_.begin());
        const auto previous_probability =
            triangle_index == 0U ? Scalar{0} : cumulative_probabilities_[triangle_index - 1U];
        const auto selection_probability =
            cumulative_probabilities_[triangle_index] - previous_probability;
        const auto remapped_x = (canonical_sample.x - previous_probability) / selection_probability;
        if (!std::isfinite(selection_probability) || !(selection_probability > Scalar{0}) ||
            !std::isfinite(remapped_x) || remapped_x < Scalar{0} || !(remapped_x < Scalar{1})) {
            return std::unexpected(area_light_detail::invalid_area_light(
                "Mesh area-light canonical remapping is not representable."));
        }
        const auto area_density = triangle_area_density(triangle_index);
        if (!area_density) {
            return std::unexpected(area_density.error());
        }
        const auto root = std::sqrt(remapped_x);
        const auto barycentrics = std::array{
            Scalar{1} - root,
            root * (Scalar{1} - canonical_sample.y),
            root * canonical_sample.y,
        };
        const auto triangle =
            area_light_detail::indexed_triangle(positions_, triangles_[triangle_index]);
        if (!triangle) {
            return std::unexpected(triangle.error());
        }
        const auto positioned = area_light_detail::triangle_position_with_error(
            triangle->vertices(), barycentrics, absolute_position_error_);
        if (!positioned) {
            return std::unexpected(positioned.error());
        }
        const auto sampled =
            area_light_detail::finish_sample(context,
                                             area_light_detail::SurfacePointT<Scalar>{
                                                 .position = positioned->first,
                                                 .absolute_position_error = positioned->second,
                                                 .geometric_normal = triangle->geometric_normal(),
                                             },
                                             *area_density, sidedness_, *radiance);
        return sampled;
    }

    [[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>& context, const Vector3T<Scalar> direction_to_light,
           const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        const auto ray = area_light_detail::query_ray(context, direction_to_light);
        if (!ray) {
            return std::unexpected(ray.error());
        }

        auto closest_parameter = std::numeric_limits<Scalar>::infinity();
        auto closest_position = Point3T<Scalar>{};
        auto closest_normal = Normal3T<Scalar>{};
        auto closest_triangle_index = std::size_t{0};
        auto found_hit = false;
        for (auto triangle_index = std::size_t{0}; triangle_index < triangles_.size();
             ++triangle_index) {
            const auto indices = triangles_[triangle_index];
            const auto triangle = area_light_detail::indexed_triangle(positions_, indices);
            if (!triangle) {
                return std::unexpected(triangle.error());
            }
            const auto hit = triangle->intersect(*ray);
            if (!hit) {
                return std::unexpected(hit.error());
            }
            if (!hit->has_value()) {
                continue;
            }
            if (!found_hit || (**hit).parameter < closest_parameter) {
                closest_parameter = (**hit).parameter;
                closest_position = (**hit).position;
                closest_normal = (**hit).geometric_normal;
                closest_triangle_index = triangle_index;
                found_hit = true;
            }
        }
        if (!found_hit) {
            return DirectionalLightPdfT<Scalar>::create(Scalar{0});
        }
        const auto area_density = triangle_area_density(closest_triangle_index);
        if (!area_density) {
            return std::unexpected(area_density.error());
        }
        return area_light_detail::finish_pdf(context, closest_position, closest_normal,
                                             *area_density, sidedness_, *radiance);
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    le(const RayT<Scalar>&, const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return LightSpectrumT<Scalar>{};
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    power(const Bounds3T<Scalar>& scene_bounds,
          const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto bounds_status = area_light_detail::validate_finite_scene_bounds(scene_bounds);
        if (!bounds_status) {
            return std::unexpected(bounds_status.error());
        }
        const auto radiance = radiance_.evaluate(wavelengths);
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        return area_light_detail::power_spectrum(*radiance, measure_.area, sidedness_);
    }

    [[nodiscard]] constexpr Bounds3T<Scalar> bounds() const noexcept {
        return bounds_;
    }
    [[nodiscard]] constexpr Scalar surface_area() const noexcept {
        return measure_.area;
    }
    [[nodiscard]] constexpr AreaLightSidedness sidedness() const noexcept {
        return sidedness_;
    }
    [[nodiscard]] const std::vector<Point3T<Scalar>>& positions() const noexcept {
        return positions_;
    }
    [[nodiscard]] const std::vector<AreaLightTriangleIndices>& triangles() const noexcept {
        return triangles_;
    }

  private:
    [[nodiscard]] core::Result<Scalar>
    triangle_area_density(const std::size_t triangle_index) const {
        if (triangle_index >= triangle_areas_.size() ||
            triangle_index >= cumulative_probabilities_.size()) {
            return std::unexpected(area_light_detail::invalid_area_light(
                "Mesh area-light triangle density index is out of range."));
        }
        const auto previous_probability =
            triangle_index == 0U ? Scalar{0} : cumulative_probabilities_[triangle_index - 1U];
        const auto selection_probability =
            cumulative_probabilities_[triangle_index] - previous_probability;
        const auto density = selection_probability / triangle_areas_[triangle_index];
        if (!std::isfinite(selection_probability) || !(selection_probability > Scalar{0}) ||
            !std::isfinite(density) || !(density > Scalar{0})) {
            return std::unexpected(area_light_detail::invalid_area_light(
                "Mesh area-light triangle density is not representable."));
        }
        return density;
    }

    MeshAreaLightT(std::vector<Point3T<Scalar>> positions,
                   std::vector<AreaLightTriangleIndices> triangles,
                   std::vector<Scalar> triangle_areas, std::vector<Scalar> cumulative_probabilities,
                   const Vector3T<Scalar> absolute_position_error,
                   const AreaLightSidedness sidedness,
                   const area_light_detail::AreaMeasureT<Scalar> measure,
                   const Bounds3T<Scalar> bounds,
                   const light_detail::PacketLightSpectrumT<Scalar> radiance) noexcept
        : positions_{std::move(positions)}, triangles_{std::move(triangles)},
          triangle_areas_{std::move(triangle_areas)},
          cumulative_probabilities_{std::move(cumulative_probabilities)},
          absolute_position_error_{absolute_position_error}, sidedness_{sidedness},
          measure_{measure}, bounds_{bounds}, radiance_{radiance} {}

    std::vector<Point3T<Scalar>> positions_;
    std::vector<AreaLightTriangleIndices> triangles_;
    std::vector<Scalar> triangle_areas_;
    std::vector<Scalar> cumulative_probabilities_;
    Vector3T<Scalar> absolute_position_error_;
    AreaLightSidedness sidedness_;
    area_light_detail::AreaMeasureT<Scalar> measure_;
    Bounds3T<Scalar> bounds_;
    light_detail::PacketLightSpectrumT<Scalar> radiance_;
};

using RectangleAreaLight = RectangleAreaLightT<TransportScalar>;
using ReferenceRectangleAreaLight = RectangleAreaLightT<ReferenceScalar>;
using DiskAreaLight = DiskAreaLightT<TransportScalar>;
using ReferenceDiskAreaLight = DiskAreaLightT<ReferenceScalar>;
using SphereAreaLight = SphereAreaLightT<TransportScalar>;
using ReferenceSphereAreaLight = SphereAreaLightT<ReferenceScalar>;
using MeshAreaLight = MeshAreaLightT<TransportScalar>;
using ReferenceMeshAreaLight = MeshAreaLightT<ReferenceScalar>;

static_assert(LightModelFor<RectangleAreaLight, TransportScalar>);
static_assert(LightModelFor<ReferenceRectangleAreaLight, ReferenceScalar>);
static_assert(LightModelFor<DiskAreaLight, TransportScalar>);
static_assert(LightModelFor<ReferenceDiskAreaLight, ReferenceScalar>);
static_assert(LightModelFor<SphereAreaLight, TransportScalar>);
static_assert(LightModelFor<ReferenceSphereAreaLight, ReferenceScalar>);
static_assert(LightModelFor<MeshAreaLight, TransportScalar>);
static_assert(LightModelFor<ReferenceMeshAreaLight, ReferenceScalar>);
static_assert(std::is_standard_layout_v<AreaLightTriangleIndices>);
static_assert(std::is_trivially_copyable_v<AreaLightTriangleIndices>);

} // namespace blackframe::renderer
