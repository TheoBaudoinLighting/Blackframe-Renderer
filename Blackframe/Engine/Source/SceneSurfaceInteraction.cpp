#include <Blackframe/Engine/SceneSurfaceInteraction.hpp>
#include <Blackframe/Renderer/GeometryOperations.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error surface_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

[[nodiscard]] core::Status validate_hit_identity(const FrameScene& scene, const AccelHit& hit) {
    const auto instance = scene.instance(hit.identifiers.instance);
    if (!instance) {
        return std::unexpected(instance.error());
    }
    if (instance->get().object != hit.object ||
        instance->get().geometry != hit.identifiers.geometry ||
        instance->get().material != hit.identifiers.material) {
        return std::unexpected(surface_error(
            core::StatusCode::incompatible,
            "An acceleration hit does not match its committed frame scene instance."));
    }
    return {};
}

[[nodiscard]] core::Result<std::reference_wrapper<const SceneSpectralMaterial>>
surface_material(const FrameScene& scene, const AccelHit& hit) {
    const auto material = scene.material(hit.identifiers.material);
    if (!material) {
        return std::unexpected(material.error());
    }
    if (!material->get().spectral) {
        return std::unexpected(
            surface_error(core::StatusCode::unavailable,
                          "The hit frame scene material has no spectral transport data."));
    }
    return std::cref(*material->get().spectral);
}

[[nodiscard]] bool has_surface_map(const SceneSpectralMaterial& material) noexcept {
    return material.normal_map.has_value() || material.bump_map.has_value();
}

[[nodiscard]] core::Status
validate_barycentrics(const renderer::TriangleBarycentrics barycentrics) {
    const auto values =
        std::array{barycentrics.vertex0, barycentrics.vertex1, barycentrics.vertex2};
    constexpr auto tolerance =
        renderer::TransportScalar{32} * std::numeric_limits<renderer::TransportScalar>::epsilon();
    for (const auto value : values) {
        if (!std::isfinite(value) || value < -tolerance ||
            value > renderer::TransportScalar{1} + tolerance) {
            return std::unexpected(surface_error(
                core::StatusCode::invalid_argument,
                "An acceleration hit contains invalid triangle barycentric coordinates."));
        }
    }
    const auto sum = values[0] + values[1] + values[2];
    if (!std::isfinite(sum) || std::abs(sum - renderer::TransportScalar{1}) > tolerance) {
        return std::unexpected(surface_error(
            core::StatusCode::invalid_argument,
            "An acceleration hit contains barycentric coordinates that do not sum to one."));
    }
    return {};
}

struct PositionWithError final {
    renderer::Point3 point;
    renderer::Vector3 absolute_error;
};

[[nodiscard]] core::Result<PositionWithError>
interpolate_position_with_error(const std::array<renderer::Point3, 3>& vertices,
                                const renderer::TriangleBarycentrics barycentrics) {
    constexpr auto epsilon = std::numeric_limits<renderer::TransportScalar>::epsilon();
    constexpr auto gamma7 = (renderer::TransportScalar{7} * epsilon) /
                            (renderer::TransportScalar{1} - renderer::TransportScalar{7} * epsilon);
    const auto weights =
        std::array{barycentrics.vertex0, barycentrics.vertex1, barycentrics.vertex2};
    const auto coordinates = std::array{
        std::array{vertices[0].x, vertices[0].y, vertices[0].z},
        std::array{vertices[1].x, vertices[1].y, vertices[1].z},
        std::array{vertices[2].x, vertices[2].y, vertices[2].z},
    };

    auto triangle_scale = renderer::TransportScalar{0};
    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
        triangle_scale =
            std::max({triangle_scale, std::abs(coordinates[1][axis] - coordinates[0][axis]),
                      std::abs(coordinates[2][axis] - coordinates[0][axis])});
    }
    const auto representable_floor = epsilon * triangle_scale;
    if (!std::isfinite(triangle_scale) || !std::isfinite(representable_floor) ||
        !(representable_floor > renderer::TransportScalar{0})) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A scene triangle position-error scale is not representable."));
    }

    auto point = std::array<renderer::TransportScalar, 3>{};
    auto error = std::array<renderer::TransportScalar, 3>{};
    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
        const auto products =
            std::array{weights[0] * coordinates[0][axis], weights[1] * coordinates[1][axis],
                       weights[2] * coordinates[2][axis]};
        point[axis] = std::fma(weights[0], coordinates[0][axis],
                               std::fma(weights[1], coordinates[1][axis], products[2]));
        const auto interpolation_magnitude =
            std::abs(products[0]) + std::abs(products[1]) + std::abs(products[2]);
        error[axis] = std::max(gamma7 * interpolation_magnitude, representable_floor);
        if (!std::isfinite(point[axis]) || !std::isfinite(interpolation_magnitude) ||
            !std::isfinite(error[axis]) || error[axis] < renderer::TransportScalar{0}) {
            return std::unexpected(
                surface_error(core::StatusCode::invalid_argument,
                              "A scene triangle position reconstruction is not representable."));
        }
    }

    return PositionWithError{
        .point = {.x = point[0], .y = point[1], .z = point[2]},
        .absolute_error = {.x = error[0], .y = error[1], .z = error[2]},
    };
}

[[nodiscard]] core::Result<PositionWithError>
transform_position_with_error(const renderer::AffineTransform& transform,
                              const PositionWithError& object_position,
                              const renderer::Point3 hit_position) {
    const auto world_position = transform.apply(object_position.point);
    const auto source =
        std::array{object_position.point.x, object_position.point.y, object_position.point.z};
    const auto source_error =
        std::array{object_position.absolute_error.x, object_position.absolute_error.y,
                   object_position.absolute_error.z};
    const auto world = std::array{world_position.x, world_position.y, world_position.z};
    const auto hit = std::array{hit_position.x, hit_position.y, hit_position.z};
    constexpr auto epsilon =
        static_cast<double>(std::numeric_limits<renderer::TransportScalar>::epsilon());
    constexpr auto gamma7 = (7.0 * epsilon) / (1.0 - 7.0 * epsilon);
    constexpr auto maximum =
        static_cast<double>(std::numeric_limits<renderer::TransportScalar>::max());

    auto error = std::array<renderer::TransportScalar, 3>{};
    for (auto row = std::size_t{0}; row < 3U; ++row) {
        auto transformed_magnitude = std::abs(static_cast<double>(transform.matrix()(row, 3)));
        auto propagated_error = 0.0;
        for (auto column = std::size_t{0}; column < 3U; ++column) {
            const auto coefficient = std::abs(static_cast<double>(transform.matrix()(row, column)));
            transformed_magnitude += coefficient * std::abs(static_cast<double>(source[column]));
            propagated_error += coefficient * static_cast<double>(source_error[column]);
        }
        const auto traversal_discrepancy =
            std::abs(static_cast<double>(world[row]) - static_cast<double>(hit[row]));
        const auto bound = gamma7 * transformed_magnitude + (1.0 + gamma7) * propagated_error +
                           traversal_discrepancy;
        if (!std::isfinite(static_cast<double>(world[row])) || !std::isfinite(bound) ||
            bound < 0.0 || bound > maximum) {
            return std::unexpected(
                surface_error(core::StatusCode::invalid_argument,
                              "A transformed scene triangle position error is not representable."));
        }
        error[row] = static_cast<renderer::TransportScalar>(bound);
        if (static_cast<double>(error[row]) < bound) {
            error[row] = std::nextafter(error[row],
                                        std::numeric_limits<renderer::TransportScalar>::infinity());
        }
        if (!std::isfinite(error[row]) || error[row] < renderer::TransportScalar{0} ||
            (bound > 0.0 && error[row] == renderer::TransportScalar{0})) {
            return std::unexpected(
                surface_error(core::StatusCode::invalid_argument,
                              "A transformed scene triangle position error is not representable."));
        }
    }

    return PositionWithError{
        .point = world_position,
        .absolute_error = {.x = error[0], .y = error[1], .z = error[2]},
    };
}

[[nodiscard]] renderer::Point2
interpolate_texture_coordinate(const std::array<renderer::Point2, 3>& coordinates,
                               const renderer::TriangleBarycentrics barycentrics) noexcept {
    return {
        .x = std::fma(barycentrics.vertex0, coordinates[0].x,
                      std::fma(barycentrics.vertex1, coordinates[1].x,
                               barycentrics.vertex2 * coordinates[2].x)),
        .y = std::fma(barycentrics.vertex0, coordinates[0].y,
                      std::fma(barycentrics.vertex1, coordinates[1].y,
                               barycentrics.vertex2 * coordinates[2].y)),
    };
}

[[nodiscard]] core::Result<int> transform_orientation(const renderer::AffineTransform& transform) {
    auto matrix = std::array<double, 9>{};
    for (auto row = std::size_t{0}; row < 3U; ++row) {
        for (auto column = std::size_t{0}; column < 3U; ++column) {
            matrix[row * 3U + column] = static_cast<double>(transform.matrix()(row, column));
        }
    }

    auto orientation = 1;
    for (auto column = std::size_t{0}; column < 3U; ++column) {
        auto pivot_row = column;
        auto pivot_magnitude = std::abs(matrix[column * 3U + column]);
        for (auto candidate = column + 1U; candidate < 3U; ++candidate) {
            const auto candidate_magnitude = std::abs(matrix[candidate * 3U + column]);
            if (candidate_magnitude > pivot_magnitude) {
                pivot_magnitude = candidate_magnitude;
                pivot_row = candidate;
            }
        }
        if (!std::isfinite(pivot_magnitude) || pivot_magnitude == 0.0) {
            return std::unexpected(
                surface_error(core::StatusCode::invalid_argument,
                              "A scene instance transform orientation is not representable."));
        }
        if (pivot_row != column) {
            for (auto swap_column = column; swap_column < 3U; ++swap_column) {
                std::swap(matrix[column * 3U + swap_column], matrix[pivot_row * 3U + swap_column]);
            }
            orientation = -orientation;
        }

        const auto pivot = matrix[column * 3U + column];
        orientation = pivot < 0.0 ? -orientation : orientation;
        for (auto row = column + 1U; row < 3U; ++row) {
            const auto factor = matrix[row * 3U + column] / pivot;
            for (auto update_column = column + 1U; update_column < 3U; ++update_column) {
                matrix[row * 3U + update_column] -= factor * matrix[column * 3U + update_column];
                if (!std::isfinite(matrix[row * 3U + update_column])) {
                    return std::unexpected(surface_error(
                        core::StatusCode::invalid_argument,
                        "A scene instance transform orientation is not representable."));
                }
            }
        }
    }
    return orientation;
}

[[nodiscard]] core::Result<renderer::Normal3>
robust_transformed_normal(const renderer::AffineTransform& transform,
                          const std::array<double, 3>& normal) {
    const auto& inverse = transform.inverse_matrix();
    const auto transformed = std::array{
        std::fma(static_cast<double>(inverse(0, 0)), normal[0],
                 std::fma(static_cast<double>(inverse(1, 0)), normal[1],
                          static_cast<double>(inverse(2, 0)) * normal[2])),
        std::fma(static_cast<double>(inverse(0, 1)), normal[0],
                 std::fma(static_cast<double>(inverse(1, 1)), normal[1],
                          static_cast<double>(inverse(2, 1)) * normal[2])),
        std::fma(static_cast<double>(inverse(0, 2)), normal[0],
                 std::fma(static_cast<double>(inverse(1, 2)), normal[1],
                          static_cast<double>(inverse(2, 2)) * normal[2])),
    };
    const auto maximum_component =
        std::max({std::abs(transformed[0]), std::abs(transformed[1]), std::abs(transformed[2])});
    if (!std::isfinite(maximum_component) || maximum_component == 0.0) {
        return std::unexpected(surface_error(core::StatusCode::invalid_argument,
                                             "A scene shading normal cannot be normalized."));
    }
    const auto scaled =
        std::array{transformed[0] / maximum_component, transformed[1] / maximum_component,
                   transformed[2] / maximum_component};
    const auto squared_length =
        std::fma(scaled[0], scaled[0], std::fma(scaled[1], scaled[1], scaled[2] * scaled[2]));
    const auto magnitude = std::sqrt(squared_length);
    if (!std::isfinite(magnitude) || !(magnitude > 0.0)) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A scene shading normal normalization is not representable."));
    }
    const auto result = renderer::Normal3{
        .x = static_cast<renderer::TransportScalar>(scaled[0] / magnitude),
        .y = static_cast<renderer::TransportScalar>(scaled[1] / magnitude),
        .z = static_cast<renderer::TransportScalar>(scaled[2] / magnitude),
    };
    if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A scene shading normal normalization is not representable."));
    }
    return result;
}

[[nodiscard]] core::Result<renderer::Normal3>
robust_transformed_normal(const renderer::AffineTransform& transform,
                          const renderer::Normal3 normal) {
    return robust_transformed_normal(transform, std::array{static_cast<double>(normal.x),
                                                           static_cast<double>(normal.y),
                                                           static_cast<double>(normal.z)});
}

[[nodiscard]] core::Result<renderer::Normal3>
interpolate_shading_normal(const std::array<renderer::Normal3, 3>& normals,
                           const renderer::TriangleBarycentrics barycentrics,
                           const renderer::AffineTransform& object_to_world,
                           const renderer::Normal3 geometric_normal) {
    const auto object_normal = renderer::Normal3{
        .x = std::fma(
            barycentrics.vertex0, normals[0].x,
            std::fma(barycentrics.vertex1, normals[1].x, barycentrics.vertex2 * normals[2].x)),
        .y = std::fma(
            barycentrics.vertex0, normals[0].y,
            std::fma(barycentrics.vertex1, normals[1].y, barycentrics.vertex2 * normals[2].y)),
        .z = std::fma(
            barycentrics.vertex0, normals[0].z,
            std::fma(barycentrics.vertex1, normals[1].z, barycentrics.vertex2 * normals[2].z)),
    };
    auto shading_normal = robust_transformed_normal(object_to_world, object_normal);
    if (!shading_normal) {
        return std::unexpected(shading_normal.error());
    }
    const auto orientation = transform_orientation(object_to_world);
    if (!orientation) {
        return std::unexpected(orientation.error());
    }
    if (*orientation < 0) {
        *shading_normal = -*shading_normal;
    }
    if (!(renderer::dot(geometric_normal, *shading_normal) > renderer::TransportScalar{0})) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A scene shading normal lies outside the geometric-normal hemisphere."));
    }
    return *shading_normal;
}

struct SurfaceDerivatives final {
    renderer::Vector3 dpdu;
    renderer::Vector3 dpdv;
};

struct WideBarycentrics final {
    double vertex0{};
    double vertex1{};
    double vertex2{};
};

[[nodiscard]] std::array<double, 2> projected_point(const renderer::Point3 point,
                                                    const std::uint32_t dropped_axis) noexcept {
    switch (dropped_axis) {
    case 0U:
        return {static_cast<double>(point.y), static_cast<double>(point.z)};
    case 1U:
        return {static_cast<double>(point.x), static_cast<double>(point.z)};
    default:
        return {static_cast<double>(point.x), static_cast<double>(point.y)};
    }
}

[[nodiscard]] core::Result<WideBarycentrics>
extrapolated_barycentrics(const std::array<renderer::Point3, 3>& vertices,
                          const renderer::Point3 point, const renderer::Normal3 geometric_normal) {
    const auto absolute_normal = std::array{
        std::abs(geometric_normal.x), std::abs(geometric_normal.y), std::abs(geometric_normal.z)};
    const auto dropped_axis = absolute_normal[0] > absolute_normal[1]
                                  ? (absolute_normal[0] > absolute_normal[2] ? 0U : 2U)
                                  : (absolute_normal[1] > absolute_normal[2] ? 1U : 2U);
    const auto vertex0 = projected_point(vertices[0], dropped_axis);
    const auto vertex1 = projected_point(vertices[1], dropped_axis);
    const auto vertex2 = projected_point(vertices[2], dropped_axis);
    const auto sample = projected_point(point, dropped_axis);
    const auto edge1 = std::array{vertex1[0] - vertex0[0], vertex1[1] - vertex0[1]};
    const auto edge2 = std::array{vertex2[0] - vertex0[0], vertex2[1] - vertex0[1]};
    const auto offset = std::array{sample[0] - vertex0[0], sample[1] - vertex0[1]};
    const auto determinant = std::fma(edge1[0], edge2[1], -edge1[1] * edge2[0]);
    if (!std::isfinite(determinant) || determinant == 0.0) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A scene ray differential triangle projection is not representable."));
    }
    const auto vertex1_weight = std::fma(offset[0], edge2[1], -offset[1] * edge2[0]) / determinant;
    const auto vertex2_weight = std::fma(edge1[0], offset[1], -edge1[1] * offset[0]) / determinant;
    const auto vertex0_weight = 1.0 - vertex1_weight - vertex2_weight;
    if (!std::isfinite(vertex0_weight) || !std::isfinite(vertex1_weight) ||
        !std::isfinite(vertex2_weight)) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A scene ray differential barycentric coordinate is not representable."));
    }
    return WideBarycentrics{
        .vertex0 = vertex0_weight,
        .vertex1 = vertex1_weight,
        .vertex2 = vertex2_weight,
    };
}

[[nodiscard]] core::Result<renderer::Normal3> interpolate_differential_shading_normal(
    const std::array<renderer::Normal3, 3>& normals, const WideBarycentrics barycentrics,
    const renderer::AffineTransform& object_to_world, const renderer::Normal3 geometric_normal) {
    const auto object_normal = std::array{
        std::fma(barycentrics.vertex0, static_cast<double>(normals[0].x),
                 std::fma(barycentrics.vertex1, static_cast<double>(normals[1].x),
                          barycentrics.vertex2 * static_cast<double>(normals[2].x))),
        std::fma(barycentrics.vertex0, static_cast<double>(normals[0].y),
                 std::fma(barycentrics.vertex1, static_cast<double>(normals[1].y),
                          barycentrics.vertex2 * static_cast<double>(normals[2].y))),
        std::fma(barycentrics.vertex0, static_cast<double>(normals[0].z),
                 std::fma(barycentrics.vertex1, static_cast<double>(normals[1].z),
                          barycentrics.vertex2 * static_cast<double>(normals[2].z))),
    };
    auto shading_normal = robust_transformed_normal(object_to_world, object_normal);
    if (!shading_normal) {
        return std::unexpected(shading_normal.error());
    }
    const auto orientation = transform_orientation(object_to_world);
    if (!orientation) {
        return std::unexpected(orientation.error());
    }
    if (*orientation < 0) {
        *shading_normal = -*shading_normal;
    }
    if (!(renderer::dot(geometric_normal, *shading_normal) > renderer::TransportScalar{0})) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A differential shading normal lies outside the geometric hemisphere."));
    }
    return *shading_normal;
}

[[nodiscard]] std::array<double, 2>
texture_coordinate_delta(const std::array<renderer::Point2, 3>& coordinates,
                         const WideBarycentrics auxiliary,
                         const WideBarycentrics central) noexcept {
    const auto delta0 = auxiliary.vertex0 - central.vertex0;
    const auto delta1 = auxiliary.vertex1 - central.vertex1;
    const auto delta2 = auxiliary.vertex2 - central.vertex2;
    return {
        std::fma(delta0, static_cast<double>(coordinates[0].x),
                 std::fma(delta1, static_cast<double>(coordinates[1].x),
                          delta2 * static_cast<double>(coordinates[2].x))),
        std::fma(delta0, static_cast<double>(coordinates[0].y),
                 std::fma(delta1, static_cast<double>(coordinates[1].y),
                          delta2 * static_cast<double>(coordinates[2].y))),
    };
}

[[nodiscard]] core::Result<renderer::TransportScalar>
narrow_texture_derivative(const double value) {
    const auto narrowed = static_cast<renderer::TransportScalar>(value);
    if (!std::isfinite(value) || !std::isfinite(narrowed) || (value != 0.0 && narrowed == 0.0F)) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A scene texture-coordinate differential is not representable."));
    }
    return narrowed;
}

[[nodiscard]] core::Result<renderer::TextureCoordinateDifferentials>
resolve_cone_texture_differentials(const ResolvedSceneSurface& surface, const AccelHit& hit,
                                   const renderer::Ray& ray, const renderer::RayCone cone) {
    const auto radius = renderer::ray_cone_surface_footprint_radius(
        cone, ray, hit.triangle.parameter, surface.interaction.geometric_normal());
    if (!radius) {
        return std::unexpected(radius.error());
    }
    if (!(*radius > renderer::TransportScalar{0})) {
        return std::unexpected(
            surface_error(core::StatusCode::unavailable,
                          "A mapped frame scene surface requires a non-zero ray-cone footprint."));
    }

    const auto basis = renderer::OrthonormalFrame::from_normal_and_tangent(
        surface.interaction.geometric_normal(), surface.interaction.dpdu());
    if (!basis) {
        return std::unexpected(surface_error(
            core::StatusCode::invalid_argument,
            "A ray cone cannot resolve a surface map through a singular surface tangent."));
    }

    const auto& dpdu = surface.interaction.dpdu();
    const auto& dpdv = surface.interaction.dpdv();
    const auto a = static_cast<double>(renderer::dot(basis->tangent(), dpdu));
    const auto b = static_cast<double>(renderer::dot(basis->tangent(), dpdv));
    const auto c = static_cast<double>(renderer::dot(basis->bitangent(), dpdu));
    const auto d = static_cast<double>(renderer::dot(basis->bitangent(), dpdv));
    const auto determinant = std::fma(a, d, -b * c);
    if (!std::isfinite(determinant) || determinant == 0.0) {
        return std::unexpected(surface_error(
            core::StatusCode::invalid_argument,
            "A ray cone cannot resolve a surface map through a singular UV Jacobian."));
    }
    const auto footprint_radius = static_cast<double>(*radius);
    const auto dudx = narrow_texture_derivative(footprint_radius * d / determinant);
    const auto dvdx = narrow_texture_derivative(-footprint_radius * c / determinant);
    const auto dudy = narrow_texture_derivative(-footprint_radius * b / determinant);
    const auto dvdy = narrow_texture_derivative(footprint_radius * a / determinant);
    for (const auto* derivative : {&dudx, &dvdx, &dudy, &dvdy}) {
        if (!derivative->has_value()) {
            return std::unexpected(derivative->error());
        }
    }
    const auto result = renderer::TextureCoordinateDifferentials{
        .dudx = *dudx,
        .dvdx = *dvdx,
        .dudy = *dudy,
        .dvdy = *dvdy,
    };
    const auto footprint_determinant =
        std::fma(result.dudx, result.dvdy, -result.dvdx * result.dudy);
    if (!std::isfinite(footprint_determinant) || footprint_determinant == 0.0F) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "A ray-cone texture footprint is not full-rank in transport precision."));
    }
    return result;
}

[[nodiscard]] core::Result<renderer::SurfaceInteraction>
with_shading_normal(const renderer::SurfaceInteraction& interaction,
                    const renderer::Normal3 shading_normal) {
    return renderer::SurfaceInteraction::create(
        interaction.position(), interaction.geometric_normal(), shading_normal, interaction.uv(),
        interaction.dpdu(), interaction.dpdv(), interaction.identifiers(), interaction.time());
}

[[nodiscard]] core::Result<renderer::OrthonormalFrame>
closure_frame_for_interaction(const SceneSpectralMaterial& material,
                              const renderer::SurfaceInteraction& interaction) {
    const auto& mixture = material.closure_mixture;
    auto frame = mixture.frame_mode == SceneClosureFrameMode::surface_tangent
                     ? renderer::OrthonormalFrame::from_normal_and_tangent(
                           interaction.shading_normal(), interaction.dpdu())
                     : renderer::OrthonormalFrame::from_normal(interaction.shading_normal());
    if (!frame) {
        return std::unexpected(frame.error());
    }
    return frame->rotated_about_normal(mixture.tangent_rotation_radians);
}

[[nodiscard]] core::Result<renderer::SurfaceInteraction>
evaluate_surface_maps(const FrameScene& scene, const SceneSpectralMaterial& material,
                      renderer::SurfaceInteraction interaction,
                      const renderer::TextureCoordinateDifferentials differentials) {
    if (!has_surface_map(material)) {
        return interaction;
    }
    if (material.normal_map) {
        const auto& binding = *material.normal_map;
        const auto texture = scene.host_image_texture(binding.texture);
        if (!texture || !texture->get().image) {
            return std::unexpected(
                surface_error(core::StatusCode::internal_error,
                              "A validated normal map lost its immutable host-image texture."));
        }
        const auto mapped = renderer::evaluate_host_tangent_space_normal_map(
            *texture->get().image, interaction, differentials,
            renderer::HostNormalMapOptions{
                .channels =
                    {
                        .x = binding.red_channel,
                        .y = binding.green_channel,
                        .z = binding.blue_channel,
                    },
                .y_convention = binding.y_convention,
                .u_wrap = binding.u_wrap,
                .v_wrap = binding.v_wrap,
                .ewa_limits = binding.ewa_limits,
            });
        if (!mapped) {
            return std::unexpected(mapped.error());
        }
        auto replaced = with_shading_normal(interaction, *mapped);
        if (!replaced) {
            return std::unexpected(replaced.error());
        }
        interaction = *replaced;
    }
    if (material.bump_map) {
        const auto& binding = *material.bump_map;
        const auto texture = scene.host_image_texture(binding.texture);
        if (!texture || !texture->get().image) {
            return std::unexpected(
                surface_error(core::StatusCode::internal_error,
                              "A validated bump map lost its immutable host-image texture."));
        }
        const auto mapped = renderer::evaluate_host_filtered_bump_map(
            *texture->get().image, interaction, differentials,
            renderer::HostBumpMapOptions{
                .height_channel = binding.channel,
                .u_wrap = binding.u_wrap,
                .v_wrap = binding.v_wrap,
                .ewa_limits = binding.ewa_limits,
                .displacement_scale = binding.scale,
            });
        if (!mapped) {
            return std::unexpected(mapped.error());
        }
        auto replaced = with_shading_normal(interaction, *mapped);
        if (!replaced) {
            return std::unexpected(replaced.error());
        }
        interaction = *replaced;
    }
    return interaction;
}

[[nodiscard]] core::Status
apply_surface_maps(const FrameScene& scene, const AccelHit& hit,
                   const renderer::TextureCoordinateDifferentials differentials,
                   ResolvedSceneSurface& surface) {
    const auto material = surface_material(scene, hit);
    if (!material) {
        return std::unexpected(material.error());
    }
    if (!has_surface_map(material->get())) {
        return {};
    }
    const auto mapped =
        evaluate_surface_maps(scene, material->get(), surface.interaction, differentials);
    if (!mapped) {
        return std::unexpected(mapped.error());
    }
    const auto frame = closure_frame_for_interaction(material->get(), *mapped);
    if (!frame) {
        return std::unexpected(frame.error());
    }
    surface.interaction = *mapped;
    surface.closure_frame = *frame;
    return {};
}

[[nodiscard]] core::Result<renderer::SurfaceInteraction>
auxiliary_interaction(const renderer::SurfaceInteraction& central,
                      const renderer::Vector3 position_delta, const renderer::TransportScalar du,
                      const renderer::TransportScalar dv, const renderer::Normal3 shading_normal) {
    return renderer::SurfaceInteraction::create(
        central.position() + position_delta, central.geometric_normal(), shading_normal,
        renderer::Point2{.x = central.uv().x + du, .y = central.uv().y + dv}, central.dpdu(),
        central.dpdv(), central.identifiers(), central.time());
}

[[nodiscard]] core::Status
apply_surface_maps_to_differentials(const FrameScene& scene, const AccelHit& hit,
                                    ResolvedSceneSurfaceWithDifferentials& resolved) {
    const auto material = surface_material(scene, hit);
    if (!material) {
        return std::unexpected(material.error());
    }
    if (!has_surface_map(material->get())) {
        return {};
    }

    const auto central = resolved.surface.interaction;
    const auto& footprint = resolved.differentials.texture_coordinates;
    auto rx = auxiliary_interaction(central, resolved.differentials.positions.dpdx, footprint.dudx,
                                    footprint.dvdx, resolved.differentials.rx_shading_normal);
    auto ry = auxiliary_interaction(central, resolved.differentials.positions.dpdy, footprint.dudy,
                                    footprint.dvdy, resolved.differentials.ry_shading_normal);
    if (!rx) {
        return std::unexpected(rx.error());
    }
    if (!ry) {
        return std::unexpected(ry.error());
    }
    auto mapped_rx = evaluate_surface_maps(scene, material->get(), *rx, footprint);
    auto mapped_ry = evaluate_surface_maps(scene, material->get(), *ry, footprint);
    if (!mapped_rx) {
        return std::unexpected(mapped_rx.error());
    }
    if (!mapped_ry) {
        return std::unexpected(mapped_ry.error());
    }
    if (auto status = apply_surface_maps(scene, hit, footprint, resolved.surface); !status) {
        return status;
    }
    resolved.differentials.rx_shading_normal = mapped_rx->shading_normal();
    resolved.differentials.ry_shading_normal = mapped_ry->shading_normal();
    return {};
}

[[nodiscard]] core::Result<ResolvedSceneSurfaceDifferentials>
resolve_surface_differentials(const FrameScene& scene, const AccelHit& hit,
                              const renderer::RayDifferential& ray,
                              const ResolvedSceneSurface& surface) {
    const auto positions = renderer::surface_point_differentials(ray, surface.interaction);
    if (!positions) {
        return std::unexpected(positions.error());
    }
    const auto geometry = scene.geometry(hit.identifiers.geometry);
    const auto transform = scene.world_transform(hit.identifiers.instance);
    if (!geometry || !transform) {
        return std::unexpected(
            surface_error(core::StatusCode::internal_error,
                          "A validated differential hit could not resolve its scene geometry."));
    }
    const auto primitive_index = static_cast<std::size_t>(hit.identifiers.primitive.value);
    const auto mesh = geometry->get().mesh;
    if (!mesh || primitive_index >= mesh->triangles().size()) {
        return std::unexpected(
            surface_error(core::StatusCode::incompatible,
                          "A differential hit references an unknown scene triangle."));
    }
    const auto& indices = mesh->triangles()[primitive_index].vertices;
    const auto object_positions =
        std::array{mesh->positions()[indices[0]], mesh->positions()[indices[1]],
                   mesh->positions()[indices[2]]};
    const auto world_positions = std::array{transform->get().apply(object_positions[0]),
                                            transform->get().apply(object_positions[1]),
                                            transform->get().apply(object_positions[2])};
    const auto object_normals = std::array{mesh->normals()[indices[0]], mesh->normals()[indices[1]],
                                           mesh->normals()[indices[2]]};
    const auto texture_coordinates =
        std::array{mesh->texture_coordinates()[indices[0]], mesh->texture_coordinates()[indices[1]],
                   mesh->texture_coordinates()[indices[2]]};
    const auto rx_position = surface.interaction.position() + positions->dpdx;
    const auto ry_position = surface.interaction.position() + positions->dpdy;
    const auto rx_barycentrics = extrapolated_barycentrics(world_positions, rx_position,
                                                           surface.interaction.geometric_normal());
    const auto ry_barycentrics = extrapolated_barycentrics(world_positions, ry_position,
                                                           surface.interaction.geometric_normal());
    const auto central_barycentrics = extrapolated_barycentrics(
        world_positions, surface.interaction.position(), surface.interaction.geometric_normal());
    if (!rx_barycentrics) {
        return std::unexpected(rx_barycentrics.error());
    }
    if (!ry_barycentrics) {
        return std::unexpected(ry_barycentrics.error());
    }
    if (!central_barycentrics) {
        return std::unexpected(central_barycentrics.error());
    }
    const auto rx_uv_delta =
        texture_coordinate_delta(texture_coordinates, *rx_barycentrics, *central_barycentrics);
    const auto ry_uv_delta =
        texture_coordinate_delta(texture_coordinates, *ry_barycentrics, *central_barycentrics);
    const auto dudx = narrow_texture_derivative(rx_uv_delta[0]);
    const auto dvdx = narrow_texture_derivative(rx_uv_delta[1]);
    const auto dudy = narrow_texture_derivative(ry_uv_delta[0]);
    const auto dvdy = narrow_texture_derivative(ry_uv_delta[1]);
    for (const auto* derivative : {&dudx, &dvdx, &dudy, &dvdy}) {
        if (!derivative->has_value()) {
            return std::unexpected(derivative->error());
        }
    }
    const auto rx_normal = interpolate_differential_shading_normal(
        object_normals, *rx_barycentrics, transform->get(), surface.interaction.geometric_normal());
    const auto ry_normal = interpolate_differential_shading_normal(
        object_normals, *ry_barycentrics, transform->get(), surface.interaction.geometric_normal());
    if (!rx_normal) {
        return std::unexpected(rx_normal.error());
    }
    if (!ry_normal) {
        return std::unexpected(ry_normal.error());
    }
    const auto texture_differentials = renderer::TextureCoordinateDifferentials{
        .dudx = *dudx,
        .dvdx = *dvdx,
        .dudy = *dudy,
        .dvdy = *dvdy,
    };
    return ResolvedSceneSurfaceDifferentials{
        .positions = *positions,
        .texture_coordinates = texture_differentials,
        .rx_shading_normal = *rx_normal,
        .ry_shading_normal = *ry_normal,
    };
}

[[nodiscard]] core::Result<SurfaceDerivatives>
surface_derivatives(const std::array<renderer::Point3, 3>& positions,
                    const std::array<renderer::Point2, 3>& coordinates) {
    const auto edge1 = positions[1] - positions[0];
    const auto edge2 = positions[2] - positions[0];
    const auto du1 = coordinates[1].x - coordinates[0].x;
    const auto dv1 = coordinates[1].y - coordinates[0].y;
    const auto du2 = coordinates[2].x - coordinates[0].x;
    const auto dv2 = coordinates[2].y - coordinates[0].y;
    const auto determinant = std::fma(du1, dv2, -dv1 * du2);
    if (!std::isfinite(determinant)) {
        return std::unexpected(surface_error(core::StatusCode::invalid_argument,
                                             "A scene triangle UV Jacobian is not representable."));
    }
    if (determinant == renderer::TransportScalar{0}) {
        return SurfaceDerivatives{};
    }

    const auto reciprocal = renderer::TransportScalar{1} / determinant;
    const auto dpdu = (edge1 * dv2 - edge2 * dv1) * reciprocal;
    const auto dpdv = (edge2 * du1 - edge1 * du2) * reciprocal;
    const auto finite = [](const renderer::Vector3 value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };
    if (!std::isfinite(reciprocal) || !finite(dpdu) || !finite(dpdv)) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "Scene triangle surface derivatives are not representable."));
    }
    return SurfaceDerivatives{.dpdu = dpdu, .dpdv = dpdv};
}

core::Result<ResolvedSceneSurface> resolve_scene_surface_hit_impl(const FrameScene& scene,
                                                                  const AccelHit& hit,
                                                                  const renderer::Ray& ray) {
    if (!std::isfinite(hit.triangle.parameter) || !ray.contains_parameter(hit.triangle.parameter)) {
        return std::unexpected(
            surface_error(core::StatusCode::invalid_argument,
                          "An acceleration hit parameter lies outside its source ray."));
    }
    const auto ray_position = ray.at(hit.triangle.parameter);
    if (!ray_position || *ray_position != hit.triangle.position) {
        return std::unexpected(
            surface_error(core::StatusCode::incompatible,
                          "An acceleration hit position does not match its source ray parameter."));
    }
    if (auto status = validate_barycentrics(hit.triangle.barycentrics); !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = validate_hit_identity(scene, hit); !status) {
        return std::unexpected(std::move(status.error()));
    }

    const auto instance = scene.instance(hit.identifiers.instance);
    const auto geometry = scene.geometry(hit.identifiers.geometry);
    const auto material = scene.material(hit.identifiers.material);
    const auto transform = scene.world_transform(hit.identifiers.instance);
    if (!instance || !geometry || !material || !transform) {
        return std::unexpected(surface_error(
            core::StatusCode::internal_error,
            "A validated acceleration hit could not resolve its frame scene records."));
    }
    if (!material->get().spectral) {
        return std::unexpected(
            surface_error(core::StatusCode::unavailable,
                          "The hit frame scene material has no spectral transport data."));
    }

    const auto primitive_index = static_cast<std::size_t>(hit.identifiers.primitive.value);
    const auto mesh = geometry->get().mesh;
    if (!mesh || primitive_index >= mesh->triangles().size()) {
        return std::unexpected(
            surface_error(core::StatusCode::incompatible,
                          "An acceleration hit references an unknown scene triangle."));
    }
    const auto& indices = mesh->triangles()[primitive_index].vertices;
    const auto object_positions =
        std::array{mesh->positions()[indices[0]], mesh->positions()[indices[1]],
                   mesh->positions()[indices[2]]};
    const auto object_normals = std::array{mesh->normals()[indices[0]], mesh->normals()[indices[1]],
                                           mesh->normals()[indices[2]]};
    const auto texture_coordinates =
        std::array{mesh->texture_coordinates()[indices[0]], mesh->texture_coordinates()[indices[1]],
                   mesh->texture_coordinates()[indices[2]]};
    const auto& object_to_world = transform->get();
    const auto world_positions = std::array{object_to_world.apply(object_positions[0]),
                                            object_to_world.apply(object_positions[1]),
                                            object_to_world.apply(object_positions[2])};

    const auto world_triangle =
        renderer::Triangle::create(world_positions[0], world_positions[1], world_positions[2]);
    if (!world_triangle) {
        return std::unexpected(
            surface_error(core::StatusCode::internal_error,
                          "A committed scene triangle could not be reconstructed in world space."));
    }
    constexpr auto normal_tolerance =
        renderer::TransportScalar{128} * std::numeric_limits<renderer::TransportScalar>::epsilon();
    const auto normal_alignment =
        renderer::dot(world_triangle->geometric_normal(), hit.triangle.geometric_normal);
    if (!std::isfinite(normal_alignment) ||
        normal_alignment < renderer::TransportScalar{1} - normal_tolerance) {
        return std::unexpected(surface_error(
            core::StatusCode::incompatible,
            "An acceleration hit geometric normal disagrees with the committed triangle."));
    }

    const auto object_position =
        interpolate_position_with_error(object_positions, hit.triangle.barycentrics);
    if (!object_position) {
        return std::unexpected(object_position.error());
    }
    const auto position =
        transform_position_with_error(object_to_world, *object_position, hit.triangle.position);
    const auto shading_normal = interpolate_shading_normal(
        object_normals, hit.triangle.barycentrics, object_to_world, hit.triangle.geometric_normal);
    const auto derivatives = surface_derivatives(world_positions, texture_coordinates);
    if (!position || !shading_normal || !derivatives) {
        if (!position) {
            return std::unexpected(position.error());
        }
        if (!shading_normal) {
            return std::unexpected(shading_normal.error());
        }
        return std::unexpected(derivatives.error());
    }

    const auto interaction = renderer::SurfaceInteraction::create(
        position->point, hit.triangle.geometric_normal, *shading_normal,
        interpolate_texture_coordinate(texture_coordinates, hit.triangle.barycentrics),
        derivatives->dpdu, derivatives->dpdv, hit.identifiers, ray.time());
    if (!interaction) {
        return std::unexpected(interaction.error());
    }
    const auto& scene_closures = material->get().spectral->closure_mixture;
    const auto closures = renderer::ClosureMixture::create(
        scene_closures.closures,
        std::span<const renderer::TransportScalar>{scene_closures.component_probabilities.data(),
                                                   scene_closures.closures.size()});
    auto closure_frame = scene_closures.frame_mode == SceneClosureFrameMode::surface_tangent
                             ? renderer::OrthonormalFrame::from_normal_and_tangent(
                                   *shading_normal, derivatives->dpdu)
                             : renderer::OrthonormalFrame::from_normal(*shading_normal);
    if (closure_frame) {
        closure_frame =
            closure_frame->rotated_about_normal(scene_closures.tangent_rotation_radians);
    }
    const auto emission =
        renderer::OneSidedSurfaceEmission::create(material->get().spectral->emitted_radiance);
    if (!closures || !closure_frame || !emission) {
        return std::unexpected(surface_error(
            core::StatusCode::internal_error,
            "A validated frame scene material could not recreate its spectral closures."));
    }

    return ResolvedSceneSurface{
        .interaction = *interaction,
        .position_error = position->absolute_error,
        .closures = *closures,
        .closure_frame = *closure_frame,
        .emission = *emission,
        .ray_parameter = hit.triangle.parameter,
    };
}

} // namespace

core::Result<ResolvedSceneSurface>
resolve_scene_surface_hit(const FrameScene& scene, const AccelHit& hit, const renderer::Ray& ray) {
    auto surface = resolve_scene_surface_hit_impl(scene, hit, ray);
    if (!surface) {
        return std::unexpected(surface.error());
    }
    const auto material = surface_material(scene, hit);
    if (!material) {
        return std::unexpected(material.error());
    }
    if (has_surface_map(material->get())) {
        return std::unexpected(surface_error(
            core::StatusCode::unavailable,
            "A mapped frame scene surface requires explicit ray differentials or a ray cone."));
    }
    return surface;
}

core::Result<ResolvedSceneSurfaceWithDifferentials>
resolve_scene_surface_hit(const FrameScene& scene, const AccelHit& hit,
                          const renderer::RayDifferential& ray) {
    auto surface = resolve_scene_surface_hit_impl(scene, hit, ray.ray());
    if (!surface) {
        return std::unexpected(surface.error());
    }
    auto differentials = resolve_surface_differentials(scene, hit, ray, *surface);
    if (!differentials) {
        return std::unexpected(differentials.error());
    }
    auto resolved = ResolvedSceneSurfaceWithDifferentials{
        .surface = std::move(*surface),
        .differentials = *differentials,
    };
    if (auto status = apply_surface_maps_to_differentials(scene, hit, resolved); !status) {
        return std::unexpected(status.error());
    }
    return resolved;
}

core::Result<ResolvedSceneSurface> resolve_scene_surface_hit(const FrameScene& scene,
                                                             const AccelHit& hit,
                                                             const renderer::Ray& ray,
                                                             const renderer::RayCone cone) {
    auto surface = resolve_scene_surface_hit_impl(scene, hit, ray);
    if (!surface) {
        return std::unexpected(surface.error());
    }
    const auto material = surface_material(scene, hit);
    if (!material) {
        return std::unexpected(material.error());
    }
    if (!has_surface_map(material->get())) {
        return surface;
    }
    const auto differentials = resolve_cone_texture_differentials(*surface, hit, ray, cone);
    if (!differentials) {
        return std::unexpected(differentials.error());
    }
    if (auto status = apply_surface_maps(scene, hit, *differentials, *surface); !status) {
        return std::unexpected(status.error());
    }
    return surface;
}

core::Result<std::optional<ResolvedSceneSurface>>
resolve_scene_surface(const AccelBackend& acceleration, const renderer::Ray& ray) {
    const auto scene = acceleration.frame_scene();
    if (!scene) {
        return std::unexpected(
            surface_error(core::StatusCode::internal_error,
                          "The acceleration backend has no committed frame scene."));
    }
    const auto hit = acceleration.closest_hit(ray);
    if (!hit) {
        return std::unexpected(hit.error());
    }
    if (!*hit) {
        return std::optional<ResolvedSceneSurface>{};
    }
    auto resolved = resolve_scene_surface_hit(*scene, **hit, ray);
    if (!resolved) {
        return std::unexpected(resolved.error());
    }
    return std::optional<ResolvedSceneSurface>{std::move(*resolved)};
}

core::Result<std::optional<ResolvedSceneSurfaceWithDifferentials>>
resolve_scene_surface(const AccelBackend& acceleration, const renderer::RayDifferential& ray) {
    const auto scene = acceleration.frame_scene();
    if (!scene) {
        return std::unexpected(
            surface_error(core::StatusCode::internal_error,
                          "The acceleration backend has no committed frame scene."));
    }
    const auto hit = acceleration.closest_hit(ray.ray());
    if (!hit) {
        return std::unexpected(hit.error());
    }
    if (!*hit) {
        return std::optional<ResolvedSceneSurfaceWithDifferentials>{};
    }
    auto resolved = resolve_scene_surface_hit(*scene, **hit, ray);
    if (!resolved) {
        return std::unexpected(resolved.error());
    }
    return std::optional<ResolvedSceneSurfaceWithDifferentials>{std::move(*resolved)};
}

core::Result<std::optional<ResolvedSceneSurface>>
resolve_scene_surface(const AccelBackend& acceleration, const renderer::Ray& ray,
                      const renderer::RayCone cone) {
    const auto scene = acceleration.frame_scene();
    if (!scene) {
        return std::unexpected(
            surface_error(core::StatusCode::internal_error,
                          "The acceleration backend has no committed frame scene."));
    }
    const auto hit = acceleration.closest_hit(ray);
    if (!hit) {
        return std::unexpected(hit.error());
    }
    if (!*hit) {
        return std::optional<ResolvedSceneSurface>{};
    }
    auto resolved = resolve_scene_surface_hit(*scene, **hit, ray, cone);
    if (!resolved) {
        return std::unexpected(resolved.error());
    }
    return std::optional<ResolvedSceneSurface>{std::move(*resolved)};
}

} // namespace blackframe::engine
