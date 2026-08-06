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
                          const renderer::Normal3 normal) {
    const auto& inverse = transform.inverse_matrix();
    const auto transformed = std::array{
        std::fma(static_cast<double>(inverse(0, 0)), static_cast<double>(normal.x),
                 std::fma(static_cast<double>(inverse(1, 0)), static_cast<double>(normal.y),
                          static_cast<double>(inverse(2, 0)) * static_cast<double>(normal.z))),
        std::fma(static_cast<double>(inverse(0, 1)), static_cast<double>(normal.x),
                 std::fma(static_cast<double>(inverse(1, 1)), static_cast<double>(normal.y),
                          static_cast<double>(inverse(2, 1)) * static_cast<double>(normal.z))),
        std::fma(static_cast<double>(inverse(0, 2)), static_cast<double>(normal.x),
                 std::fma(static_cast<double>(inverse(1, 2)), static_cast<double>(normal.y),
                          static_cast<double>(inverse(2, 2)) * static_cast<double>(normal.z))),
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
    };
}

} // namespace

core::Result<ResolvedSceneSurface>
resolve_scene_surface_hit(const FrameScene& scene, const AccelHit& hit, const renderer::Ray& ray) {
    return resolve_scene_surface_hit_impl(scene, hit, ray);
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

} // namespace blackframe::engine
