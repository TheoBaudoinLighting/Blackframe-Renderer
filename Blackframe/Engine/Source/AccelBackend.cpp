#include <Blackframe/Engine/AccelBackend.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error accel_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

class AnalyticAccelBackend final : public AccelBackend {
  public:
    [[nodiscard]] static core::Result<std::unique_ptr<AccelBackend>>
    create(FrameSceneHandle scene) {
        auto instances = prepare_instances(scene);
        if (!instances) {
            return std::unexpected(std::move(instances.error()));
        }

        try {
            return std::unique_ptr<AccelBackend>{
                new AnalyticAccelBackend{std::move(scene), std::move(*instances)}};
        } catch (const std::bad_alloc&) {
            return std::unexpected(
                accel_error(core::StatusCode::resource_exhausted,
                            "Analytic acceleration construction exhausted host memory."));
        }
    }

    [[nodiscard]] AccelBackendKind kind() const noexcept override {
        return AccelBackendKind::analytic_reference;
    }

    [[nodiscard]] core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray& ray) const override {
        const auto validation = validate_ray(ray);
        if (!validation) {
            return std::unexpected(validation.error());
        }

        auto closest = std::optional<AccelHit>{};
        for (const auto& instance : instances_) {
            if ((ray.mask() & instance.visibility_mask) == 0U) {
                continue;
            }

            auto local_ray = object_space_ray(ray, instance);
            if (!local_ray) {
                return std::unexpected(std::move(local_ray.error()));
            }
            for (auto triangle_index = std::size_t{};
                 triangle_index < instance.mesh->triangles().size(); ++triangle_index) {
                auto triangle = instance.mesh->geometric_triangle(triangle_index);
                if (!triangle) {
                    return std::unexpected(std::move(triangle.error()));
                }
                auto candidate = triangle->intersect_classified(*local_ray);
                if (!candidate) {
                    if (candidate.error().kind ==
                        renderer::TriangleIntersectionErrorKind::coplanar_ambiguity) {
                        continue;
                    }
                    return std::unexpected(std::move(candidate.error().diagnostic));
                }
                if (!candidate->has_value()) {
                    continue;
                }
                if (closest.has_value() &&
                    closest->triangle.parameter <= candidate->value().parameter) {
                    continue;
                }

                auto world_triangle = world_space_triangle(instance, triangle_index);
                if (!world_triangle) {
                    return std::unexpected(std::move(world_triangle.error()));
                }
                auto position = ray.at(candidate->value().parameter);
                if (!position) {
                    return std::unexpected(std::move(position.error()));
                }

                closest = AccelHit{
                    .object = instance.object,
                    .triangle =
                        {
                            .parameter = candidate->value().parameter,
                            .position = *position,
                            .geometric_normal = world_triangle->geometric_normal(),
                            .barycentrics = candidate->value().barycentrics,
                        },
                    .identifiers =
                        {
                            .instance = instance.instance,
                            .geometry = instance.geometry,
                            .primitive =
                                renderer::PrimitiveId{
                                    .value = static_cast<std::uint32_t>(triangle_index),
                                },
                            .material = instance.material,
                        },
                };
            }
        }
        return closest;
    }

    [[nodiscard]] core::Result<bool> occluded(const renderer::Ray& ray) const override {
        const auto validation = validate_ray(ray);
        if (!validation) {
            return std::unexpected(validation.error());
        }

        for (const auto& instance : instances_) {
            if ((ray.mask() & instance.visibility_mask) == 0U) {
                continue;
            }

            auto local_ray = object_space_ray(ray, instance);
            if (!local_ray) {
                return std::unexpected(std::move(local_ray.error()));
            }
            for (auto triangle_index = std::size_t{};
                 triangle_index < instance.mesh->triangles().size(); ++triangle_index) {
                auto triangle = instance.mesh->geometric_triangle(triangle_index);
                if (!triangle) {
                    return std::unexpected(std::move(triangle.error()));
                }
                auto candidate = triangle->intersect_classified(*local_ray);
                if (!candidate) {
                    if (candidate.error().kind ==
                        renderer::TriangleIntersectionErrorKind::coplanar_ambiguity) {
                        continue;
                    }
                    return std::unexpected(std::move(candidate.error().diagnostic));
                }
                if (candidate->has_value()) {
                    return true;
                }
            }
        }
        return false;
    }

    [[nodiscard]] core::Status rebuild(FrameSceneHandle scene) override {
        auto instances = prepare_instances(scene);
        if (!instances) {
            return std::unexpected(std::move(instances.error()));
        }

        instances_ = std::move(*instances);
        publish_rebuild(std::move(scene));
        return {};
    }

    [[nodiscard]] core::Status refit(FrameSceneHandle scene) override {
        const auto compatibility = validate_refit_scene(scene);
        if (!compatibility) {
            return std::unexpected(compatibility.error());
        }

        auto instances = prepare_instances(scene);
        if (!instances) {
            return std::unexpected(std::move(instances.error()));
        }

        instances_ = std::move(*instances);
        publish_refit(std::move(scene));
        return {};
    }

  private:
    AnalyticAccelBackend(FrameSceneHandle&& scene, std::vector<AccelInstance>&& instances) noexcept
        : AccelBackend{std::move(scene)}, instances_{std::move(instances)} {}

    std::vector<AccelInstance> instances_;
};

} // namespace

AccelBackend::AccelBackend(FrameSceneHandle scene) noexcept : scene_{std::move(scene)} {}

AccelBuildStatistics AccelBackend::build_statistics() const noexcept {
    return build_statistics_;
}

FrameSceneHandle AccelBackend::frame_scene() const noexcept {
    return scene_;
}

core::Result<std::vector<AccelInstance>>
AccelBackend::prepare_instances(const FrameSceneHandle& scene) {
    if (!scene) {
        return std::unexpected(
            accel_error(core::StatusCode::invalid_argument,
                        "Acceleration construction requires an immutable frame scene."));
    }

    try {
        auto instances = std::vector<AccelInstance>{};
        instances.reserve(scene->instances().size());
        for (const auto& source : scene->instances()) {
            auto geometry = scene->geometry(source.geometry);
            if (!geometry) {
                return std::unexpected(std::move(geometry.error()));
            }
            auto world_transform = scene->world_transform(source.id);
            if (!world_transform) {
                return std::unexpected(std::move(world_transform.error()));
            }
            if (!geometry->get().mesh) {
                return std::unexpected(
                    accel_error(core::StatusCode::internal_error,
                                "A validated frame scene lost an instance triangle mesh."));
            }

            auto instance = AccelInstance{
                .mesh = geometry->get().mesh,
                .object_to_world = world_transform->get(),
                .object = source.object,
                .instance = source.id,
                .geometry = source.geometry,
                .material = source.material,
                .visibility_mask = source.visibility_mask,
            };
            for (auto triangle_index = std::size_t{};
                 triangle_index < instance.mesh->triangles().size(); ++triangle_index) {
                if (!world_space_triangle(instance, triangle_index)) {
                    return std::unexpected(accel_error(
                        core::StatusCode::invalid_argument,
                        "A frame scene instance produces an unrepresentable world-space "
                        "triangle."));
                }
            }
            instances.push_back(std::move(instance));
        }
        return instances;
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            accel_error(core::StatusCode::resource_exhausted,
                        "Acceleration instance preparation exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            accel_error(core::StatusCode::resource_exhausted,
                        "Acceleration instance preparation exceeded host container limits."));
    }
}

core::Status AccelBackend::validate_refit_scene(const FrameSceneHandle& scene) const {
    if (!scene) {
        return std::unexpected(
            accel_error(core::StatusCode::invalid_argument,
                        "Acceleration refit requires an immutable frame scene."));
    }
    if (!scene_) {
        return std::unexpected(accel_error(core::StatusCode::internal_error,
                                           "Acceleration refit lost the committed frame scene."));
    }

    if (!std::ranges::equal(scene_->objects(), scene->objects())) {
        return std::unexpected(
            accel_error(core::StatusCode::incompatible,
                        "Acceleration refit cannot change the frame scene object identifiers."));
    }
    if (!std::ranges::equal(scene_->geometries(), scene->geometries(),
                            [](const SceneGeometry& left, const SceneGeometry& right) {
                                return left.id == right.id && left.mesh == right.mesh;
                            })) {
        return std::unexpected(
            accel_error(core::StatusCode::incompatible,
                        "Acceleration refit cannot change geometry identifiers or meshes."));
    }
    if (!std::ranges::equal(scene_->constant_textures(), scene->constant_textures()) ||
        !std::ranges::equal(scene_->host_image_textures(), scene->host_image_textures()) ||
        !std::ranges::equal(scene_->materials(), scene->materials()) ||
        !std::ranges::equal(scene_->punctual_lights(), scene->punctual_lights()) ||
        scene_->spectral_environment() != scene->spectral_environment()) {
        return std::unexpected(
            accel_error(core::StatusCode::incompatible,
                        "Acceleration refit cannot change frame scene texture, material, "
                        "punctual-light, or environment records."));
    }
    if (!std::ranges::equal(scene_->instances(), scene->instances(),
                            [](const SceneInstance& left, const SceneInstance& right) {
                                return left.id == right.id && left.parent == right.parent &&
                                       left.object == right.object &&
                                       left.geometry == right.geometry &&
                                       left.material == right.material &&
                                       left.visibility_mask == right.visibility_mask;
                            })) {
        return std::unexpected(
            accel_error(core::StatusCode::incompatible,
                        "Acceleration refit accepts only instance transform changes."));
    }
    return {};
}

void AccelBackend::publish_rebuild(FrameSceneHandle scene) noexcept {
    scene_ = std::move(scene);
    ++build_statistics_.commits;
    ++build_statistics_.rebuilds;
}

void AccelBackend::publish_refit(FrameSceneHandle scene) noexcept {
    scene_ = std::move(scene);
    ++build_statistics_.commits;
    ++build_statistics_.refits;
}

core::Status AccelBackend::validate_ray(const renderer::Ray& ray) {
    if (ray.time() < renderer::TransportScalar{0} || ray.time() > renderer::TransportScalar{1}) {
        return std::unexpected(
            accel_error(core::StatusCode::invalid_argument,
                        "Acceleration queries require ray time in the normalized [0, 1] range."));
    }
    return {};
}

core::Result<renderer::Ray> AccelBackend::object_space_ray(const renderer::Ray& world_ray,
                                                           const AccelInstance& instance) {
    return renderer::Ray::create(instance.object_to_world.apply_inverse(world_ray.origin()),
                                 instance.object_to_world.apply_inverse(world_ray.direction()),
                                 world_ray.t_min(), world_ray.t_max(), world_ray.time(),
                                 world_ray.mask(), world_ray.current_medium());
}

core::Result<renderer::Triangle>
AccelBackend::world_space_triangle(const AccelInstance& instance,
                                   const std::size_t triangle_index) {
    auto triangle = instance.mesh->geometric_triangle(triangle_index);
    if (!triangle) {
        return std::unexpected(std::move(triangle.error()));
    }
    const auto& vertices = triangle->vertices();
    return renderer::Triangle::create(instance.object_to_world.apply(vertices[0]),
                                      instance.object_to_world.apply(vertices[1]),
                                      instance.object_to_world.apply(vertices[2]));
}

core::Result<std::unique_ptr<AccelBackend>> create_analytic_accel_backend(FrameSceneHandle scene) {
    return AnalyticAccelBackend::create(std::move(scene));
}

} // namespace blackframe::engine
