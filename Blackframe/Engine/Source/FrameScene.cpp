#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Renderer/Emission.hpp>
#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/PunctualLights.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error scene_error(const core::StatusCode code, const std::string_view message) {
    return core::Error{
        .code = code,
        .message = std::string{message},
    };
}

template <typename Record> void sort_by_identifier(std::vector<Record>& records) {
    std::ranges::sort(records, {}, [](const Record& record) { return record.id.value; });
}

template <typename Record>
[[nodiscard]] core::Status reject_duplicate_identifiers(const std::vector<Record>& records,
                                                        const std::string_view message) {
    const auto duplicate = std::ranges::adjacent_find(
        records, [](const Record& left, const Record& right) { return left.id == right.id; });
    if (duplicate != records.end()) {
        return std::unexpected(scene_error(core::StatusCode::invalid_argument, message));
    }
    return {};
}

template <typename Record, typename Identifier>
[[nodiscard]] bool contains_identifier(const std::vector<Record>& records,
                                       const Identifier id) noexcept {
    const auto candidate = std::ranges::lower_bound(
        records, id.value, {}, [](const Record& record) { return record.id.value; });
    return candidate != records.end() && candidate->id == id;
}

[[nodiscard]] core::Status
validate_punctual_light(const ScenePunctualLight& record,
                        const renderer::SampledWavelengths& wavelengths) {
    if (record.valueless_by_exception()) {
        return std::unexpected(scene_error(core::StatusCode::invalid_argument,
                                           "A frame scene punctual-light slot has no value."));
    }

    return std::visit(
        [&](const auto& light) -> core::Status {
            using Light = std::remove_cvref_t<decltype(light)>;
            if constexpr (std::is_same_v<Light, ScenePointLight>) {
                auto validated =
                    renderer::PointLight::create(light.position, light.absolute_position_error,
                                                 wavelengths, light.spectral_radiant_intensity);
                if (!validated) {
                    return std::unexpected(std::move(validated.error()));
                }
            } else if constexpr (std::is_same_v<Light, SceneDirectionalLight>) {
                auto validated = renderer::DirectionalLight::create(
                    light.propagation_direction, wavelengths, light.spectral_irradiance);
                if (!validated) {
                    return std::unexpected(std::move(validated.error()));
                }
            } else if constexpr (std::is_same_v<Light, SceneSpotLight>) {
                auto validated = renderer::SpotLight::create(
                    light.position, light.absolute_position_error, light.emission_direction,
                    light.inner_half_angle_radians, light.outer_half_angle_radians, wavelengths,
                    light.on_axis_spectral_radiant_intensity);
                if (!validated) {
                    return std::unexpected(std::move(validated.error()));
                }
            }
            return {};
        },
        record);
}

template <typename Record, typename Identifier>
[[nodiscard]] std::optional<std::size_t> find_record_index(const std::vector<Record>& records,
                                                           const Identifier id) noexcept {
    const auto candidate = std::ranges::lower_bound(
        records, id.value, {}, [](const Record& record) { return record.id.value; });
    if (candidate == records.end() || candidate->id != id) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(candidate - records.begin());
}

template <typename Record, typename Identifier>
[[nodiscard]] core::Result<std::reference_wrapper<const Record>>
find_record(const std::vector<Record>& records, const Identifier id,
            const std::string_view missing_message) {
    const auto candidate = std::ranges::lower_bound(
        records, id.value, {}, [](const Record& record) { return record.id.value; });
    if (candidate == records.end() || candidate->id != id) {
        return std::unexpected(scene_error(core::StatusCode::not_found, missing_message));
    }
    return std::cref(*candidate);
}

enum class HierarchyState : std::uint8_t {
    unresolved,
    resolving,
    resolved,
};

struct ResolvedInstanceTransforms final {
    std::vector<renderer::AffineTransform> local;
    std::vector<renderer::AffineTransform> world;
};

struct TransformedMeshPositions final {
    std::vector<renderer::Point3> positions;
    renderer::Vector3 absolute_position_error;
};

struct DerivedMeshAreaLights final {
    std::vector<renderer::MeshAreaLight> models;
    std::vector<renderer::InstanceId> instance_ids;
};

[[nodiscard]] bool is_black(const renderer::TransportSpectrum& spectrum) noexcept {
    return std::ranges::all_of(spectrum.values,
                               [](const renderer::TransportScalar value) { return value == 0.0F; });
}

[[nodiscard]] core::Result<TransformedMeshPositions>
transform_mesh_positions(const TriangleMesh& mesh, const renderer::AffineTransform& transform) {
    auto transformed = TransformedMeshPositions{};
    transformed.positions.reserve(mesh.positions().size());

    constexpr auto scalar_epsilon =
        static_cast<double>(std::numeric_limits<renderer::TransportScalar>::epsilon());
    constexpr auto gamma7 = (7.0 * scalar_epsilon) / (1.0 - 7.0 * scalar_epsilon);
    constexpr auto underflow_allowance =
        7.0 * static_cast<double>(std::numeric_limits<renderer::TransportScalar>::denorm_min());
    constexpr auto maximum =
        static_cast<double>(std::numeric_limits<renderer::TransportScalar>::max());

    auto maximum_error = std::array<renderer::TransportScalar, 3>{};
    for (const auto source_position : mesh.positions()) {
        const auto world_position = transform.apply(source_position);
        const auto source = std::array{source_position.x, source_position.y, source_position.z};
        const auto world = std::array{world_position.x, world_position.y, world_position.z};

        for (auto row = std::size_t{}; row < 3U; ++row) {
            auto magnitude = std::abs(static_cast<double>(transform.matrix()(row, 3)));
            for (auto column = std::size_t{}; column < 3U; ++column) {
                magnitude += std::abs(static_cast<double>(transform.matrix()(row, column))) *
                             std::abs(static_cast<double>(source[column]));
            }
            const auto bound = std::fma(gamma7, magnitude, underflow_allowance);
            if (!std::isfinite(static_cast<double>(world[row])) || !std::isfinite(magnitude) ||
                !std::isfinite(bound) || bound < 0.0 || bound > maximum) {
                return std::unexpected(scene_error(
                    core::StatusCode::invalid_argument,
                    "An emissive mesh world position or its error bound is not representable."));
            }

            auto rounded_bound = static_cast<renderer::TransportScalar>(bound);
            if (static_cast<double>(rounded_bound) < bound) {
                rounded_bound = std::nextafter(
                    rounded_bound, std::numeric_limits<renderer::TransportScalar>::infinity());
            }
            if (!std::isfinite(rounded_bound) || rounded_bound < 0.0F ||
                (bound > 0.0 && rounded_bound == 0.0F)) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "An emissive mesh world-position error is not representable."));
            }
            maximum_error[row] = std::max(maximum_error[row], rounded_bound);
        }
        transformed.positions.push_back(world_position);
    }

    transformed.absolute_position_error = renderer::Vector3{
        .x = maximum_error[0],
        .y = maximum_error[1],
        .z = maximum_error[2],
    };
    return transformed;
}

[[nodiscard]] core::Result<DerivedMeshAreaLights>
derive_mesh_area_lights(const FrameSceneDescription& description,
                        const std::vector<renderer::AffineTransform>& world_transforms) {
    if (description.instances.size() != world_transforms.size()) {
        return std::unexpected(
            scene_error(core::StatusCode::internal_error,
                        "Frame scene area-light derivation lost the aligned world transforms."));
    }

    auto derived = DerivedMeshAreaLights{};
    derived.models.reserve(description.instances.size());
    derived.instance_ids.reserve(description.instances.size());

    for (auto instance_index = std::size_t{}; instance_index < description.instances.size();
         ++instance_index) {
        const auto& instance = description.instances[instance_index];
        const auto material_index = find_record_index(description.materials, instance.material);
        const auto geometry_index = find_record_index(description.geometries, instance.geometry);
        if (!material_index || !geometry_index) {
            return std::unexpected(scene_error(
                core::StatusCode::internal_error,
                "Frame scene area-light derivation lost validated instance references."));
        }

        const auto& material = description.materials[*material_index];
        if (!material.spectral || is_black(material.spectral->emitted_radiance)) {
            continue;
        }
        if (!description.spectral_environment) {
            return std::unexpected(
                scene_error(core::StatusCode::internal_error,
                            "An emissive mesh lost the frame scene spectral environment."));
        }
        if (derived.models.size() == std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(scene_error(
                core::StatusCode::resource_exhausted,
                "Frame scene mesh area lights exceed the stable 32-bit registry-slot domain."));
        }

        const auto& mesh = *description.geometries[*geometry_index].mesh;
        auto transformed = transform_mesh_positions(mesh, world_transforms[instance_index]);
        if (!transformed) {
            return std::unexpected(std::move(transformed.error()));
        }

        auto triangles = std::vector<renderer::AreaLightTriangleIndices>{};
        triangles.reserve(mesh.triangles().size());
        for (const auto& triangle : mesh.triangles()) {
            triangles.push_back(renderer::AreaLightTriangleIndices{
                .vertex0 = triangle.vertices[0],
                .vertex1 = triangle.vertices[1],
                .vertex2 = triangle.vertices[2],
            });
        }

        auto model = renderer::MeshAreaLight::create(
            std::move(transformed->positions), std::move(triangles),
            transformed->absolute_position_error, renderer::AreaLightSidedness::one_sided,
            description.spectral_environment->wavelengths, material.spectral->emitted_radiance);
        if (!model) {
            return std::unexpected(std::move(model.error()));
        }
        derived.models.push_back(std::move(*model));
        derived.instance_ids.push_back(instance.id);
    }
    return derived;
}

[[nodiscard]] core::Status validate_spectral_transport(const FrameSceneDescription& description) {
    if (description.punctual_lights.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(scene_error(
            core::StatusCode::resource_exhausted,
            "Frame scene punctual lights exceed the stable 32-bit registry-slot domain."));
    }
    if (!description.spectral_environment) {
        if (!description.punctual_lights.empty()) {
            return std::unexpected(scene_error(
                core::StatusCode::invalid_argument,
                "A frame scene cannot contain punctual lights without an explicit spectral "
                "environment."));
        }
        const auto material_with_transport =
            std::ranges::find_if(description.materials, [](const SceneMaterial& material) {
                return material.spectral.has_value();
            });
        if (material_with_transport != description.materials.end()) {
            return std::unexpected(scene_error(
                core::StatusCode::invalid_argument,
                "A frame scene cannot contain spectral materials without an explicit spectral "
                "environment."));
        }
        return {};
    }

    const auto& environment = *description.spectral_environment;
    const auto wavelength_state =
        renderer::PathState::create_initial(environment.wavelengths, renderer::VacuumMedium);
    if (!wavelength_state) {
        return std::unexpected(
            scene_error(core::StatusCode::invalid_argument,
                        "A frame scene spectral environment requires a valid wavelength packet."));
    }
    if (!renderer::ConstantEnvironment::create(environment.radiance)) {
        return std::unexpected(scene_error(
            core::StatusCode::invalid_argument,
            "A frame scene spectral environment requires finite non-negative radiance."));
    }

    for (const auto& light : description.punctual_lights) {
        if (auto status = validate_punctual_light(light, environment.wavelengths); !status) {
            return std::unexpected(std::move(status.error()));
        }
    }

    for (const auto& material : description.materials) {
        if (!material.spectral) {
            return std::unexpected(
                scene_error(core::StatusCode::invalid_argument,
                            "A renderable frame scene requires spectral data for every material."));
        }
        if (material.spectral->wavelengths != environment.wavelengths) {
            return std::unexpected(scene_error(
                core::StatusCode::invalid_argument,
                "Frame scene materials and environment must use the same wavelength packet."));
        }
        if (!renderer::LambertianReflection::create(material.spectral->reflectance)) {
            return std::unexpected(
                scene_error(core::StatusCode::invalid_argument,
                            "A frame scene material requires finite reflectance lanes in [0, 1]."));
        }
        if (!renderer::OneSidedSurfaceEmission::create(material.spectral->emitted_radiance)) {
            return std::unexpected(scene_error(
                core::StatusCode::invalid_argument,
                "A frame scene material requires finite non-negative emitted radiance."));
        }
    }
    return {};
}

[[nodiscard]] core::Result<ResolvedInstanceTransforms>
resolve_instance_transforms(const std::vector<SceneInstance>& instances) {
    auto resolved = ResolvedInstanceTransforms{};
    auto states = std::vector<HierarchyState>{};
    auto pending_world = std::vector<std::optional<renderer::AffineTransform>>{};
    auto chain = std::vector<std::size_t>{};
    const auto instance_count = instances.size();
    if (instance_count > resolved.local.max_size() || instance_count > resolved.world.max_size() ||
        instance_count > states.max_size() || instance_count > pending_world.max_size() ||
        instance_count > chain.max_size()) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene hierarchy exceeds host container limits."));
    }

    resolved.local.reserve(instances.size());
    for (const auto& instance : instances) {
        auto local = renderer::AffineTransform::from_matrix(instance.local_to_parent);
        if (!local) {
            return std::unexpected(
                scene_error(core::StatusCode::invalid_argument,
                            "A frame scene instance local transform must be finite, affine, "
                            "and invertible."));
        }
        resolved.local.push_back(std::move(*local));
    }

    states.assign(instances.size(), HierarchyState::unresolved);
    pending_world.resize(instances.size());
    chain.reserve(instances.size());

    for (std::size_t start = 0; start < instances.size(); ++start) {
        if (states[start] == HierarchyState::resolved) {
            continue;
        }

        chain.clear();
        auto current = start;
        while (states[current] != HierarchyState::resolved) {
            if (states[current] == HierarchyState::resolving) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "A frame scene instance hierarchy contains a cycle."));
            }

            states[current] = HierarchyState::resolving;
            chain.push_back(current);
            if (!instances[current].parent) {
                break;
            }

            const auto parent_index = find_record_index(instances, *instances[current].parent);
            if (!parent_index) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "A frame scene instance references an unknown parent instance."));
            }
            current = *parent_index;
        }

        while (!chain.empty()) {
            const auto instance_index = chain.back();
            chain.pop_back();
            const auto& instance = instances[instance_index];

            if (instance.parent) {
                const auto parent_index = find_record_index(instances, *instance.parent);
                if (!parent_index || states[*parent_index] != HierarchyState::resolved ||
                    !pending_world[*parent_index]) {
                    return std::unexpected(
                        scene_error(core::StatusCode::internal_error,
                                    "Frame scene hierarchy resolution lost a validated parent."));
                }

                auto world =
                    renderer::AffineTransform::from_matrix(pending_world[*parent_index]->matrix() *
                                                           resolved.local[instance_index].matrix());
                if (!world) {
                    return std::unexpected(scene_error(
                        core::StatusCode::invalid_argument,
                        "A frame scene hierarchy produced an invalid world transform."));
                }
                pending_world[instance_index] = std::move(*world);
            } else {
                pending_world[instance_index] = resolved.local[instance_index];
            }
            states[instance_index] = HierarchyState::resolved;
        }
    }

    resolved.world.reserve(instances.size());
    for (auto& world : pending_world) {
        if (!world) {
            return std::unexpected(
                scene_error(core::StatusCode::internal_error,
                            "Frame scene hierarchy resolution left an instance unresolved."));
        }
        resolved.world.push_back(std::move(*world));
    }
    return resolved;
}

} // namespace

core::Result<FrameSceneHandle> FrameScene::create(const FrameSceneDescription& description) {
    try {
        auto owned_description = description;
        return create(std::move(owned_description));
    } catch (const std::bad_alloc&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exceeds host container limits."));
    }
}

core::Result<FrameSceneHandle> FrameScene::create(FrameSceneDescription&& description) {
    try {
        sort_by_identifier(description.objects);
        sort_by_identifier(description.geometries);
        sort_by_identifier(description.materials);
        sort_by_identifier(description.instances);

        if (auto status = reject_duplicate_identifiers(
                description.objects, "A frame scene contains duplicate object identifiers.");
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        if (auto status = reject_duplicate_identifiers(
                description.geometries, "A frame scene contains duplicate geometry identifiers.");
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        if (auto status = reject_duplicate_identifiers(
                description.materials, "A frame scene contains duplicate material identifiers.");
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        if (auto status = reject_duplicate_identifiers(
                description.instances, "A frame scene contains duplicate instance identifiers.");
            !status) {
            return std::unexpected(std::move(status.error()));
        }

        for (const auto& geometry : description.geometries) {
            if (!geometry.mesh) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "A frame scene geometry requires an immutable triangle mesh."));
            }
        }

        for (const auto& instance : description.instances) {
            if (!contains_identifier(description.objects, instance.object)) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "A frame scene instance references an unknown object identifier."));
            }
            if (!contains_identifier(description.geometries, instance.geometry)) {
                return std::unexpected(scene_error(
                    core::StatusCode::invalid_argument,
                    "A frame scene instance references an unknown geometry identifier."));
            }
            if (!contains_identifier(description.materials, instance.material)) {
                return std::unexpected(scene_error(
                    core::StatusCode::invalid_argument,
                    "A frame scene instance references an unknown material identifier."));
            }
            if (instance.parent && !contains_identifier(description.instances, *instance.parent)) {
                return std::unexpected(
                    scene_error(core::StatusCode::invalid_argument,
                                "A frame scene instance references an unknown parent instance."));
            }
        }

        if (auto status = validate_spectral_transport(description); !status) {
            return std::unexpected(std::move(status.error()));
        }

        auto transforms = resolve_instance_transforms(description.instances);
        if (!transforms) {
            return std::unexpected(std::move(transforms.error()));
        }

        auto mesh_area_lights = derive_mesh_area_lights(description, transforms->world);
        if (!mesh_area_lights) {
            return std::unexpected(std::move(mesh_area_lights.error()));
        }

        return FrameSceneHandle{new FrameScene{
            std::move(description), std::move(transforms->local), std::move(transforms->world),
            std::move(mesh_area_lights->models), std::move(mesh_area_lights->instance_ids)}};
    } catch (const std::bad_alloc&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(scene_error(core::StatusCode::resource_exhausted,
                                           "Frame scene storage exceeds host container limits."));
    }
}

FrameScene::FrameScene(FrameSceneDescription&& description,
                       std::vector<renderer::AffineTransform>&& local_transforms,
                       std::vector<renderer::AffineTransform>&& world_transforms,
                       std::vector<renderer::MeshAreaLight>&& mesh_area_lights,
                       std::vector<renderer::InstanceId>&& mesh_area_light_instance_ids) noexcept
    : objects_{std::move(description.objects)}, geometries_{std::move(description.geometries)},
      materials_{std::move(description.materials)}, instances_{std::move(description.instances)},
      punctual_lights_{std::move(description.punctual_lights)},
      spectral_environment_{std::move(description.spectral_environment)},
      local_transforms_{std::move(local_transforms)},
      world_transforms_{std::move(world_transforms)},
      mesh_area_lights_{std::move(mesh_area_lights)},
      mesh_area_light_instance_ids_{std::move(mesh_area_light_instance_ids)} {}

std::span<const SceneObject> FrameScene::objects() const noexcept {
    return objects_;
}

std::span<const SceneGeometry> FrameScene::geometries() const noexcept {
    return geometries_;
}

std::span<const SceneMaterial> FrameScene::materials() const noexcept {
    return materials_;
}

std::span<const SceneInstance> FrameScene::instances() const noexcept {
    return instances_;
}

std::span<const ScenePunctualLight> FrameScene::punctual_lights() const noexcept {
    return punctual_lights_;
}

std::span<const renderer::MeshAreaLight> FrameScene::mesh_area_lights() const noexcept {
    return mesh_area_lights_;
}

std::span<const renderer::InstanceId> FrameScene::mesh_area_light_instance_ids() const noexcept {
    return mesh_area_light_instance_ids_;
}

const std::optional<SceneSpectralEnvironment>& FrameScene::spectral_environment() const noexcept {
    return spectral_environment_;
}

core::Result<std::reference_wrapper<const SceneObject>>
FrameScene::object(const renderer::ObjectId id) const {
    return find_record(objects_, id, "The frame scene does not contain the requested object.");
}

core::Result<std::reference_wrapper<const SceneGeometry>>
FrameScene::geometry(const renderer::GeometryId id) const {
    return find_record(geometries_, id, "The frame scene does not contain the requested geometry.");
}

core::Result<std::reference_wrapper<const SceneMaterial>>
FrameScene::material(const renderer::MaterialId id) const {
    return find_record(materials_, id, "The frame scene does not contain the requested material.");
}

core::Result<std::reference_wrapper<const SceneInstance>>
FrameScene::instance(const renderer::InstanceId id) const {
    return find_record(instances_, id, "The frame scene does not contain the requested instance.");
}

core::Result<std::reference_wrapper<const renderer::AffineTransform>>
FrameScene::local_transform(const renderer::InstanceId id) const {
    const auto index = find_record_index(instances_, id);
    if (!index) {
        return std::unexpected(scene_error(
            core::StatusCode::not_found,
            "The frame scene does not contain the requested instance local transform."));
    }
    return std::cref(local_transforms_[*index]);
}

core::Result<std::reference_wrapper<const renderer::AffineTransform>>
FrameScene::world_transform(const renderer::InstanceId id) const {
    const auto index = find_record_index(instances_, id);
    if (!index) {
        return std::unexpected(scene_error(
            core::StatusCode::not_found,
            "The frame scene does not contain the requested instance world transform."));
    }
    return std::cref(world_transforms_[*index]);
}

} // namespace blackframe::engine
