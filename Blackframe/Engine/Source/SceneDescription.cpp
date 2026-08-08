#include "SceneHierarchyValidation.hpp"
#include "SceneRecordValidation.hpp"

#include <Blackframe/Engine/SceneDescription.hpp>
#include <Blackframe/Renderer/Emission.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/PunctualLights.hpp>
#include <Blackframe/Renderer/Transforms.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error description_error(const core::StatusCode code,
                                            const std::string_view message) {
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
                                                        const std::string_view domain) {
    const auto duplicate = std::ranges::adjacent_find(
        records, [](const Record& left, const Record& right) { return left.id == right.id; });
    if (duplicate != records.end()) {
        return std::unexpected(
            description_error(core::StatusCode::invalid_argument,
                              std::string{"A scene description contains duplicate "} +
                                  std::string{domain} + " identifiers."));
    }
    return {};
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
    const auto index = find_record_index(records, id);
    if (!index) {
        return std::unexpected(description_error(core::StatusCode::not_found, missing_message));
    }
    return std::cref(records[*index]);
}

[[nodiscard]] core::Status validate_film(const SceneFilmDescription& film) {
    switch (film.accumulation_precision) {
    case renderer::AccumulationPrecision::float32:
    case renderer::AccumulationPrecision::float64:
        break;
    default:
        return std::unexpected(description_error(
            core::StatusCode::invalid_argument,
            "A scene film requires an explicit supported accumulation precision."));
    }
    return renderer::validate_film_crop(film.extent, film.crop);
}

[[nodiscard]] core::Status validate_camera(const SceneCameraDescription& camera,
                                           const SceneFilmDescription& film) {
    if (camera.model.valueless_by_exception()) {
        return std::unexpected(description_error(core::StatusCode::invalid_argument,
                                                 "A scene camera model has no value."));
    }
    return std::visit(
        [&](const auto& model) -> core::Status {
            using Model = std::remove_cvref_t<decltype(model)>;
            if constexpr (std::is_same_v<Model, ScenePinholeCameraDescription>) {
                if (model.current_medium != renderer::VacuumMedium) {
                    return std::unexpected(description_error(
                        core::StatusCode::invalid_argument,
                        "A scene pinhole camera cannot reference a medium because this scene "
                        "description has no medium registry."));
                }
                const auto validated = renderer::PinholeCamera::create(
                    model.origin, model.orientation, film.extent,
                    model.vertical_field_of_view_radians, model.t_min, model.t_max,
                    model.visibility_mask, model.current_medium);
                if (!validated) {
                    return std::unexpected(validated.error());
                }
            }
            return {};
        },
        camera.model);
}

[[nodiscard]] bool known_pixel_jitter(const renderer::PixelJitterMode mode) noexcept {
    switch (mode) {
    case renderer::PixelJitterMode::center:
    case renderer::PixelJitterMode::uniform:
        return true;
    }
    return false;
}

[[nodiscard]] bool known_mis_heuristic(const renderer::MisHeuristic heuristic) noexcept {
    switch (heuristic) {
    case renderer::MisHeuristic::balance:
    case renderer::MisHeuristic::power:
        return true;
    }
    return false;
}

[[nodiscard]] bool
known_light_sampling_strategy(const renderer::LightSamplingStrategy strategy) noexcept {
    switch (strategy) {
    case renderer::LightSamplingStrategy::uniform:
    case renderer::LightSamplingStrategy::power_weighted:
    case renderer::LightSamplingStrategy::spatial_tree:
        return true;
    }
    return false;
}

[[nodiscard]] core::Status validate_render_options(const SceneRenderOptionsDescription& description,
                                                   const SceneFilmDescription& film) {
    const auto& options = description.options;
    const auto configuration = renderer::RenderConfiguration{
        .extent = film.extent,
        .samples_per_pixel = options.samples_per_pixel,
        .maximum_path_depth = options.maximum_path_depth,
        .tile_edge_length = options.tile_edge_length,
        .seed = options.seed,
        .xpu_device_id = {},
    };
    if (auto status = renderer::validate_render_configuration(configuration); !status) {
        return status;
    }
    if (!known_pixel_jitter(options.pixel_jitter)) {
        return std::unexpected(description_error(
            core::StatusCode::invalid_argument,
            "Scene render options require an explicit supported pixel-jitter mode."));
    }
    if (!known_mis_heuristic(options.mis_heuristic)) {
        return std::unexpected(
            description_error(core::StatusCode::invalid_argument,
                              "Scene render options require an explicit supported MIS heuristic."));
    }
    if (!known_light_sampling_strategy(options.light_sampling_strategy)) {
        return std::unexpected(description_error(
            core::StatusCode::invalid_argument,
            "Scene render options require an explicit supported light-sampling strategy."));
    }
    if (auto status = renderer::validate_russian_roulette_policy(options.roulette_policy);
        !status) {
        return status;
    }
    const auto limit_exceeds_global = [&](const std::uint32_t limit) {
        return limit > options.maximum_path_depth;
    };
    if (limit_exceeds_global(options.depth_limits.diffuse) ||
        limit_exceeds_global(options.depth_limits.glossy) ||
        limit_exceeds_global(options.depth_limits.specular) ||
        limit_exceeds_global(options.depth_limits.transmission) ||
        limit_exceeds_global(options.depth_limits.volume)) {
        return std::unexpected(description_error(
            core::StatusCode::invalid_argument,
            "A scene path-category limit cannot exceed the global path-depth limit."));
    }
    return {};
}

[[nodiscard]] core::Status
validate_material_texture_references(const std::vector<SceneMaterial>& materials,
                                     const std::vector<SceneHostImageTexture>& host_images) {
    for (const auto& material : materials) {
        if (!material.spectral) {
            continue;
        }
        if (material.spectral->normal_map) {
            if (auto status = scene_record_validation::validate_normal_map_binding(
                    host_images, *material.spectral->normal_map);
                !status) {
                return status;
            }
        }
        if (material.spectral->bump_map) {
            if (auto status = scene_record_validation::validate_bump_map_binding(
                    host_images, *material.spectral->bump_map);
                !status) {
                return status;
            }
        }
    }
    return {};
}

[[nodiscard]] core::Status validate_instance_references(
    const std::vector<SceneObject>& objects, const std::vector<SceneGeometry>& geometries,
    const std::vector<SceneMaterial>& materials, const std::vector<SceneInstance>& instances) {
    for (const auto& geometry : geometries) {
        if (!geometry.mesh) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "A scene geometry requires an immutable triangle mesh."));
        }
    }

    for (const auto& instance : instances) {
        if (!find_record_index(objects, instance.object)) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "A scene instance references an unknown object identifier."));
        }
        if (!find_record_index(geometries, instance.geometry)) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "A scene instance references an unknown geometry identifier."));
        }
        if (!find_record_index(materials, instance.material)) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "A scene instance references an unknown material identifier."));
        }
        if (instance.parent && !find_record_index(instances, *instance.parent)) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "A scene instance references an unknown parent identifier."));
        }
    }
    return {};
}

[[nodiscard]] core::Result<std::reference_wrapper<const SceneSpectralEnvironment>>
validate_light_registry(const std::vector<SceneLightDescription>& lights) {
    const SceneSpectralEnvironment* environment = nullptr;
    for (const auto& light : lights) {
        if (light.light.valueless_by_exception()) {
            return std::unexpected(description_error(core::StatusCode::invalid_argument,
                                                     "A scene light has no value."));
        }
        if (const auto* candidate = std::get_if<SceneSpectralEnvironment>(&light.light)) {
            if (environment != nullptr) {
                return std::unexpected(description_error(
                    core::StatusCode::invalid_argument,
                    "A scene description supports at most one spectral environment light."));
            }
            environment = candidate;
        }
    }
    if (environment == nullptr) {
        return std::unexpected(description_error(core::StatusCode::not_found,
                                                 "The scene has no spectral environment light."));
    }
    return std::cref(*environment);
}

[[nodiscard]] core::Status
validate_spectral_records(const std::vector<SceneMaterial>& materials,
                          const std::vector<SceneLightDescription>& lights) {
    const auto environment = validate_light_registry(lights);
    const auto requires_environment =
        std::ranges::any_of(
            materials,
            [](const SceneMaterial& material) { return material.spectral.has_value(); }) ||
        std::ranges::any_of(lights, [](const SceneLightDescription& light) {
            return !std::holds_alternative<SceneSpectralEnvironment>(light.light);
        });
    if (!environment) {
        if (environment.error().code == core::StatusCode::not_found && !requires_environment) {
            return {};
        }
        if (environment.error().code == core::StatusCode::not_found) {
            return std::unexpected(description_error(
                core::StatusCode::invalid_argument,
                "Spectral materials and punctual lights require one explicit environment "
                "wavelength packet."));
        }
        return std::unexpected(environment.error());
    }

    const auto& spectral_environment = environment->get();
    if (!renderer::PathState::create_initial(spectral_environment.wavelengths,
                                             renderer::VacuumMedium) ||
        !renderer::ConstantEnvironment::create(spectral_environment.radiance)) {
        return std::unexpected(description_error(
            core::StatusCode::invalid_argument,
            "A scene spectral environment requires valid wavelengths and finite non-negative "
            "radiance."));
    }

    for (const auto& light : lights) {
        const auto status = std::visit(
            [&](const auto& value) -> core::Status {
                using Value = std::remove_cvref_t<decltype(value)>;
                if constexpr (std::is_same_v<Value, ScenePointLight>) {
                    if (!renderer::PointLight::create(value.position, value.absolute_position_error,
                                                      spectral_environment.wavelengths,
                                                      value.spectral_radiant_intensity)) {
                        return std::unexpected(description_error(
                            core::StatusCode::invalid_argument,
                            "A scene point light contains invalid transport data."));
                    }
                } else if constexpr (std::is_same_v<Value, SceneDirectionalLight>) {
                    if (!renderer::DirectionalLight::create(value.propagation_direction,
                                                            spectral_environment.wavelengths,
                                                            value.spectral_irradiance)) {
                        return std::unexpected(description_error(
                            core::StatusCode::invalid_argument,
                            "A scene directional light contains invalid transport data."));
                    }
                } else if constexpr (std::is_same_v<Value, SceneSpotLight>) {
                    if (!renderer::SpotLight::create(
                            value.position, value.absolute_position_error, value.emission_direction,
                            value.inner_half_angle_radians, value.outer_half_angle_radians,
                            spectral_environment.wavelengths,
                            value.on_axis_spectral_radiant_intensity)) {
                        return std::unexpected(description_error(
                            core::StatusCode::invalid_argument,
                            "A scene spot light contains invalid transport data."));
                    }
                }
                return {};
            },
            light.light);
        if (!status) {
            return status;
        }
    }

    for (const auto& material : materials) {
        if (!material.spectral) {
            return std::unexpected(description_error(
                core::StatusCode::invalid_argument,
                "Every material in a spectral scene requires explicit spectral data."));
        }
        if (material.spectral->wavelengths != spectral_environment.wavelengths) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "Scene materials and lights must share one wavelength packet."));
        }
        if (auto status = scene_record_validation::validate_closure_mixture(
                material.spectral->closure_mixture);
            !status) {
            return status;
        }
        if (!renderer::OneSidedSurfaceEmission::create(material.spectral->emitted_radiance)) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "A scene material contains invalid spectral emission data."));
        }
    }
    return {};
}

} // namespace

core::Result<SceneDescription> SceneDescription::create(SceneDescriptionInput input) {
    try {
        sort_by_identifier(input.films);
        sort_by_identifier(input.cameras);
        sort_by_identifier(input.render_options);
        sort_by_identifier(input.constant_textures);
        sort_by_identifier(input.host_image_textures);
        sort_by_identifier(input.objects);
        sort_by_identifier(input.geometries);
        sort_by_identifier(input.materials);
        sort_by_identifier(input.instances);
        sort_by_identifier(input.lights);

        const auto reject_duplicates = [&]() -> core::Status {
            if (auto status = reject_duplicate_identifiers(input.films, "film"); !status) {
                return status;
            }
            if (auto status = reject_duplicate_identifiers(input.cameras, "camera"); !status) {
                return status;
            }
            if (auto status = reject_duplicate_identifiers(input.render_options, "render-options");
                !status) {
                return status;
            }
            if (auto status =
                    reject_duplicate_identifiers(input.constant_textures, "constant-texture");
                !status) {
                return status;
            }
            if (auto status =
                    reject_duplicate_identifiers(input.host_image_textures, "host-image-texture");
                !status) {
                return status;
            }
            if (auto status = reject_duplicate_identifiers(input.objects, "object"); !status) {
                return status;
            }
            if (auto status = reject_duplicate_identifiers(input.geometries, "geometry"); !status) {
                return status;
            }
            if (auto status = reject_duplicate_identifiers(input.materials, "material"); !status) {
                return status;
            }
            if (auto status = reject_duplicate_identifiers(input.instances, "instance"); !status) {
                return status;
            }
            return reject_duplicate_identifiers(input.lights, "light");
        }();
        if (!reject_duplicates) {
            return std::unexpected(reject_duplicates.error());
        }
        if (auto status = scene_record_validation::reject_shared_texture_identifiers(
                input.constant_textures, input.host_image_textures);
            !status) {
            return std::unexpected(status.error());
        }

        const auto active_film_index = find_record_index(input.films, input.active_film);
        const auto active_camera_index = find_record_index(input.cameras, input.active_camera);
        const auto active_options_index =
            find_record_index(input.render_options, input.active_render_options);
        if (!active_film_index) {
            return std::unexpected(description_error(
                core::StatusCode::invalid_argument,
                "A scene description references an unknown active film identifier."));
        }
        if (!active_camera_index) {
            return std::unexpected(description_error(
                core::StatusCode::invalid_argument,
                "A scene description references an unknown active camera identifier."));
        }
        if (!active_options_index) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "A scene description references unknown active render-options."));
        }

        for (const auto& film : input.films) {
            if (auto status = validate_film(film); !status) {
                return std::unexpected(status.error());
            }
        }
        for (const auto& camera : input.cameras) {
            const auto film_index = find_record_index(input.films, camera.film);
            if (!film_index) {
                return std::unexpected(
                    description_error(core::StatusCode::invalid_argument,
                                      "A scene camera references an unknown film identifier."));
            }
            if (auto status = validate_camera(camera, input.films[*film_index]); !status) {
                return std::unexpected(status.error());
            }
        }
        for (const auto& options : input.render_options) {
            const auto film_index = find_record_index(input.films, options.film);
            if (!film_index) {
                return std::unexpected(description_error(
                    core::StatusCode::invalid_argument,
                    "Scene render options reference an unknown film identifier."));
            }
            if (auto status = validate_render_options(options, input.films[*film_index]); !status) {
                return std::unexpected(status.error());
            }
        }
        if (input.cameras[*active_camera_index].film != input.active_film) {
            return std::unexpected(
                description_error(core::StatusCode::invalid_argument,
                                  "The active scene camera must reference the active film."));
        }
        if (input.render_options[*active_options_index].film != input.active_film) {
            return std::unexpected(description_error(
                core::StatusCode::invalid_argument,
                "The active scene render-options must reference the active film."));
        }

        for (const auto& texture : input.constant_textures) {
            if (auto status = scene_record_validation::validate_constant_texture(texture);
                !status) {
                return std::unexpected(status.error());
            }
        }
        for (const auto& texture : input.host_image_textures) {
            if (auto status = scene_record_validation::validate_host_image_texture(texture);
                !status) {
                return std::unexpected(status.error());
            }
        }
        if (auto status =
                validate_material_texture_references(input.materials, input.host_image_textures);
            !status) {
            return std::unexpected(status.error());
        }
        if (auto status = validate_instance_references(input.objects, input.geometries,
                                                       input.materials, input.instances);
            !status) {
            return std::unexpected(status.error());
        }
        if (auto transforms =
                scene_hierarchy_validation::resolve_instance_transforms(input.instances);
            !transforms) {
            return std::unexpected(transforms.error());
        }
        if (auto status = validate_spectral_records(input.materials, input.lights); !status) {
            return std::unexpected(status.error());
        }

        return SceneDescription{std::move(input), *active_film_index, *active_camera_index,
                                *active_options_index};
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            description_error(core::StatusCode::resource_exhausted,
                              "Scene description validation exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            description_error(core::StatusCode::resource_exhausted,
                              "Scene description storage exceeds host container limits."));
    }
}

SceneDescription::SceneDescription(SceneDescriptionInput&& input,
                                   const std::size_t active_film_index,
                                   const std::size_t active_camera_index,
                                   const std::size_t active_render_options_index) noexcept
    : active_film_id_{input.active_film}, active_camera_id_{input.active_camera},
      active_render_options_id_{input.active_render_options}, active_film_index_{active_film_index},
      active_camera_index_{active_camera_index},
      active_render_options_index_{active_render_options_index}, films_{std::move(input.films)},
      cameras_{std::move(input.cameras)}, render_options_{std::move(input.render_options)},
      constant_textures_{std::move(input.constant_textures)},
      host_image_textures_{std::move(input.host_image_textures)},
      objects_{std::move(input.objects)}, geometries_{std::move(input.geometries)},
      materials_{std::move(input.materials)}, instances_{std::move(input.instances)},
      lights_{std::move(input.lights)} {}

renderer::FilmId SceneDescription::active_film_id() const noexcept {
    return active_film_id_;
}

renderer::CameraId SceneDescription::active_camera_id() const noexcept {
    return active_camera_id_;
}

renderer::RenderOptionsId SceneDescription::active_render_options_id() const noexcept {
    return active_render_options_id_;
}

const SceneFilmDescription& SceneDescription::active_film() const noexcept {
    return films_[active_film_index_];
}

const SceneCameraDescription& SceneDescription::active_camera() const noexcept {
    return cameras_[active_camera_index_];
}

const SceneRenderOptionsDescription& SceneDescription::active_render_options() const noexcept {
    return render_options_[active_render_options_index_];
}

std::span<const SceneFilmDescription> SceneDescription::films() const noexcept {
    return films_;
}

std::span<const SceneCameraDescription> SceneDescription::cameras() const noexcept {
    return cameras_;
}

std::span<const SceneRenderOptionsDescription> SceneDescription::render_options() const noexcept {
    return render_options_;
}

std::span<const SceneConstantTexture> SceneDescription::constant_textures() const noexcept {
    return constant_textures_;
}

std::span<const SceneHostImageTexture> SceneDescription::host_image_textures() const noexcept {
    return host_image_textures_;
}

std::span<const SceneObject> SceneDescription::objects() const noexcept {
    return objects_;
}

std::span<const SceneGeometry> SceneDescription::geometries() const noexcept {
    return geometries_;
}

std::span<const SceneMaterial> SceneDescription::materials() const noexcept {
    return materials_;
}

std::span<const SceneInstance> SceneDescription::instances() const noexcept {
    return instances_;
}

std::span<const SceneLightDescription> SceneDescription::lights() const noexcept {
    return lights_;
}

core::Result<std::reference_wrapper<const SceneFilmDescription>>
SceneDescription::film(const renderer::FilmId id) const {
    return find_record(films_, id, "The scene description does not contain the requested film.");
}

core::Result<std::reference_wrapper<const SceneCameraDescription>>
SceneDescription::camera(const renderer::CameraId id) const {
    return find_record(cameras_, id,
                       "The scene description does not contain the requested camera.");
}

core::Result<std::reference_wrapper<const SceneRenderOptionsDescription>>
SceneDescription::options(const renderer::RenderOptionsId id) const {
    return find_record(render_options_, id,
                       "The scene description does not contain the requested render-options.");
}

core::Result<std::reference_wrapper<const SceneConstantTexture>>
SceneDescription::constant_texture(const renderer::TextureId id) const {
    return find_record(constant_textures_, id,
                       "The scene description does not contain the requested constant texture.");
}

core::Result<std::reference_wrapper<const SceneHostImageTexture>>
SceneDescription::host_image_texture(const renderer::TextureId id) const {
    return find_record(host_image_textures_, id,
                       "The scene description does not contain the requested host-image texture.");
}

core::Result<std::reference_wrapper<const SceneObject>>
SceneDescription::object(const renderer::ObjectId id) const {
    return find_record(objects_, id,
                       "The scene description does not contain the requested object.");
}

core::Result<std::reference_wrapper<const SceneGeometry>>
SceneDescription::geometry(const renderer::GeometryId id) const {
    return find_record(geometries_, id,
                       "The scene description does not contain the requested geometry.");
}

core::Result<std::reference_wrapper<const SceneMaterial>>
SceneDescription::material(const renderer::MaterialId id) const {
    return find_record(materials_, id,
                       "The scene description does not contain the requested material.");
}

core::Result<std::reference_wrapper<const SceneInstance>>
SceneDescription::instance(const renderer::InstanceId id) const {
    return find_record(instances_, id,
                       "The scene description does not contain the requested instance.");
}

core::Result<std::reference_wrapper<const SceneLightDescription>>
SceneDescription::light(const renderer::LightId id) const {
    return find_record(lights_, id, "The scene description does not contain the requested light.");
}

} // namespace blackframe::engine
