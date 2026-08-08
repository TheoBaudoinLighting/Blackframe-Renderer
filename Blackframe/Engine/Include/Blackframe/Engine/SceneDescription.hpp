#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/MisHeuristics.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PixelJitter.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <variant>
#include <vector>

namespace blackframe::engine {

struct SceneFilmDescription final {
    renderer::FilmId id{};
    renderer::RenderExtent extent{};
    renderer::FilmCrop crop{};
    renderer::AccumulationPrecision accumulation_precision{
        renderer::AccumulationPrecision::float32};

    [[nodiscard]] constexpr bool operator==(const SceneFilmDescription& other) const noexcept {
        return id == other.id && extent.width == other.extent.width &&
               extent.height == other.extent.height && crop == other.crop &&
               accumulation_precision == other.accumulation_precision;
    }
};

struct ScenePinholeCameraDescription final {
    renderer::Point3 origin{};
    renderer::OrthonormalFrame orientation;
    renderer::TransportScalar vertical_field_of_view_radians{};
    renderer::TransportScalar t_min{};
    renderer::TransportScalar t_max{};
    renderer::RayMask visibility_mask{renderer::AllRayVisibility};
    renderer::MediumId current_medium{renderer::VacuumMedium};
};

using SceneCameraModel = std::variant<ScenePinholeCameraDescription>;

struct SceneCameraDescription final {
    renderer::CameraId id{};
    renderer::FilmId film{};
    SceneCameraModel model;
};

struct SceneRenderOptions final {
    std::uint32_t samples_per_pixel{1U};
    std::uint32_t maximum_path_depth{4U};
    std::uint32_t tile_edge_length{16U};
    std::uint64_t seed{};
    renderer::PixelJitterMode pixel_jitter{renderer::PixelJitterMode::uniform};
    renderer::MisHeuristic mis_heuristic{renderer::MisHeuristic::power};
    renderer::LightSamplingStrategy light_sampling_strategy{
        renderer::LightSamplingStrategy::uniform};
    renderer::PathDepthLimits depth_limits{
        .diffuse = 4U, .glossy = 4U, .specular = 4U, .transmission = 4U, .volume = 4U};
    renderer::RussianRoulettePolicy roulette_policy{renderer::RussianRoulettePolicy::disabled()};

    [[nodiscard]] constexpr bool operator==(const SceneRenderOptions&) const noexcept = default;
};

struct SceneRenderOptionsDescription final {
    renderer::RenderOptionsId id{};
    renderer::FilmId film{};
    SceneRenderOptions options{};

    [[nodiscard]] constexpr bool
    operator==(const SceneRenderOptionsDescription&) const noexcept = default;
};

using SceneLightValue =
    std::variant<ScenePointLight, SceneDirectionalLight, SceneSpotLight, SceneSpectralEnvironment>;

struct SceneLightDescription final {
    renderer::LightId id{};
    SceneLightValue light{};

    [[nodiscard]] constexpr bool operator==(const SceneLightDescription&) const noexcept = default;
};

// Mutable authoring input. SceneDescription::create validates and canonicalizes every registry
// before publishing an immutable value. Assets here are immutable host values; no
// Embree, CUDA, XPU, queue, device-memory, or backend-scene object crosses this contract.
struct SceneDescriptionInput final {
    renderer::FilmId active_film{};
    renderer::CameraId active_camera{};
    renderer::RenderOptionsId active_render_options{};
    std::vector<SceneFilmDescription> films{};
    std::vector<SceneCameraDescription> cameras{};
    std::vector<SceneRenderOptionsDescription> render_options{};
    std::vector<SceneConstantTexture> constant_textures{};
    std::vector<SceneHostImageTexture> host_image_textures{};
    std::vector<SceneObject> objects{};
    std::vector<SceneGeometry> geometries{};
    std::vector<SceneMaterial> materials{};
    std::vector<SceneInstance> instances{};
    std::vector<SceneLightDescription> lights{};
};

class SceneDescription final {
  public:
    [[nodiscard]] static core::Result<SceneDescription> create(SceneDescriptionInput input);

    [[nodiscard]] renderer::FilmId active_film_id() const noexcept;
    [[nodiscard]] renderer::CameraId active_camera_id() const noexcept;
    [[nodiscard]] renderer::RenderOptionsId active_render_options_id() const noexcept;

    [[nodiscard]] const SceneFilmDescription& active_film() const noexcept;
    [[nodiscard]] const SceneCameraDescription& active_camera() const noexcept;
    [[nodiscard]] const SceneRenderOptionsDescription& active_render_options() const noexcept;

    [[nodiscard]] std::span<const SceneFilmDescription> films() const noexcept;
    [[nodiscard]] std::span<const SceneCameraDescription> cameras() const noexcept;
    [[nodiscard]] std::span<const SceneRenderOptionsDescription> render_options() const noexcept;
    [[nodiscard]] std::span<const SceneConstantTexture> constant_textures() const noexcept;
    [[nodiscard]] std::span<const SceneHostImageTexture> host_image_textures() const noexcept;
    [[nodiscard]] std::span<const SceneObject> objects() const noexcept;
    [[nodiscard]] std::span<const SceneGeometry> geometries() const noexcept;
    [[nodiscard]] std::span<const SceneMaterial> materials() const noexcept;
    [[nodiscard]] std::span<const SceneInstance> instances() const noexcept;
    [[nodiscard]] std::span<const SceneLightDescription> lights() const noexcept;

    [[nodiscard]] core::Result<std::reference_wrapper<const SceneFilmDescription>>
    film(renderer::FilmId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneCameraDescription>>
    camera(renderer::CameraId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneRenderOptionsDescription>>
    options(renderer::RenderOptionsId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneConstantTexture>>
    constant_texture(renderer::TextureId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneHostImageTexture>>
    host_image_texture(renderer::TextureId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneObject>>
    object(renderer::ObjectId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneGeometry>>
    geometry(renderer::GeometryId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneMaterial>>
    material(renderer::MaterialId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneInstance>>
    instance(renderer::InstanceId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneLightDescription>>
    light(renderer::LightId id) const;

  private:
    explicit SceneDescription(SceneDescriptionInput&& input, std::size_t active_film_index,
                              std::size_t active_camera_index,
                              std::size_t active_render_options_index) noexcept;

    renderer::FilmId active_film_id_{};
    renderer::CameraId active_camera_id_{};
    renderer::RenderOptionsId active_render_options_id_{};
    std::size_t active_film_index_{};
    std::size_t active_camera_index_{};
    std::size_t active_render_options_index_{};
    std::vector<SceneFilmDescription> films_{};
    std::vector<SceneCameraDescription> cameras_{};
    std::vector<SceneRenderOptionsDescription> render_options_{};
    std::vector<SceneConstantTexture> constant_textures_{};
    std::vector<SceneHostImageTexture> host_image_textures_{};
    std::vector<SceneObject> objects_{};
    std::vector<SceneGeometry> geometries_{};
    std::vector<SceneMaterial> materials_{};
    std::vector<SceneInstance> instances_{};
    std::vector<SceneLightDescription> lights_{};
};

} // namespace blackframe::engine
