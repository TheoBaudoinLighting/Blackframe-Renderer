#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/AreaLights.hpp>
#include <Blackframe/Renderer/ClosureSet.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/SceneIdentifiers.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/Transforms.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

namespace blackframe::engine {

enum class SceneClosureFrameMode : std::uint8_t {
    shading_normal = 0U,
    surface_tangent = 1U,
};

[[nodiscard]] constexpr bool
is_known_scene_closure_frame_mode(const SceneClosureFrameMode mode) noexcept {
    switch (mode) {
    case SceneClosureFrameMode::shading_normal:
    case SceneClosureFrameMode::surface_tangent:
        return true;
    }
    return false;
}

// A scene material stores an explicit bounded closure distribution. The active probability prefix
// follows ClosureSet insertion order and the inactive tail is exactly zero. The tangent rotation is
// canonical in [-pi, pi). shading_normal selects the deterministic frame derived from Ns;
// surface_tangent selects the projected surface parameterization and must fail at resolution time
// when that tangent is unusable rather than substituting the shading-normal frame.
struct SceneClosureMixture final {
    [[nodiscard]] static core::Result<SceneClosureMixture>
    create(renderer::ClosureSet closures,
           std::span<const renderer::TransportScalar> component_probabilities,
           SceneClosureFrameMode frame_mode = SceneClosureFrameMode::shading_normal,
           renderer::TransportScalar tangent_rotation_radians = 0.0F);

    [[nodiscard]] static core::Result<SceneClosureMixture>
    create_lambertian(renderer::TransportSpectrum reflectance,
                      SceneClosureFrameMode frame_mode = SceneClosureFrameMode::shading_normal,
                      renderer::TransportScalar tangent_rotation_radians = 0.0F);

    renderer::ClosureSet closures{};
    std::array<renderer::TransportScalar, renderer::MaximumClosureCount> component_probabilities{};
    SceneClosureFrameMode frame_mode{SceneClosureFrameMode::shading_normal};
    renderer::TransportScalar tangent_rotation_radians{};

    [[nodiscard]] constexpr std::span<const renderer::TransportScalar>
    active_component_probabilities() const noexcept {
        return {component_probabilities.data(), static_cast<std::size_t>(closures.size())};
    }

    [[nodiscard]] constexpr bool operator==(const SceneClosureMixture& other) const noexcept {
        if (closures.size() != other.closures.size() ||
            component_probabilities != other.component_probabilities ||
            frame_mode != other.frame_mode ||
            tangent_rotation_radians != other.tangent_rotation_radians) {
            return false;
        }
        const auto closure_count = static_cast<std::size_t>(closures.size());
        for (auto index = std::size_t{}; index < closure_count; ++index) {
            const auto& left = closures.closures()[index];
            const auto& right = other.closures.closures()[index];
            if (left.kind != right.kind || left.lobes != right.lobes ||
                left.weight != right.weight || left.parameters != right.parameters) {
                return false;
            }
        }
        return true;
    }
};

struct SceneObject final {
    renderer::ObjectId id{};

    [[nodiscard]] constexpr bool operator==(const SceneObject&) const noexcept = default;
};

struct SceneGeometry final {
    renderer::GeometryId id{};
    std::shared_ptr<const TriangleMesh> mesh;

    [[nodiscard]] bool operator==(const SceneGeometry&) const noexcept = default;
};

// Wavelength-resolved closure parameters and emission stay in the immutable scene record;
// construction validates them before a render can observe the snapshot.
struct SceneSpectralMaterial final {
    renderer::SampledWavelengths wavelengths{};
    SceneClosureMixture closure_mixture{};
    renderer::TransportSpectrum emitted_radiance{};

    [[nodiscard]] constexpr bool operator==(const SceneSpectralMaterial&) const noexcept = default;
};

static_assert(sizeof(SceneClosureFrameMode) == sizeof(std::uint8_t));
static_assert(std::is_standard_layout_v<SceneClosureMixture>);
static_assert(std::is_trivially_copyable_v<SceneClosureMixture>);

struct SceneMaterial final {
    renderer::MaterialId id{};
    std::optional<SceneSpectralMaterial> spectral{std::nullopt};

    [[nodiscard]] constexpr bool operator==(const SceneMaterial&) const noexcept = default;
};

struct SceneSpectralEnvironment final {
    renderer::SampledWavelengths wavelengths{};
    renderer::TransportSpectrum radiance{};

    [[nodiscard]] constexpr bool
    operator==(const SceneSpectralEnvironment&) const noexcept = default;
};

// Punctual records remain plain comparable scene data. Their spectra are
// interpreted only at the frame's single environment wavelength packet, and
// FrameScene reconstructs the corresponding validated renderer model before
// publishing the snapshot. Vector order is the stable light-registry slot.
struct ScenePointLight final {
    renderer::Point3 position;
    renderer::Vector3 absolute_position_error;
    renderer::TransportSpectrum spectral_radiant_intensity;

    [[nodiscard]] constexpr bool operator==(const ScenePointLight&) const noexcept = default;
};

struct SceneDirectionalLight final {
    renderer::Vector3 propagation_direction;
    renderer::TransportSpectrum spectral_irradiance;

    [[nodiscard]] constexpr bool operator==(const SceneDirectionalLight&) const noexcept = default;
};

struct SceneSpotLight final {
    renderer::Point3 position;
    renderer::Vector3 absolute_position_error;
    renderer::Vector3 emission_direction;
    renderer::TransportScalar inner_half_angle_radians;
    renderer::TransportScalar outer_half_angle_radians;
    renderer::TransportSpectrum on_axis_spectral_radiant_intensity;

    [[nodiscard]] constexpr bool operator==(const SceneSpotLight&) const noexcept = default;
};

using ScenePunctualLight = std::variant<ScenePointLight, SceneDirectionalLight, SceneSpotLight>;

// Objects are logical identities. An instance owns the graph edges that bind
// one object identity to one geometry and one material. A local matrix maps
// instance space into its parent's space; for a root it maps directly to
// world space. Every local matrix is mandatory and validated when the scene is
// closed.
struct SceneInstance final {
    renderer::InstanceId id{};
    std::optional<renderer::InstanceId> parent;
    renderer::ObjectId object{};
    renderer::GeometryId geometry{};
    renderer::MaterialId material{};
    renderer::Matrix4 local_to_parent{};
    renderer::RayMask visibility_mask{renderer::AllRayVisibility};

    [[nodiscard]] constexpr bool operator==(const SceneInstance&) const noexcept = default;
};

struct FrameSceneDescription final {
    std::vector<SceneObject> objects;
    std::vector<SceneGeometry> geometries;
    std::vector<SceneMaterial> materials;
    std::vector<SceneInstance> instances;
    std::vector<ScenePunctualLight> punctual_lights{};
    std::optional<SceneSpectralEnvironment> spectral_environment{std::nullopt};
};

class FrameScene;
using FrameSceneHandle = std::shared_ptr<const FrameScene>;

// Construction closes and validates the graph once. The returned const handle
// keeps its canonically ordered storage alive for every worker rendering the
// frame; identifiers are explicit values and are never derived from addresses
// or storage indices.
class FrameScene final {
  public:
    [[nodiscard]] static core::Result<FrameSceneHandle>
    create(const FrameSceneDescription& description);
    [[nodiscard]] static core::Result<FrameSceneHandle> create(FrameSceneDescription&& description);

    FrameScene(const FrameScene&) = delete;
    FrameScene(FrameScene&&) = delete;
    FrameScene& operator=(const FrameScene&) = delete;
    FrameScene& operator=(FrameScene&&) = delete;
    ~FrameScene() noexcept = default;

    [[nodiscard]] std::span<const SceneObject> objects() const noexcept;
    [[nodiscard]] std::span<const SceneGeometry> geometries() const noexcept;
    [[nodiscard]] std::span<const SceneMaterial> materials() const noexcept;
    [[nodiscard]] std::span<const SceneInstance> instances() const noexcept;
    [[nodiscard]] std::span<const ScenePunctualLight> punctual_lights() const noexcept;
    // Derived in stable instance-identifier order from the exact committed mesh, world transform,
    // and non-black spectral material emission. The two spans are index-aligned.
    [[nodiscard]] std::span<const renderer::MeshAreaLight> mesh_area_lights() const noexcept;
    [[nodiscard]] std::span<const renderer::InstanceId>
    mesh_area_light_instance_ids() const noexcept;
    [[nodiscard]] const std::optional<SceneSpectralEnvironment>&
    spectral_environment() const noexcept;

    [[nodiscard]] core::Result<std::reference_wrapper<const SceneObject>>
    object(renderer::ObjectId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneGeometry>>
    geometry(renderer::GeometryId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneMaterial>>
    material(renderer::MaterialId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const SceneInstance>>
    instance(renderer::InstanceId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const renderer::AffineTransform>>
    local_transform(renderer::InstanceId id) const;
    [[nodiscard]] core::Result<std::reference_wrapper<const renderer::AffineTransform>>
    world_transform(renderer::InstanceId id) const;

  private:
    explicit FrameScene(FrameSceneDescription&& description,
                        std::vector<renderer::AffineTransform>&& local_transforms,
                        std::vector<renderer::AffineTransform>&& world_transforms,
                        std::vector<renderer::MeshAreaLight>&& mesh_area_lights,
                        std::vector<renderer::InstanceId>&& mesh_area_light_instance_ids) noexcept;

    std::vector<SceneObject> objects_;
    std::vector<SceneGeometry> geometries_;
    std::vector<SceneMaterial> materials_;
    std::vector<SceneInstance> instances_;
    std::vector<ScenePunctualLight> punctual_lights_;
    std::optional<SceneSpectralEnvironment> spectral_environment_;
    std::vector<renderer::AffineTransform> local_transforms_;
    std::vector<renderer::AffineTransform> world_transforms_;
    std::vector<renderer::MeshAreaLight> mesh_area_lights_;
    std::vector<renderer::InstanceId> mesh_area_light_instance_ids_;
};

static_assert(std::is_nothrow_destructible_v<FrameScene>);

} // namespace blackframe::engine
