#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/SceneIdentifiers.hpp>
#include <Blackframe/Renderer/Transforms.hpp>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::engine {

struct SceneObject final {
    renderer::ObjectId id{};

    [[nodiscard]] constexpr bool operator==(const SceneObject&) const noexcept = default;
};

struct SceneGeometry final {
    renderer::GeometryId id{};
    std::shared_ptr<const TriangleMesh> mesh;

    [[nodiscard]] bool operator==(const SceneGeometry&) const noexcept = default;
};

struct SceneMaterial final {
    renderer::MaterialId id{};

    [[nodiscard]] constexpr bool operator==(const SceneMaterial&) const noexcept = default;
};

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
                        std::vector<renderer::AffineTransform>&& world_transforms) noexcept;

    std::vector<SceneObject> objects_;
    std::vector<SceneGeometry> geometries_;
    std::vector<SceneMaterial> materials_;
    std::vector<SceneInstance> instances_;
    std::vector<renderer::AffineTransform> local_transforms_;
    std::vector<renderer::AffineTransform> world_transforms_;
};

static_assert(std::is_nothrow_destructible_v<FrameScene>);

} // namespace blackframe::engine
