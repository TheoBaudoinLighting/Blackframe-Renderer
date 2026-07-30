#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/SceneIdentifiers.hpp>
#include <functional>
#include <memory>
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

    [[nodiscard]] constexpr bool operator==(const SceneGeometry&) const noexcept = default;
};

struct SceneMaterial final {
    renderer::MaterialId id{};

    [[nodiscard]] constexpr bool operator==(const SceneMaterial&) const noexcept = default;
};

// Objects are logical identities. An instance owns the graph edges that bind
// one object identity to one geometry and one material; no relationship is
// inferred from equal numeric values in different identifier domains.
struct SceneInstance final {
    renderer::InstanceId id{};
    renderer::ObjectId object{};
    renderer::GeometryId geometry{};
    renderer::MaterialId material{};

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

  private:
    explicit FrameScene(FrameSceneDescription&& description) noexcept;

    std::vector<SceneObject> objects_;
    std::vector<SceneGeometry> geometries_;
    std::vector<SceneMaterial> materials_;
    std::vector<SceneInstance> instances_;
};

static_assert(std::is_nothrow_destructible_v<FrameScene>);

} // namespace blackframe::engine
