#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/SceneIdentifiers.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace blackframe::engine {

enum class AccelBackendKind : std::uint8_t {
    analytic_reference,
    embree,
};

// Canonical backend-neutral instance data prepared from one immutable frame
// scene. The mesh remains in object space and object_to_world is the fully
// resolved hierarchy transform.
struct AccelInstance final {
    std::shared_ptr<const TriangleMesh> mesh;
    renderer::AffineTransform object_to_world;
    renderer::ObjectId object{};
    renderer::InstanceId instance{};
    renderer::GeometryId geometry{};
    renderer::MaterialId material{};
    renderer::RayMask visibility_mask{renderer::AllRayVisibility};
};

struct AccelHit final {
    renderer::ObjectId object;
    renderer::TriangleHit triangle;
    renderer::SurfaceIdentifiers identifiers;

    [[nodiscard]] constexpr bool operator==(const AccelHit&) const noexcept = default;
};

// Scene ray queries are separate from the renderer's five central transport
// interfaces. A backend retains the immutable scene snapshot passed to its
// factory. Ray time must be normalized to [0, 1]. A coplanar triangle does not
// define a surface crossing and is skipped; every other numerical or backend
// failure remains an explicit error. A miss is represented by an empty
// optional or false. An instance is eligible only when
// (ray.mask() & visibility_mask) != 0, so a zero ray mask sees no instances.
// occluded() is an opaque any-hit query that returns at the first crossing in
// the closed [tMin, tMax] interval without reconstructing surface data.
class AccelBackend {
  public:
    virtual ~AccelBackend() noexcept = default;

    AccelBackend(const AccelBackend&) = delete;
    AccelBackend& operator=(const AccelBackend&) = delete;
    AccelBackend(AccelBackend&&) = delete;
    AccelBackend& operator=(AccelBackend&&) = delete;

    [[nodiscard]] virtual AccelBackendKind kind() const noexcept = 0;

    [[nodiscard]] virtual core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray& ray) const = 0;

    [[nodiscard]] virtual core::Result<bool> occluded(const renderer::Ray& ray) const = 0;

  protected:
    explicit AccelBackend(FrameSceneHandle scene) noexcept;

    [[nodiscard]] static core::Result<std::vector<AccelInstance>>
    prepare_instances(const FrameSceneHandle& scene);

    [[nodiscard]] static core::Status validate_ray(const renderer::Ray& ray);

    [[nodiscard]] static core::Result<renderer::Ray>
    object_space_ray(const renderer::Ray& world_ray, const AccelInstance& instance);

    [[nodiscard]] static core::Result<renderer::Triangle>
    world_space_triangle(const AccelInstance& instance, std::size_t triangle_index);

  private:
    FrameSceneHandle scene_;
};

using AccelBackendFactory = core::Result<std::unique_ptr<AccelBackend>> (*)(FrameSceneHandle scene);

[[nodiscard]] core::Result<std::unique_ptr<AccelBackend>>
create_analytic_accel_backend(FrameSceneHandle scene);

} // namespace blackframe::engine
