#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/SceneIdentifiers.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace blackframe::engine {

enum class AccelBackendKind : std::uint8_t {
    analytic_reference,
    embree,
};

// Describes one immutable triangle mesh that is already expressed in world
// space. The backend retains the mesh for its complete lifetime.
struct AccelGeometry final {
    std::shared_ptr<const TriangleMesh> mesh;
    renderer::InstanceId instance{};
    renderer::GeometryId geometry{};
    renderer::MaterialId material{};
    renderer::RayMask visibility_mask{renderer::AllRayVisibility};
};

struct AccelHit final {
    renderer::TriangleHit triangle;
    renderer::SurfaceIdentifiers identifiers;

    [[nodiscard]] constexpr bool operator==(const AccelHit&) const noexcept = default;
};

// Scene ray queries are separate from the renderer's five central transport
// interfaces. Ray time must be normalized to [0, 1]. A coplanar triangle does
// not define a surface crossing and is skipped; every other numerical or
// backend failure remains an explicit error. A miss is represented by an
// empty optional or false.
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
    AccelBackend() noexcept = default;

    [[nodiscard]] static core::Status
    validate_geometry_input(std::span<const AccelGeometry> geometries);

    [[nodiscard]] static core::Status validate_ray(const renderer::Ray& ray);
};

using AccelBackendFactory =
    core::Result<std::unique_ptr<AccelBackend>> (*)(std::span<const AccelGeometry> geometries);

[[nodiscard]] core::Result<std::unique_ptr<AccelBackend>>
create_analytic_accel_backend(std::span<const AccelGeometry> geometries);

} // namespace blackframe::engine
