#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Renderer/CapabilityRegistry.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <embree4/rtcore.h>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

struct DeviceDeleter final {
    void operator()(RTCDevice device) const noexcept {
        rtcReleaseDevice(device);
    }
};

struct SceneDeleter final {
    void operator()(RTCScene scene) const noexcept {
        rtcReleaseScene(scene);
    }
};

struct GeometryDeleter final {
    void operator()(RTCGeometry geometry) const noexcept {
        rtcReleaseGeometry(geometry);
    }
};

using DeviceHandle = std::unique_ptr<std::remove_pointer_t<RTCDevice>, DeviceDeleter>;
using SceneHandle = std::unique_ptr<std::remove_pointer_t<RTCScene>, SceneDeleter>;
using GeometryHandle = std::unique_ptr<std::remove_pointer_t<RTCGeometry>, GeometryDeleter>;

[[nodiscard]] core::StatusCode embree_status_code(const RTCError error) noexcept {
    switch (error) {
    case RTC_ERROR_OUT_OF_MEMORY:
        return core::StatusCode::resource_exhausted;
    case RTC_ERROR_UNSUPPORTED_CPU:
    case RTC_ERROR_LEVEL_ZERO_RAYTRACING_SUPPORT_MISSING:
        return core::StatusCode::unavailable;
    case RTC_ERROR_INVALID_ARGUMENT:
    case RTC_ERROR_INVALID_OPERATION:
        return core::StatusCode::internal_error;
    case RTC_ERROR_UNKNOWN:
    case RTC_ERROR_CANCELLED:
        return core::StatusCode::platform_error;
    case RTC_ERROR_NONE:
        return core::StatusCode::internal_error;
    }
    return core::StatusCode::platform_error;
}

[[nodiscard]] core::Error embree_error(const RTCDevice device, const RTCError error,
                                       const std::string_view operation) {
    auto message = std::string{"Embree failed while "};
    message.append(operation);
    message.append(": ");
    const auto* const error_name = rtcGetErrorString(error);
    message.append(error_name != nullptr ? error_name : "unknown error");

    const auto* const detail = rtcGetDeviceLastErrorMessage(device);
    if (detail != nullptr && detail[0] != '\0') {
        message.append(" (");
        message.append(detail);
        message.push_back(')');
    }
    return core::Error{
        .code = embree_status_code(error),
        .message = std::move(message),
    };
}

[[nodiscard]] core::Status check_embree(const RTCDevice device, const std::string_view operation) {
    const auto error = rtcGetDeviceError(device);
    if (error == RTC_ERROR_NONE) {
        return {};
    }
    return std::unexpected(embree_error(device, error, operation));
}

[[nodiscard]] core::Error missing_embree_result(const core::StatusCode code,
                                                const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

struct EmbreeGeometry final {
    AccelGeometry descriptor;
    unsigned int embree_geometry_id{};
};

[[nodiscard]] RTCRay embree_ray(const renderer::Ray& ray) noexcept {
    auto result = RTCRay{};
    result.org_x = ray.origin().x;
    result.org_y = ray.origin().y;
    result.org_z = ray.origin().z;
    result.tnear = ray.t_min();
    result.dir_x = ray.direction().x;
    result.dir_y = ray.direction().y;
    result.dir_z = ray.direction().z;
    result.time = ray.time();
    result.tfar = ray.t_max();
    result.mask = ray.mask();
    result.id = 0U;
    result.flags = 0U;
    return result;
}

void initialize_hit(RTCHit& hit) noexcept {
    hit.geomID = RTC_INVALID_GEOMETRY_ID;
    hit.primID = RTC_INVALID_GEOMETRY_ID;
    std::fill(std::begin(hit.instID), std::end(hit.instID), RTC_INVALID_GEOMETRY_ID);
}

class EmbreeAccelBackend final : public AccelBackend {
  public:
    [[nodiscard]] static core::Result<std::unique_ptr<AccelBackend>>
    create(const std::span<const AccelGeometry> geometries) {
        const auto capability = renderer::require_backend_capability("cpu_embree");
        if (!capability) {
            return std::unexpected(capability.error());
        }

        const auto validation = validate_geometry_input(geometries);
        if (!validation) {
            return std::unexpected(validation.error());
        }

        try {
            static_cast<void>(rtcGetDeviceError(nullptr));
            auto device = DeviceHandle{rtcNewDevice(nullptr)};
            if (!device) {
                const auto error = rtcGetDeviceError(nullptr);
                if (error != RTC_ERROR_NONE) {
                    return std::unexpected(embree_error(nullptr, error, "creating the device"));
                }
                return std::unexpected(
                    missing_embree_result(core::StatusCode::platform_error,
                                          "Embree returned no device without reporting an error."));
            }
            if (auto status = check_embree(device.get(), "creating the device"); !status) {
                return std::unexpected(status.error());
            }

            const auto ray_masks =
                rtcGetDeviceProperty(device.get(), RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED);
            if (auto status = check_embree(device.get(), "querying ray-mask support"); !status) {
                return std::unexpected(status.error());
            }
            if (ray_masks == 0) {
                return std::unexpected(missing_embree_result(
                    core::StatusCode::unavailable,
                    "The selected Embree build does not support the required 32-bit ray masks."));
            }

            const auto triangles =
                rtcGetDeviceProperty(device.get(), RTC_DEVICE_PROPERTY_TRIANGLE_GEOMETRY_SUPPORTED);
            if (auto status = check_embree(device.get(), "querying triangle support"); !status) {
                return std::unexpected(status.error());
            }
            if (triangles == 0) {
                return std::unexpected(missing_embree_result(
                    core::StatusCode::unavailable,
                    "The selected Embree device does not support triangle geometry."));
            }

            const auto backface_culling =
                rtcGetDeviceProperty(device.get(), RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED);
            if (auto status = check_embree(device.get(), "querying backface-culling state");
                !status) {
                return std::unexpected(status.error());
            }
            if (backface_culling != 0) {
                return std::unexpected(missing_embree_result(
                    core::StatusCode::incompatible,
                    "The selected Embree build enables backface culling but Blackframe requires "
                    "two-sided triangle queries."));
            }

            auto scene = SceneHandle{rtcNewScene(device.get())};
            if (!scene) {
                if (auto status = check_embree(device.get(), "creating the scene"); !status) {
                    return std::unexpected(status.error());
                }
                return std::unexpected(
                    missing_embree_result(core::StatusCode::platform_error,
                                          "Embree returned no scene without reporting an error."));
            }
            if (auto status = check_embree(device.get(), "creating the scene"); !status) {
                return std::unexpected(status.error());
            }

            rtcSetSceneFlags(scene.get(), RTC_SCENE_FLAG_ROBUST);
            if (auto status = check_embree(device.get(), "enabling robust traversal"); !status) {
                return std::unexpected(status.error());
            }

            auto embree_geometries = std::vector<EmbreeGeometry>{};
            embree_geometries.reserve(geometries.size());
            for (const auto& descriptor : geometries) {
                auto geometry =
                    GeometryHandle{rtcNewGeometry(device.get(), RTC_GEOMETRY_TYPE_TRIANGLE)};
                if (!geometry) {
                    if (auto status = check_embree(device.get(), "creating triangle geometry");
                        !status) {
                        return std::unexpected(status.error());
                    }
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::platform_error,
                        "Embree returned no triangle geometry without reporting an error."));
                }
                if (auto status = check_embree(device.get(), "creating triangle geometry");
                    !status) {
                    return std::unexpected(status.error());
                }

                const auto positions = descriptor.mesh->positions();
                // The owned Embree buffer is required here: shared RTC_FORMAT_FLOAT3 buffers may
                // read 16 bytes for their final item, while Blackframe Point3 is compact at
                // 12 bytes. The owned index buffer below also keeps traversal storage independent
                // from the caller while the mesh itself is retained for exact hit reconstruction.
                auto* const vertex_buffer = rtcSetNewGeometryBuffer(
                    geometry.get(), RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                    sizeof(renderer::Point3), positions.size());
                if (auto status = check_embree(device.get(), "allocating the vertex buffer");
                    !status) {
                    return std::unexpected(status.error());
                }
                if (vertex_buffer == nullptr) {
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::platform_error,
                        "Embree returned no vertex buffer without reporting an error."));
                }
                std::memcpy(vertex_buffer, positions.data(), positions.size_bytes());

                const auto indices = descriptor.mesh->triangles();
                auto* const index_buffer = rtcSetNewGeometryBuffer(
                    geometry.get(), RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                    sizeof(TriangleVertexIndices), indices.size());
                if (auto status = check_embree(device.get(), "allocating the index buffer");
                    !status) {
                    return std::unexpected(status.error());
                }
                if (index_buffer == nullptr) {
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::platform_error,
                        "Embree returned no index buffer without reporting an error."));
                }
                std::memcpy(index_buffer, indices.data(), indices.size_bytes());

                rtcSetGeometryMask(geometry.get(), descriptor.visibility_mask);
                if (auto status = check_embree(device.get(), "setting the geometry mask");
                    !status) {
                    return std::unexpected(status.error());
                }

                rtcCommitGeometry(geometry.get());
                if (auto status = check_embree(device.get(), "committing triangle geometry");
                    !status) {
                    return std::unexpected(status.error());
                }

                const auto embree_geometry_id = rtcAttachGeometry(scene.get(), geometry.get());
                if (auto status = check_embree(device.get(), "attaching triangle geometry");
                    !status) {
                    return std::unexpected(status.error());
                }
                if (embree_geometry_id == RTC_INVALID_GEOMETRY_ID) {
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::platform_error,
                        "Embree returned no geometry identifier without reporting an error."));
                }

                embree_geometries.push_back(EmbreeGeometry{
                    .descriptor = descriptor,
                    .embree_geometry_id = embree_geometry_id,
                });
            }

            rtcCommitScene(scene.get());
            if (auto status = check_embree(device.get(), "committing the scene"); !status) {
                return std::unexpected(status.error());
            }

            return std::unique_ptr<AccelBackend>{new EmbreeAccelBackend{
                std::move(embree_geometries), std::move(device), std::move(scene)}};
        } catch (const std::bad_alloc&) {
            return std::unexpected(
                missing_embree_result(core::StatusCode::resource_exhausted,
                                      "Embree acceleration construction exhausted host memory."));
        } catch (const std::length_error&) {
            return std::unexpected(missing_embree_result(
                core::StatusCode::resource_exhausted,
                "Embree acceleration construction exceeded host container limits."));
        }
    }

    [[nodiscard]] AccelBackendKind kind() const noexcept override {
        return AccelBackendKind::embree;
    }

    [[nodiscard]] core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray& ray) const override {
        const auto validation = validate_ray(ray);
        if (!validation) {
            return std::unexpected(validation.error());
        }

        auto ray_hit = RTCRayHit{};
        ray_hit.ray = embree_ray(ray);
        initialize_hit(ray_hit.hit);
        auto arguments = RTCIntersectArguments{};
        rtcInitIntersectArguments(&arguments);
        rtcIntersect1(scene_.get(), &ray_hit, &arguments);
        if (auto status = check_embree(device_.get(), "tracing a closest-hit query"); !status) {
            return std::unexpected(status.error());
        }

        if (ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            if (ray_hit.hit.primID != RTC_INVALID_GEOMETRY_ID || ray_hit.ray.tfar != ray.t_max()) {
                return std::unexpected(missing_embree_result(
                    core::StatusCode::internal_error,
                    "Embree modified closest-hit output while reporting a miss."));
            }
            return std::optional<AccelHit>{};
        }
        const auto geometry = std::find_if(
            geometries_.begin(), geometries_.end(), [&ray_hit](const EmbreeGeometry& candidate) {
                return candidate.embree_geometry_id == ray_hit.hit.geomID;
            });
        if (geometry == geometries_.end()) {
            return std::unexpected(missing_embree_result(
                core::StatusCode::internal_error,
                "Embree returned an unknown geometry identifier for a closest hit."));
        }
        if (static_cast<std::size_t>(ray_hit.hit.primID) >=
            geometry->descriptor.mesh->triangles().size()) {
            return std::unexpected(missing_embree_result(
                core::StatusCode::internal_error,
                "Embree returned an out-of-range primitive identifier for a closest hit."));
        }
        if (!std::isfinite(ray_hit.ray.tfar) || !ray.contains_parameter(ray_hit.ray.tfar) ||
            !std::isfinite(ray_hit.hit.u) || !std::isfinite(ray_hit.hit.v)) {
            return std::unexpected(missing_embree_result(
                core::StatusCode::internal_error,
                "Embree returned non-finite or out-of-range closest-hit data."));
        }

        auto position = ray.at(ray_hit.ray.tfar);
        if (!position) {
            return std::unexpected(position.error());
        }
        auto triangle = geometry->descriptor.mesh->geometric_triangle(ray_hit.hit.primID);
        if (!triangle) {
            return std::unexpected(triangle.error());
        }
        const auto vertex0 = renderer::TransportScalar{1} - ray_hit.hit.u - ray_hit.hit.v;
        if (!std::isfinite(vertex0)) {
            return std::unexpected(
                missing_embree_result(core::StatusCode::internal_error,
                                      "Embree returned unrepresentable closest-hit barycentrics."));
        }

        return std::optional<AccelHit>{AccelHit{
            .triangle =
                renderer::TriangleHit{
                    .parameter = ray_hit.ray.tfar,
                    .position = *position,
                    .geometric_normal = triangle->geometric_normal(),
                    .barycentrics =
                        {
                            .vertex0 = vertex0,
                            .vertex1 = ray_hit.hit.u,
                            .vertex2 = ray_hit.hit.v,
                        },
                },
            .identifiers =
                {
                    .instance = geometry->descriptor.instance,
                    .geometry = geometry->descriptor.geometry,
                    .primitive = renderer::PrimitiveId{.value = ray_hit.hit.primID},
                    .material = geometry->descriptor.material,
                },
        }};
    }

    [[nodiscard]] core::Result<bool> occluded(const renderer::Ray& ray) const override {
        const auto validation = validate_ray(ray);
        if (!validation) {
            return std::unexpected(validation.error());
        }

        auto query_ray = embree_ray(ray);
        auto arguments = RTCOccludedArguments{};
        rtcInitOccludedArguments(&arguments);
        rtcOccluded1(scene_.get(), &query_ray, &arguments);
        if (auto status = check_embree(device_.get(), "tracing an occlusion query"); !status) {
            return std::unexpected(status.error());
        }
        if (std::isinf(query_ray.tfar) && std::signbit(query_ray.tfar)) {
            return true;
        }
        if (query_ray.tfar == ray.t_max()) {
            return false;
        }
        return std::unexpected(
            missing_embree_result(core::StatusCode::internal_error,
                                  "Embree returned an invalid occlusion-query result."));
    }

  private:
    EmbreeAccelBackend(std::vector<EmbreeGeometry>&& geometries, DeviceHandle&& device,
                       SceneHandle&& scene) noexcept
        : geometries_{std::move(geometries)}, device_{std::move(device)}, scene_{std::move(scene)} {
    }

    // Declaration order retains descriptors until after Embree releases the
    // scene and device, even though Embree owns copied traversal buffers.
    std::vector<EmbreeGeometry> geometries_;
    DeviceHandle device_;
    SceneHandle scene_;
};

} // namespace

core::Result<std::unique_ptr<AccelBackend>>
create_embree_accel_backend(const std::span<const AccelGeometry> geometries) {
    return EmbreeAccelBackend::create(geometries);
}

} // namespace blackframe::engine
