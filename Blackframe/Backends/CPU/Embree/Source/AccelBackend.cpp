#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Renderer/CapabilityRegistry.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <algorithm>
#include <array>
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

[[nodiscard]] std::array<float, 12>
embree_transform(const renderer::AffineTransform& transform) noexcept {
    const auto& matrix = transform.matrix();
    return {
        matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3), matrix(1, 0), matrix(1, 1),
        matrix(1, 2), matrix(1, 3), matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3),
    };
}

struct EmbreeMesh final {
    renderer::GeometryId geometry{};
    std::shared_ptr<const TriangleMesh> mesh;
    SceneHandle scene;
    unsigned int embree_geometry_id{};
};

struct EmbreeInstanceMapping final {
    std::size_t instance_index{};
    std::size_t mesh_index{};
    unsigned int embree_instance_id{};
};

class EmbreeAccelBackend final : public AccelBackend {
  public:
    [[nodiscard]] static core::Result<std::unique_ptr<AccelBackend>>
    create(FrameSceneHandle frame_scene) {
        const auto capability = renderer::require_backend_capability("cpu_embree");
        if (!capability) {
            return std::unexpected(capability.error());
        }

        auto instances = prepare_instances(frame_scene);
        if (!instances) {
            return std::unexpected(std::move(instances.error()));
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

            auto meshes = std::vector<EmbreeMesh>{};
            meshes.reserve(instances->size());
            for (const auto& instance : *instances) {
                const auto existing = std::ranges::find_if(meshes, [&instance](const auto& mesh) {
                    return mesh.geometry == instance.geometry;
                });
                if (existing != meshes.end()) {
                    if (existing->mesh.get() != instance.mesh.get()) {
                        return std::unexpected(missing_embree_result(
                            core::StatusCode::internal_error,
                            "One frame geometry identifier resolved to multiple triangle meshes."));
                    }
                    continue;
                }

                auto mesh_scene = SceneHandle{rtcNewScene(device.get())};
                if (!mesh_scene) {
                    if (auto status = check_embree(device.get(), "creating a mesh scene");
                        !status) {
                        return std::unexpected(status.error());
                    }
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::platform_error,
                        "Embree returned no mesh scene without reporting an error."));
                }
                if (auto status = check_embree(device.get(), "creating a mesh scene"); !status) {
                    return std::unexpected(status.error());
                }
                rtcSetSceneFlags(mesh_scene.get(), RTC_SCENE_FLAG_ROBUST);
                if (auto status = check_embree(device.get(), "enabling robust mesh traversal");
                    !status) {
                    return std::unexpected(status.error());
                }

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

                const auto positions = instance.mesh->positions();
                // The owned Embree buffer is required here: shared RTC_FORMAT_FLOAT3 buffers may
                // read 16 bytes for their final item, while Blackframe Point3 is compact at
                // 12 bytes. The owned index buffer below also keeps traversal storage independent
                // from the caller while the immutable frame scene retains the source mesh.
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

                const auto indices = instance.mesh->triangles();
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

                rtcSetGeometryMask(geometry.get(), renderer::AllRayVisibility);
                if (auto status = check_embree(device.get(), "setting the triangle geometry mask");
                    !status) {
                    return std::unexpected(status.error());
                }
                rtcCommitGeometry(geometry.get());
                if (auto status = check_embree(device.get(), "committing triangle geometry");
                    !status) {
                    return std::unexpected(status.error());
                }

                const auto embree_geometry_id = rtcAttachGeometry(mesh_scene.get(), geometry.get());
                if (auto status = check_embree(device.get(), "attaching triangle geometry");
                    !status) {
                    return std::unexpected(status.error());
                }
                if (embree_geometry_id == RTC_INVALID_GEOMETRY_ID) {
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::platform_error,
                        "Embree returned no geometry identifier without reporting an error."));
                }
                rtcCommitScene(mesh_scene.get());
                if (auto status = check_embree(device.get(), "committing a mesh scene"); !status) {
                    return std::unexpected(status.error());
                }

                meshes.push_back(EmbreeMesh{
                    .geometry = instance.geometry,
                    .mesh = instance.mesh,
                    .scene = std::move(mesh_scene),
                    .embree_geometry_id = embree_geometry_id,
                });
            }

            auto top_scene = SceneHandle{rtcNewScene(device.get())};
            if (!top_scene) {
                if (auto status = check_embree(device.get(), "creating the top-level scene");
                    !status) {
                    return std::unexpected(status.error());
                }
                return std::unexpected(missing_embree_result(
                    core::StatusCode::platform_error,
                    "Embree returned no top-level scene without reporting an error."));
            }
            if (auto status = check_embree(device.get(), "creating the top-level scene"); !status) {
                return std::unexpected(status.error());
            }
            rtcSetSceneFlags(top_scene.get(), RTC_SCENE_FLAG_ROBUST);
            if (auto status = check_embree(device.get(), "enabling robust instance traversal");
                !status) {
                return std::unexpected(status.error());
            }

            auto mappings = std::vector<EmbreeInstanceMapping>{};
            mappings.reserve(instances->size());
            for (auto instance_index = std::size_t{}; instance_index < instances->size();
                 ++instance_index) {
                const auto& instance = (*instances)[instance_index];
                const auto mesh = std::ranges::find_if(meshes, [&instance](const auto& candidate) {
                    return candidate.geometry == instance.geometry;
                });
                if (mesh == meshes.end()) {
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::internal_error,
                        "Embree instance construction lost a prepared mesh scene."));
                }

                auto geometry =
                    GeometryHandle{rtcNewGeometry(device.get(), RTC_GEOMETRY_TYPE_INSTANCE)};
                if (!geometry) {
                    if (auto status = check_embree(device.get(), "creating instance geometry");
                        !status) {
                        return std::unexpected(status.error());
                    }
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::unavailable,
                        "The selected Embree build does not support instance geometry."));
                }
                if (auto status = check_embree(device.get(), "creating instance geometry");
                    !status) {
                    return std::unexpected(status.error());
                }

                rtcSetGeometryInstancedScene(geometry.get(), mesh->scene.get());
                if (auto status = check_embree(device.get(), "assigning an instanced mesh scene");
                    !status) {
                    return std::unexpected(status.error());
                }
                const auto transform = embree_transform(instance.object_to_world);
                rtcSetGeometryTransform(geometry.get(), 0U, RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                                        transform.data());
                if (auto status = check_embree(device.get(), "setting an instance transform");
                    !status) {
                    return std::unexpected(status.error());
                }
                rtcSetGeometryMask(geometry.get(), instance.visibility_mask);
                if (auto status = check_embree(device.get(), "setting an instance mask"); !status) {
                    return std::unexpected(status.error());
                }
                rtcCommitGeometry(geometry.get());
                if (auto status = check_embree(device.get(), "committing instance geometry");
                    !status) {
                    return std::unexpected(status.error());
                }

                const auto embree_instance_id = rtcAttachGeometry(top_scene.get(), geometry.get());
                if (auto status = check_embree(device.get(), "attaching instance geometry");
                    !status) {
                    return std::unexpected(status.error());
                }
                if (embree_instance_id == RTC_INVALID_GEOMETRY_ID) {
                    return std::unexpected(missing_embree_result(
                        core::StatusCode::platform_error,
                        "Embree returned no instance identifier without reporting an error."));
                }
                mappings.push_back(EmbreeInstanceMapping{
                    .instance_index = instance_index,
                    .mesh_index = static_cast<std::size_t>(mesh - meshes.begin()),
                    .embree_instance_id = embree_instance_id,
                });
            }

            rtcCommitScene(top_scene.get());
            if (auto status = check_embree(device.get(), "committing the top-level scene");
                !status) {
                return std::unexpected(status.error());
            }

            return std::unique_ptr<AccelBackend>{new EmbreeAccelBackend{
                std::move(frame_scene), std::move(*instances), std::move(device), std::move(meshes),
                std::move(mappings), std::move(top_scene)}};
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
            const auto instance_ids_unchanged =
                std::ranges::all_of(ray_hit.hit.instID, [](const auto instance_id) {
                    return instance_id == RTC_INVALID_GEOMETRY_ID;
                });
            if (ray_hit.hit.primID != RTC_INVALID_GEOMETRY_ID || !instance_ids_unchanged ||
                ray_hit.ray.tfar != ray.t_max()) {
                return std::unexpected(missing_embree_result(
                    core::StatusCode::internal_error,
                    "Embree modified closest-hit output while reporting a miss."));
            }
            return std::optional<AccelHit>{};
        }

        if (ray_hit.hit.instID[0] == RTC_INVALID_GEOMETRY_ID) {
            return std::unexpected(missing_embree_result(
                core::StatusCode::internal_error,
                "Embree returned a direct hit in an instance-only top-level scene."));
        }
        for (auto level = std::size_t{1}; level < std::size(ray_hit.hit.instID); ++level) {
            if (ray_hit.hit.instID[level] != RTC_INVALID_GEOMETRY_ID) {
                return std::unexpected(missing_embree_result(
                    core::StatusCode::internal_error,
                    "Embree returned an unexpected nested instance identifier."));
            }
        }

        const auto mapping =
            std::ranges::find_if(mappings_, [&ray_hit](const EmbreeInstanceMapping& candidate) {
                return candidate.embree_instance_id == ray_hit.hit.instID[0];
            });
        if (mapping == mappings_.end() || mapping->instance_index >= instances_.size() ||
            mapping->mesh_index >= meshes_.size()) {
            return std::unexpected(
                missing_embree_result(core::StatusCode::internal_error,
                                      "Embree returned an unknown top-level instance identifier."));
        }
        const auto& instance = instances_[mapping->instance_index];
        const auto& mesh = meshes_[mapping->mesh_index];
        if (mesh.geometry != instance.geometry || ray_hit.hit.geomID != mesh.embree_geometry_id) {
            return std::unexpected(missing_embree_result(
                core::StatusCode::internal_error,
                "Embree returned a geometry identifier outside the hit instance mesh scene."));
        }
        if (static_cast<std::size_t>(ray_hit.hit.primID) >= instance.mesh->triangles().size()) {
            return std::unexpected(missing_embree_result(
                core::StatusCode::internal_error,
                "Embree returned an out-of-range primitive identifier for a closest hit."));
        }
        const auto embree_normal_is_finite = std::isfinite(ray_hit.hit.Ng_x) &&
                                             std::isfinite(ray_hit.hit.Ng_y) &&
                                             std::isfinite(ray_hit.hit.Ng_z);
        const auto embree_normal_is_nonzero =
            ray_hit.hit.Ng_x != 0.0F || ray_hit.hit.Ng_y != 0.0F || ray_hit.hit.Ng_z != 0.0F;
        if (!std::isfinite(ray_hit.ray.tfar) || !ray.contains_parameter(ray_hit.ray.tfar) ||
            !std::isfinite(ray_hit.hit.u) || !std::isfinite(ray_hit.hit.v) ||
            !embree_normal_is_finite || !embree_normal_is_nonzero) {
            return std::unexpected(missing_embree_result(
                core::StatusCode::internal_error,
                "Embree returned non-finite or out-of-range closest-hit data."));
        }

        auto position = ray.at(ray_hit.ray.tfar);
        if (!position) {
            return std::unexpected(position.error());
        }
        auto triangle = world_space_triangle(instance, ray_hit.hit.primID);
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
            .object = instance.object,
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
                    .instance = instance.instance,
                    .geometry = instance.geometry,
                    .primitive = renderer::PrimitiveId{.value = ray_hit.hit.primID},
                    .material = instance.material,
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
    EmbreeAccelBackend(FrameSceneHandle&& frame_scene, std::vector<AccelInstance>&& instances,
                       DeviceHandle&& device, std::vector<EmbreeMesh>&& meshes,
                       std::vector<EmbreeInstanceMapping>&& mappings, SceneHandle&& scene) noexcept
        : AccelBackend{std::move(frame_scene)}, instances_{std::move(instances)},
          device_{std::move(device)}, meshes_{std::move(meshes)}, mappings_{std::move(mappings)},
          scene_{std::move(scene)} {}

    // Declaration order releases the TLAS before its BLAS scenes and releases
    // every Embree scene before the shared device. The base snapshot outlives
    // all derived members.
    std::vector<AccelInstance> instances_;
    DeviceHandle device_;
    std::vector<EmbreeMesh> meshes_;
    std::vector<EmbreeInstanceMapping> mappings_;
    SceneHandle scene_;
};

} // namespace

core::Result<std::unique_ptr<AccelBackend>> create_embree_accel_backend(FrameSceneHandle scene) {
    return EmbreeAccelBackend::create(std::move(scene));
}

} // namespace blackframe::engine
