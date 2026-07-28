#include <cstdint>
#include <embree4/rtcore.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <type_traits>

namespace {

struct DeviceDeleter {
    void operator()(RTCDevice device) const noexcept {
        rtcReleaseDevice(device);
    }
};

struct SceneDeleter {
    void operator()(RTCScene scene) const noexcept {
        rtcReleaseScene(scene);
    }
};

struct GeometryDeleter {
    void operator()(RTCGeometry geometry) const noexcept {
        rtcReleaseGeometry(geometry);
    }
};

using DeviceHandle = std::unique_ptr<std::remove_pointer_t<RTCDevice>, DeviceDeleter>;
using SceneHandle = std::unique_ptr<std::remove_pointer_t<RTCScene>, SceneDeleter>;
using GeometryHandle = std::unique_ptr<std::remove_pointer_t<RTCGeometry>, GeometryDeleter>;

struct Vertex {
    float x;
    float y;
    float z;
};

struct Triangle {
    std::uint32_t vertex_0;
    std::uint32_t vertex_1;
    std::uint32_t vertex_2;
};

TEST(EmbreeSmokeTest, CreatesADeviceAndTracesARayAgainstATriangle) {
    DeviceHandle device{rtcNewDevice(nullptr)};
    ASSERT_NE(device, nullptr);

    SceneHandle scene{rtcNewScene(device.get())};
    ASSERT_NE(scene, nullptr);

    GeometryHandle geometry{rtcNewGeometry(device.get(), RTC_GEOMETRY_TYPE_TRIANGLE)};
    ASSERT_NE(geometry, nullptr);

    auto* vertices = static_cast<Vertex*>(rtcSetNewGeometryBuffer(
        geometry.get(), RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 3));
    ASSERT_NE(vertices, nullptr);
    vertices[0] = Vertex{0.0F, 0.0F, 0.0F};
    vertices[1] = Vertex{1.0F, 0.0F, 0.0F};
    vertices[2] = Vertex{0.0F, 1.0F, 0.0F};

    auto* triangles = static_cast<Triangle*>(rtcSetNewGeometryBuffer(
        geometry.get(), RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 1));
    ASSERT_NE(triangles, nullptr);
    triangles[0] = Triangle{0, 1, 2};

    rtcCommitGeometry(geometry.get());
    const auto geometry_id = rtcAttachGeometry(scene.get(), geometry.get());
    ASSERT_NE(geometry_id, RTC_INVALID_GEOMETRY_ID);
    geometry.reset();
    rtcCommitScene(scene.get());

    RTCRayHit ray_hit{};
    ray_hit.ray.org_x = 0.25F;
    ray_hit.ray.org_y = 0.25F;
    ray_hit.ray.org_z = -1.0F;
    ray_hit.ray.dir_x = 0.0F;
    ray_hit.ray.dir_y = 0.0F;
    ray_hit.ray.dir_z = 1.0F;
    ray_hit.ray.tnear = 0.0F;
    ray_hit.ray.tfar = std::numeric_limits<float>::infinity();
    ray_hit.ray.mask = std::numeric_limits<std::uint32_t>::max();
    ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    RTCIntersectArguments arguments{};
    rtcInitIntersectArguments(&arguments);
    rtcIntersect1(scene.get(), &ray_hit, &arguments);

    EXPECT_EQ(ray_hit.hit.geomID, geometry_id);
    EXPECT_EQ(ray_hit.hit.primID, 0U);
    EXPECT_FLOAT_EQ(ray_hit.ray.tfar, 1.0F);
    EXPECT_FLOAT_EQ(ray_hit.hit.u, 0.25F);
    EXPECT_FLOAT_EQ(ray_hit.hit.v, 0.25F);
    EXPECT_EQ(rtcGetDeviceError(device.get()), RTC_ERROR_NONE);
}

} // namespace
