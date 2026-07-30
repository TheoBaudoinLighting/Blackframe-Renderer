#include "../../../Engine/AccelBackendContract.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/CapabilityRegistry.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::engine {
namespace {

static_assert(std::is_same_v<decltype(&create_analytic_accel_backend), AccelBackendFactory>);
static_assert(std::is_same_v<decltype(&create_embree_accel_backend), AccelBackendFactory>);

TEST(AccelBackendSelectionTest, SelectsBothImplementationsThroughTheSameInterface) {
    auto geometries = test::make_geometry_set();
    auto retained_meshes = std::vector<std::weak_ptr<const TriangleMesh>>{};
    retained_meshes.reserve(geometries.size());
    for (const auto& geometry : geometries) {
        retained_meshes.emplace_back(geometry.mesh);
    }
    const auto capability = renderer::require_backend_capability("cpu_embree");
    ASSERT_TRUE(capability.has_value()) << capability.error().message;

    const AccelBackendFactory analytic_factory = &create_analytic_accel_backend;
    const AccelBackendFactory embree_factory = &create_embree_accel_backend;
    auto analytic = analytic_factory(geometries);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto embree = embree_factory(geometries);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    for (auto& geometry : geometries) {
        geometry.mesh.reset();
    }
    for (const auto& mesh : retained_meshes) {
        EXPECT_FALSE(mesh.expired());
    }

    test::expect_backend_contract(**analytic, AccelBackendKind::analytic_reference);
    test::expect_backend_contract(**embree, AccelBackendKind::embree);

    analytic->reset();
    for (const auto& mesh : retained_meshes) {
        EXPECT_FALSE(mesh.expired());
    }
    embree->reset();
    for (const auto& mesh : retained_meshes) {
        EXPECT_TRUE(mesh.expired());
    }
}

TEST(AccelBackendSelectionTest, BothFactoriesRepresentAnEmptyWorldExplicitly) {
    const auto empty = std::span<const AccelGeometry>{};
    auto analytic = create_analytic_accel_backend(empty);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto embree = create_embree_accel_backend(empty);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    test::expect_empty_backend(**analytic, AccelBackendKind::analytic_reference);
    test::expect_empty_backend(**embree, AccelBackendKind::embree);
}

} // namespace
} // namespace blackframe::engine
