#include <Blackframe/Renderer/CapabilityRegistry.hpp>
#include <gtest/gtest.h>
#include <set>
#include <string>

namespace blackframe::renderer {
namespace {

TEST(CapabilityRegistryTest, UsesTheStableMachineReadableStatusVocabulary) {
    EXPECT_EQ(backend_capability_status_name(BackendCapabilityStatus::supported), "supported");
    EXPECT_EQ(backend_capability_status_name(BackendCapabilityStatus::experimental),
              "experimental");
    EXPECT_EQ(backend_capability_status_name(BackendCapabilityStatus::unavailable), "unavailable");
}

TEST(CapabilityRegistryTest, RegistersEachBackendExactlyOnce) {
    const auto capabilities = backend_capabilities();
    ASSERT_EQ(capabilities.size(), 3U);

    auto identifiers = std::set<std::string>{};
    for (const auto& capability : capabilities) {
        EXPECT_FALSE(capability.identifier.empty());
        EXPECT_FALSE(capability.required_dependency.empty());
        EXPECT_TRUE(identifiers.emplace(capability.identifier).second);
    }

    EXPECT_TRUE(identifiers.contains("reference_cpu"));
    EXPECT_TRUE(identifiers.contains("cpu_embree"));
    EXPECT_TRUE(identifiers.contains("gpu_cuda"));
}

TEST(CapabilityRegistryTest, AcceptsEveryAvailableBackendWithoutSubstitution) {
    for (const auto& capability : backend_capabilities()) {
        if (capability.status == BackendCapabilityStatus::unavailable) {
            continue;
        }
        EXPECT_TRUE(require_backend_capability(capability.identifier).has_value());
    }
}

TEST(UnsupportedFeature, RejectsUnavailableBackendAndNamesMissingDependency) {
    for (const auto& capability : backend_capabilities()) {
        if (capability.status != BackendCapabilityStatus::unavailable) {
            continue;
        }

        const auto validation = require_backend_capability(capability.identifier);

        ASSERT_FALSE(validation.has_value());
        EXPECT_EQ(validation.error().code, core::StatusCode::unavailable);
        EXPECT_EQ(validation.error().message,
                  "Backend capability '" + std::string{capability.identifier} +
                      "' is unavailable; missing dependency '" +
                      std::string{capability.required_dependency} + "'.");
        return;
    }

    GTEST_SKIP() << "Every registered backend is available in this preset.";
}

TEST(UnsupportedFeature, RejectsUnregisteredBackendInsteadOfSelectingAFallback) {
    const auto validation = require_backend_capability("gpu_unregistered");

    ASSERT_FALSE(validation.has_value());
    EXPECT_EQ(validation.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(validation.error().message, "Unknown backend capability 'gpu_unregistered'.");
}

TEST(CapabilityRegistryTest, SerializesTheVersionedManifest) {
    const auto manifest = backend_capability_manifest();

    EXPECT_NE(manifest.find(R"("schema_version": 1)"), std::string::npos);
    for (const auto& capability : backend_capabilities()) {
        EXPECT_NE(manifest.find("\"id\": \"" + std::string{capability.identifier} + "\""),
                  std::string::npos);
        EXPECT_NE(manifest.find("\"status\": \"" +
                                std::string{backend_capability_status_name(capability.status)} +
                                "\""),
                  std::string::npos);
    }
}

} // namespace
} // namespace blackframe::renderer
