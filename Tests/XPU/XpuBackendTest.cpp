#include <Blackframe/XPU/XpuBackend.hpp>
#include <filesystem>
#include <gtest/gtest.h>

namespace blackframe::xpu {
namespace {

TEST(XpuBackendTest, LoadsTheReferenceCpuDsoFromAnExplicitPath) {
    const auto plugin_path = std::filesystem::path{BLACKFRAME_REFERENCE_XPU_PLUGIN_PATH};

    auto backend = XpuBackend::load(plugin_path);

    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    EXPECT_EQ(backend->name(), "Reference CPU");

    auto devices = backend->enumerate_devices();
    ASSERT_TRUE(devices.has_value()) << devices.error().message;
    ASSERT_EQ(devices->size(), 1U);
    EXPECT_EQ(devices->front().kind, DeviceKind::Cpu);
    EXPECT_EQ(devices->front().name, "Host CPU");
    EXPECT_EQ(devices->front().vendor, "Blackframe");
}

} // namespace
} // namespace blackframe::xpu
