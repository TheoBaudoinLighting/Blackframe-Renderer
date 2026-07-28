#include <Blackframe/Engine/Engine.hpp>
#include <Blackframe/IPC/ProtocolCodec.hpp>
#include <filesystem>
#include <gtest/gtest.h>

namespace blackframe::engine {
namespace {

TEST(EngineTest, HandlesTheFoundationControlCommandsWithoutRendering) {
    const auto engine = Engine{};

    const auto ping = engine.handle_request(ipc::MakeProtocolRequest(ipc::Command::Ping, 1));
    ASSERT_TRUE(ping.has_value()) << ping.error().message;
    EXPECT_EQ(ping->status, ipc::Status::Ok);
    EXPECT_FALSE(ping->stop_server_after_response);

    const auto version =
        engine.handle_request(ipc::MakeProtocolRequest(ipc::Command::QueryVersion, 2));
    ASSERT_TRUE(version.has_value()) << version.error().message;
    EXPECT_TRUE(ipc::DecodeVersionInformation(version->payload).has_value());

    const auto shutdown =
        engine.handle_request(ipc::MakeProtocolRequest(ipc::Command::Shutdown, 3));
    ASSERT_TRUE(shutdown.has_value()) << shutdown.error().message;
    EXPECT_TRUE(shutdown->stop_server_after_response);
}

TEST(EngineTest, OwnsAnExplicitXpuDsoAndEnumeratesItsDevice) {
    auto engine = Engine{};
    const auto plugin_path = std::filesystem::path{BLACKFRAME_REFERENCE_XPU_PLUGIN_PATH};

    const auto loaded = engine.load_xpu_extension(plugin_path);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    EXPECT_EQ(engine.xpu_backend_count(), 1U);

    const auto devices = engine.enumerate_devices();
    ASSERT_TRUE(devices.has_value()) << devices.error().message;
    ASSERT_EQ(devices->size(), 1U);
    EXPECT_EQ(devices->front().name, "Host CPU");

    const auto duplicate = engine.load_xpu_extension(plugin_path);
    ASSERT_FALSE(duplicate.has_value());
    EXPECT_EQ(duplicate.error().code, core::StatusCode::invalid_argument);
}

TEST(EngineTest, EncodesEnumeratedDevicesForIpcClients) {
    auto engine = Engine{};
    ASSERT_TRUE(
        engine.load_xpu_extension(std::filesystem::path{BLACKFRAME_REFERENCE_XPU_PLUGIN_PATH})
            .has_value());

    const auto response =
        engine.handle_request(ipc::MakeProtocolRequest(ipc::Command::EnumerateDevices, 7));
    ASSERT_TRUE(response.has_value()) << response.error().message;
    const auto devices = ipc::DecodeDeviceDescriptions(response->payload);

    ASSERT_TRUE(devices.has_value()) << devices.error().message;
    ASSERT_EQ(devices->size(), 1U);
    EXPECT_EQ(devices->front().kind, ipc::DeviceKind::Cpu);
    EXPECT_EQ(devices->front().backend_name, "Reference CPU");
    EXPECT_EQ(devices->front().device_name, "Host CPU");
}

} // namespace
} // namespace blackframe::engine
