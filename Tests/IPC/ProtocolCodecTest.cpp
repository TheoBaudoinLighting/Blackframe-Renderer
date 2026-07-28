#include <Blackframe/IPC/ProtocolCodec.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace blackframe::ipc {
namespace {

TEST(ProtocolCodecTest, EncodesTheHeaderInExplicitLittleEndianOrder) {
    const auto request = MakeProtocolRequest(Command::Ping, UINT64_C(0x0102030405060708));

    const auto encoded = EncodeProtocolMessage(request);

    ASSERT_TRUE(encoded.has_value()) << encoded.error().message;
    const auto expected = std::array{
        std::byte{'B'},  std::byte{'F'},  std::byte{'I'},  std::byte{'P'},  std::byte{0x01},
        std::byte{0x00}, std::byte{0x01}, std::byte{0x00}, std::byte{0x01}, std::byte{0x00},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x08}, std::byte{0x07}, std::byte{0x06},
        std::byte{0x05}, std::byte{0x04}, std::byte{0x03}, std::byte{0x02}, std::byte{0x01},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
    };
    EXPECT_EQ(*encoded, std::vector<std::byte>(expected.begin(), expected.end()));
}

TEST(ProtocolCodecTest, RejectsBadMagicAndTruncatedFrames) {
    auto encoded = EncodeProtocolMessage(MakeProtocolRequest(Command::Ping, 1));
    ASSERT_TRUE(encoded.has_value()) << encoded.error().message;

    (*encoded)[0] = std::byte{'X'};
    auto bad_magic = DecodeProtocolMessage(*encoded);
    ASSERT_FALSE(bad_magic.has_value());
    EXPECT_EQ(bad_magic.error().code, ProtocolErrorCode::InvalidMagic);

    encoded->resize(EncodedProtocolHeaderSize - 1);
    auto truncated = DecodeProtocolMessage(*encoded);
    ASSERT_FALSE(truncated.has_value());
    EXPECT_EQ(truncated.error().code, ProtocolErrorCode::InsufficientData);
}

TEST(ProtocolCodecTest, RejectsPayloadsAboveTheProtocolLimit) {
    auto request = MakeProtocolRequest(Command::Ping, 1,
                                       std::vector<std::byte>(MaximumProtocolPayloadSize + 1));

    const auto encoded = EncodeProtocolMessage(request);

    ASSERT_FALSE(encoded.has_value());
    EXPECT_EQ(encoded.error().code, ProtocolErrorCode::PayloadTooLarge);
}

TEST(ProtocolCodecTest, RejectsUnknownResponseStatusValues) {
    auto encoded = EncodeProtocolMessage(
        MakeProtocolResponse(MakeProtocolRequest(Command::Ping, 1), Status::Ok));
    ASSERT_TRUE(encoded.has_value()) << encoded.error().message;
    (*encoded)[10] = std::byte{0xFF};
    (*encoded)[11] = std::byte{0x7F};

    const auto decoded = DecodeProtocolMessage(*encoded);

    ASSERT_FALSE(decoded.has_value());
    EXPECT_EQ(decoded.error().code, ProtocolErrorCode::InvalidStatus);
}

TEST(ProtocolCodecTest, RejectsUnsupportedVersionsAndReservedRequestStatus) {
    auto encoded = EncodeProtocolMessage(MakeProtocolRequest(Command::Ping, 1));
    ASSERT_TRUE(encoded.has_value()) << encoded.error().message;

    (*encoded)[4] = std::byte{0x02};
    auto unsupported_version = DecodeProtocolMessage(*encoded);
    ASSERT_FALSE(unsupported_version.has_value());
    EXPECT_EQ(unsupported_version.error().code, ProtocolErrorCode::UnsupportedVersion);

    (*encoded)[4] = std::byte{0x01};
    (*encoded)[10] = std::byte{0x01};
    auto reserved_request_status = DecodeProtocolMessage(*encoded);
    ASSERT_FALSE(reserved_request_status.has_value());
    EXPECT_EQ(reserved_request_status.error().code, ProtocolErrorCode::InvalidRequestStatus);
}

TEST(ProtocolCodecTest, RoundTripsBackendNeutralDeviceDescriptions) {
    const auto devices = std::array{
        DeviceDescription{
            .identifier = {std::byte{0x01}},
            .kind = DeviceKind::Cpu,
            .backend_name = "Reference CPU",
            .device_name = "Host CPU",
        },
        DeviceDescription{
            .identifier = {std::byte{0x02}},
            .kind = DeviceKind::Gpu,
            .backend_name = "Example GPU",
            .device_name = "Device 0",
        },
    };

    const auto encoded = EncodeDeviceDescriptions(devices);
    ASSERT_TRUE(encoded.has_value()) << encoded.error().message;
    const auto decoded = DecodeDeviceDescriptions(*encoded);

    ASSERT_TRUE(decoded.has_value()) << decoded.error().message;
    ASSERT_EQ(decoded->size(), devices.size());
    EXPECT_EQ(decoded->at(0).identifier, devices[0].identifier);
    EXPECT_EQ(decoded->at(0).kind, DeviceKind::Cpu);
    EXPECT_EQ(decoded->at(0).backend_name, "Reference CPU");
    EXPECT_EQ(decoded->at(0).device_name, "Host CPU");
    EXPECT_EQ(decoded->at(1).kind, DeviceKind::Gpu);
}

} // namespace
} // namespace blackframe::ipc
