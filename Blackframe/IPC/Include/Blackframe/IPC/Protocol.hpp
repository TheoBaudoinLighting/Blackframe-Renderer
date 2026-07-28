#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace blackframe::ipc {

inline constexpr std::array<std::byte, 4> ProtocolMagic{
    static_cast<std::byte>('B'),
    static_cast<std::byte>('F'),
    static_cast<std::byte>('I'),
    static_cast<std::byte>('P'),
};

inline constexpr std::uint16_t CurrentProtocolVersion = 1;
inline constexpr std::uint16_t MinimumCompatibleProtocolVersion = 1;
inline constexpr std::size_t EncodedProtocolHeaderSize = 24;
inline constexpr std::uint32_t MaximumProtocolPayloadSize = 64U * 1024U;
inline constexpr std::size_t MaximumEncodedProtocolMessageSize =
    EncodedProtocolHeaderSize + MaximumProtocolPayloadSize;

enum class MessageType : std::uint16_t {
    Request = 1,
    Response = 2,
};

enum class Command : std::uint16_t {
    Ping = 1,
    QueryVersion = 2,
    EnumerateDevices = 3,
    Shutdown = 4,
};

enum class Status : std::uint16_t {
    Ok = 0,
    InvalidMessage = 1,
    UnsupportedVersion = 2,
    UnsupportedCommand = 3,
    InvalidPayload = 4,
    InternalError = 5,
    Unavailable = 6,
};

enum class DeviceKind : std::uint16_t {
    Unknown = 0,
    Cpu = 1,
    Gpu = 2,
    Accelerator = 3,
};

// Semantic representation of the fixed 24-byte wire header. The structure
// itself is never copied to or from the wire.
struct ProtocolHeader {
    std::uint16_t version{CurrentProtocolVersion};
    MessageType type{MessageType::Request};
    Command command{Command::Ping};
    Status status{Status::Ok};
    std::uint64_t request_id{};
    std::uint32_t payload_size{};
};

struct ProtocolMessage {
    std::uint16_t version{CurrentProtocolVersion};
    MessageType type{MessageType::Request};
    Command command{Command::Ping};
    Status status{Status::Ok};
    std::uint64_t request_id{};
    std::vector<std::byte> payload;
};

// QueryVersion response payload:
//   uint16 protocol_version
//   uint16 minimum_compatible_version
//   uint32 maximum_payload_size
struct VersionInformation {
    std::uint16_t protocol_version{CurrentProtocolVersion};
    std::uint16_t minimum_compatible_version{MinimumCompatibleProtocolVersion};
    std::uint32_t maximum_payload_size{MaximumProtocolPayloadSize};
};

// EnumerateDevices response records are deliberately backend-neutral. Names
// are UTF-8 byte strings and identifiers are opaque stable bytes.
struct DeviceDescription {
    std::array<std::byte, 16> identifier{};
    DeviceKind kind{DeviceKind::Unknown};
    std::string backend_name;
    std::string device_name;
};

[[nodiscard]] constexpr bool IsKnownCommand(const Command command) noexcept {
    switch (command) {
    case Command::Ping:
    case Command::QueryVersion:
    case Command::EnumerateDevices:
    case Command::Shutdown:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool IsKnownStatus(const Status status) noexcept {
    switch (status) {
    case Status::Ok:
    case Status::InvalidMessage:
    case Status::UnsupportedVersion:
    case Status::UnsupportedCommand:
    case Status::InvalidPayload:
    case Status::InternalError:
    case Status::Unavailable:
        return true;
    }
    return false;
}

} // namespace blackframe::ipc
