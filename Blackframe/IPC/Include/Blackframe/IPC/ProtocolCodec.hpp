#pragma once

#include <Blackframe/IPC/Protocol.hpp>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace blackframe::ipc {

enum class ProtocolErrorCode {
    InsufficientData,
    InvalidMagic,
    UnsupportedVersion,
    InvalidMessageType,
    InvalidStatus,
    InvalidRequestStatus,
    InvalidRequestIdentifier,
    PayloadTooLarge,
    PayloadSizeMismatch,
    InvalidPayload,
};

struct ProtocolError {
    ProtocolErrorCode code{ProtocolErrorCode::InvalidPayload};
    std::string message;
};

// All integer fields are encoded explicitly in little-endian order. These
// functions never serialize the native representation of a C++ object.
[[nodiscard]] std::expected<std::vector<std::byte>, ProtocolError>
EncodeProtocolMessage(const ProtocolMessage& message);

// Decodes exactly the first EncodedProtocolHeaderSize bytes. Extra bytes are
// ignored so transports can validate a header before reading its payload.
[[nodiscard]] std::expected<ProtocolHeader, ProtocolError>
DecodeProtocolHeader(std::span<const std::byte> bytes);

// Decodes exactly one complete frame. Truncated and trailing data are errors.
[[nodiscard]] std::expected<ProtocolMessage, ProtocolError>
DecodeProtocolMessage(std::span<const std::byte> bytes);

[[nodiscard]] ProtocolMessage MakeProtocolRequest(Command command, std::uint64_t request_id,
                                                  std::vector<std::byte> payload = {});

[[nodiscard]] ProtocolMessage MakeProtocolResponse(const ProtocolMessage& request, Status status,
                                                   std::vector<std::byte> payload = {});

[[nodiscard]] std::expected<std::vector<std::byte>, ProtocolError>
EncodeVersionInformation(const VersionInformation& information);

[[nodiscard]] std::expected<VersionInformation, ProtocolError>
DecodeVersionInformation(std::span<const std::byte> payload);

// Device-list wire payload:
//   uint32 record_count
//   repeated:
//     byte[16] identifier
//     uint16 device_kind
//     uint16 backend_name_size
//     uint16 device_name_size
//     byte[backend_name_size] UTF-8 backend name
//     byte[device_name_size] UTF-8 device name
[[nodiscard]] std::expected<std::vector<std::byte>, ProtocolError>
EncodeDeviceDescriptions(std::span<const DeviceDescription> devices);

[[nodiscard]] std::expected<std::vector<DeviceDescription>, ProtocolError>
DecodeDeviceDescriptions(std::span<const std::byte> payload);

} // namespace blackframe::ipc
