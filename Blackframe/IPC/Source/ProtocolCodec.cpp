#include <Blackframe/IPC/ProtocolCodec.hpp>
#include <algorithm>
#include <limits>
#include <utility>

namespace blackframe::ipc {
namespace {

[[nodiscard]] ProtocolError MakeError(const ProtocolErrorCode code, std::string message) {
    return ProtocolError{code, std::move(message)};
}

void AppendUnsigned16(std::vector<std::byte>& output, const std::uint16_t value) {
    output.push_back(static_cast<std::byte>(value & 0xFFU));
    output.push_back(static_cast<std::byte>((value >> 8U) & 0xFFU));
}

void AppendUnsigned32(std::vector<std::byte>& output, const std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xFFU));
    }
}

void AppendUnsigned64(std::vector<std::byte>& output, const std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xFFU));
    }
}

[[nodiscard]] std::uint16_t ReadUnsigned16(const std::span<const std::byte> bytes,
                                           const std::size_t offset) noexcept {
    return static_cast<std::uint16_t>(std::to_integer<std::uint16_t>(bytes[offset]) |
                                      (std::to_integer<std::uint16_t>(bytes[offset + 1]) << 8U));
}

[[nodiscard]] std::uint32_t ReadUnsigned32(const std::span<const std::byte> bytes,
                                           const std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index) {
        value |= std::to_integer<std::uint32_t>(bytes[offset + index]) << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::uint64_t ReadUnsigned64(const std::span<const std::byte> bytes,
                                           const std::size_t offset) noexcept {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= std::to_integer<std::uint64_t>(bytes[offset + index]) << (index * 8U);
    }
    return value;
}

[[nodiscard]] bool IsValidMessageType(const MessageType type) noexcept {
    return type == MessageType::Request || type == MessageType::Response;
}

[[nodiscard]] std::expected<void, ProtocolError> ValidateHeader(const ProtocolHeader& header) {
    if (header.version != CurrentProtocolVersion) {
        return std::unexpected(MakeError(ProtocolErrorCode::UnsupportedVersion,
                                         "The frame uses an unsupported protocol version."));
    }
    if (!IsValidMessageType(header.type)) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidMessageType,
                                         "The frame contains an unknown message type."));
    }
    if (!IsKnownStatus(header.status)) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidStatus,
                                         "The frame contains an unknown status value."));
    }
    if (header.type == MessageType::Request && header.status != Status::Ok) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidRequestStatus,
                                         "A request must carry the Ok status value."));
    }
    if (header.request_id == 0) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidRequestIdentifier,
                                         "Request identifier zero is reserved."));
    }
    if (header.payload_size > MaximumProtocolPayloadSize) {
        return std::unexpected(MakeError(ProtocolErrorCode::PayloadTooLarge,
                                         "The frame payload exceeds the 64 KiB protocol limit."));
    }
    return {};
}

[[nodiscard]] std::expected<void, ProtocolError>
ValidateRequestPayloadContract(const ProtocolMessage& message) {
    if (message.type != MessageType::Request || !IsKnownCommand(message.command)) {
        return {};
    }

    // Every command introduced by protocol v1 has an empty request payload.
    if (!message.payload.empty()) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidPayload,
                                         "Protocol v1 commands require an empty request payload."));
    }
    return {};
}

void AppendStringBytes(std::vector<std::byte>& output, const std::string& value) {
    for (const char character : value) {
        output.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
    }
}

[[nodiscard]] std::string ReadStringBytes(const std::span<const std::byte> bytes) {
    std::string result;
    result.reserve(bytes.size());
    for (const std::byte value : bytes) {
        result.push_back(static_cast<char>(std::to_integer<unsigned char>(value)));
    }
    return result;
}

} // namespace

std::expected<std::vector<std::byte>, ProtocolError>
EncodeProtocolMessage(const ProtocolMessage& message) {
    if (message.payload.size() > MaximumProtocolPayloadSize) {
        return std::unexpected(MakeError(ProtocolErrorCode::PayloadTooLarge,
                                         "The frame payload exceeds the 64 KiB protocol limit."));
    }

    const auto payload_size = static_cast<std::uint32_t>(message.payload.size());
    const ProtocolHeader header{
        .version = message.version,
        .type = message.type,
        .command = message.command,
        .status = message.status,
        .request_id = message.request_id,
        .payload_size = payload_size,
    };

    if (auto validation = ValidateHeader(header); !validation) {
        return std::unexpected(std::move(validation.error()));
    }
    if (auto validation = ValidateRequestPayloadContract(message); !validation) {
        return std::unexpected(std::move(validation.error()));
    }

    std::vector<std::byte> encoded;
    encoded.reserve(EncodedProtocolHeaderSize + message.payload.size());
    for (const std::byte magic_byte : ProtocolMagic) {
        encoded.push_back(magic_byte);
    }
    AppendUnsigned16(encoded, header.version);
    AppendUnsigned16(encoded, static_cast<std::uint16_t>(header.type));
    AppendUnsigned16(encoded, static_cast<std::uint16_t>(header.command));
    AppendUnsigned16(encoded, static_cast<std::uint16_t>(header.status));
    AppendUnsigned64(encoded, header.request_id);
    AppendUnsigned32(encoded, header.payload_size);
    for (const std::byte payload_byte : message.payload) {
        encoded.push_back(payload_byte);
    }
    return encoded;
}

std::expected<ProtocolHeader, ProtocolError>
DecodeProtocolHeader(const std::span<const std::byte> bytes) {
    if (bytes.size() < EncodedProtocolHeaderSize) {
        return std::unexpected(
            MakeError(ProtocolErrorCode::InsufficientData,
                      "Fewer than 24 bytes are available for the protocol header."));
    }

    for (std::size_t index = 0; index < ProtocolMagic.size(); ++index) {
        if (bytes[index] != ProtocolMagic[index]) {
            return std::unexpected(MakeError(ProtocolErrorCode::InvalidMagic,
                                             "The frame does not start with the BFIP magic."));
        }
    }

    const ProtocolHeader header{
        .version = ReadUnsigned16(bytes, 4),
        .type = static_cast<MessageType>(ReadUnsigned16(bytes, 6)),
        .command = static_cast<Command>(ReadUnsigned16(bytes, 8)),
        .status = static_cast<Status>(ReadUnsigned16(bytes, 10)),
        .request_id = ReadUnsigned64(bytes, 12),
        .payload_size = ReadUnsigned32(bytes, 20),
    };

    if (auto validation = ValidateHeader(header); !validation) {
        return std::unexpected(std::move(validation.error()));
    }
    return header;
}

std::expected<ProtocolMessage, ProtocolError>
DecodeProtocolMessage(const std::span<const std::byte> bytes) {
    auto header = DecodeProtocolHeader(bytes);
    if (!header) {
        return std::unexpected(std::move(header.error()));
    }

    const std::size_t expected_size = EncodedProtocolHeaderSize + header->payload_size;
    if (bytes.size() != expected_size) {
        return std::unexpected(MakeError(
            ProtocolErrorCode::PayloadSizeMismatch,
            bytes.size() < expected_size ? "The frame is truncated before the declared payload end."
                                         : "The frame contains trailing bytes after its payload."));
    }

    ProtocolMessage message{
        .version = header->version,
        .type = header->type,
        .command = header->command,
        .status = header->status,
        .request_id = header->request_id,
        .payload = {},
    };
    message.payload.reserve(header->payload_size);
    for (std::size_t index = EncodedProtocolHeaderSize; index < bytes.size(); ++index) {
        message.payload.push_back(bytes[index]);
    }

    return message;
}

ProtocolMessage MakeProtocolRequest(const Command command, const std::uint64_t request_id,
                                    std::vector<std::byte> payload) {
    return ProtocolMessage{
        .version = CurrentProtocolVersion,
        .type = MessageType::Request,
        .command = command,
        .status = Status::Ok,
        .request_id = request_id,
        .payload = std::move(payload),
    };
}

ProtocolMessage MakeProtocolResponse(const ProtocolMessage& request, const Status status,
                                     std::vector<std::byte> payload) {
    return ProtocolMessage{
        .version = CurrentProtocolVersion,
        .type = MessageType::Response,
        .command = request.command,
        .status = status,
        .request_id = request.request_id,
        .payload = std::move(payload),
    };
}

std::expected<std::vector<std::byte>, ProtocolError>
EncodeVersionInformation(const VersionInformation& information) {
    if (information.protocol_version == 0 || information.minimum_compatible_version == 0 ||
        information.minimum_compatible_version > information.protocol_version ||
        information.maximum_payload_size == 0 ||
        information.maximum_payload_size > MaximumProtocolPayloadSize) {
        return std::unexpected(
            MakeError(ProtocolErrorCode::InvalidPayload,
                      "Version information contains an invalid compatibility range."));
    }

    std::vector<std::byte> output;
    output.reserve(8);
    AppendUnsigned16(output, information.protocol_version);
    AppendUnsigned16(output, information.minimum_compatible_version);
    AppendUnsigned32(output, information.maximum_payload_size);
    return output;
}

std::expected<VersionInformation, ProtocolError>
DecodeVersionInformation(const std::span<const std::byte> payload) {
    if (payload.size() != 8) {
        return std::unexpected(
            MakeError(ProtocolErrorCode::InvalidPayload,
                      "A version-information payload must contain exactly 8 bytes."));
    }

    const VersionInformation information{
        .protocol_version = ReadUnsigned16(payload, 0),
        .minimum_compatible_version = ReadUnsigned16(payload, 2),
        .maximum_payload_size = ReadUnsigned32(payload, 4),
    };
    if (information.protocol_version == 0 || information.minimum_compatible_version == 0 ||
        information.minimum_compatible_version > information.protocol_version ||
        information.maximum_payload_size == 0 ||
        information.maximum_payload_size > MaximumProtocolPayloadSize) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidPayload,
                                         "Decoded version information has invalid limits."));
    }
    return information;
}

std::expected<std::vector<std::byte>, ProtocolError>
EncodeDeviceDescriptions(const std::span<const DeviceDescription> devices) {
    if (devices.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            MakeError(ProtocolErrorCode::InvalidPayload,
                      "The device count cannot be represented by the protocol."));
    }

    std::vector<std::byte> output;
    const std::size_t estimated_record_count =
        std::min<std::size_t>(devices.size(), (MaximumProtocolPayloadSize - 4) / 32);
    output.reserve(4 + estimated_record_count * 32);
    AppendUnsigned32(output, static_cast<std::uint32_t>(devices.size()));

    for (const DeviceDescription& device : devices) {
        if (device.backend_name.size() > std::numeric_limits<std::uint16_t>::max() ||
            device.device_name.size() > std::numeric_limits<std::uint16_t>::max()) {
            return std::unexpected(
                MakeError(ProtocolErrorCode::InvalidPayload,
                          "A device name exceeds the uint16 wire length limit."));
        }

        const std::size_t record_size =
            device.identifier.size() + 6 + device.backend_name.size() + device.device_name.size();
        if (record_size > MaximumProtocolPayloadSize - output.size()) {
            return std::unexpected(MakeError(ProtocolErrorCode::PayloadTooLarge,
                                             "The encoded device list exceeds the 64 KiB limit."));
        }

        for (const std::byte identifier_byte : device.identifier) {
            output.push_back(identifier_byte);
        }
        AppendUnsigned16(output, static_cast<std::uint16_t>(device.kind));
        AppendUnsigned16(output, static_cast<std::uint16_t>(device.backend_name.size()));
        AppendUnsigned16(output, static_cast<std::uint16_t>(device.device_name.size()));
        AppendStringBytes(output, device.backend_name);
        AppendStringBytes(output, device.device_name);
    }
    return output;
}

std::expected<std::vector<DeviceDescription>, ProtocolError>
DecodeDeviceDescriptions(const std::span<const std::byte> payload) {
    if (payload.size() > MaximumProtocolPayloadSize) {
        return std::unexpected(MakeError(ProtocolErrorCode::PayloadTooLarge,
                                         "The encoded device list exceeds the 64 KiB limit."));
    }
    if (payload.size() < 4) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidPayload,
                                         "A device-list payload is missing its record count."));
    }

    const std::uint32_t count = ReadUnsigned32(payload, 0);
    constexpr std::size_t MinimumRecordSize = 22;
    if (count > (payload.size() - 4) / MinimumRecordSize) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidPayload,
                                         "The device count cannot fit in the available payload."));
    }

    std::vector<DeviceDescription> devices;
    devices.reserve(count);
    std::size_t offset = 4;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (payload.size() - offset < MinimumRecordSize) {
            return std::unexpected(
                MakeError(ProtocolErrorCode::InvalidPayload, "A device record is truncated."));
        }

        DeviceDescription device;
        for (std::size_t byte_index = 0; byte_index < device.identifier.size(); ++byte_index) {
            device.identifier[byte_index] = payload[offset + byte_index];
        }
        offset += device.identifier.size();
        device.kind = static_cast<DeviceKind>(ReadUnsigned16(payload, offset));
        offset += 2;
        const std::size_t backend_name_size = ReadUnsigned16(payload, offset);
        offset += 2;
        const std::size_t device_name_size = ReadUnsigned16(payload, offset);
        offset += 2;

        const std::size_t names_size = backend_name_size + device_name_size;
        if (names_size > payload.size() - offset) {
            return std::unexpected(MakeError(ProtocolErrorCode::InvalidPayload,
                                             "A device record name extends beyond the payload."));
        }

        device.backend_name = ReadStringBytes(payload.subspan(offset, backend_name_size));
        offset += backend_name_size;
        device.device_name = ReadStringBytes(payload.subspan(offset, device_name_size));
        offset += device_name_size;
        devices.push_back(std::move(device));
    }

    if (offset != payload.size()) {
        return std::unexpected(MakeError(ProtocolErrorCode::InvalidPayload,
                                         "The device-list payload contains trailing bytes."));
    }
    return devices;
}

} // namespace blackframe::ipc
