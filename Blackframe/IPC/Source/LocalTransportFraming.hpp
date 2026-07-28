#pragma once

#include <Blackframe/IPC/LocalTransport.hpp>
#include <Blackframe/IPC/ProtocolCodec.hpp>
#include <algorithm>
#include <array>
#include <exception>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::ipc::detail {

[[nodiscard]] inline TransportError MakeTransportError(const TransportErrorCode code,
                                                       std::string message,
                                                       const std::int64_t native_error = 0) {
    return TransportError{code, native_error, std::move(message)};
}

[[nodiscard]] inline TransportError ProtocolFailure(const ProtocolError& error) {
    return MakeTransportError(TransportErrorCode::ProtocolViolation, error.message);
}

template <typename ReadExact>
[[nodiscard]] std::expected<ProtocolMessage, TransportError>
ReadProtocolMessage(ReadExact&& read_exact) {
    std::array<std::byte, EncodedProtocolHeaderSize> header_bytes{};
    if (auto read_result = read_exact(std::span<std::byte>{header_bytes}); !read_result) {
        return std::unexpected(std::move(read_result.error()));
    }

    auto header = DecodeProtocolHeader(header_bytes);
    if (!header) {
        return std::unexpected(ProtocolFailure(header.error()));
    }

    std::vector<std::byte> encoded(EncodedProtocolHeaderSize + header->payload_size);
    for (std::size_t index = 0; index < header_bytes.size(); ++index) {
        encoded[index] = header_bytes[index];
    }

    if (header->payload_size != 0) {
        std::span<std::byte> payload{encoded.data() + EncodedProtocolHeaderSize,
                                     header->payload_size};
        if (auto read_result = read_exact(payload); !read_result) {
            return std::unexpected(std::move(read_result.error()));
        }
    }

    auto decoded = DecodeProtocolMessage(encoded);
    if (!decoded) {
        return std::unexpected(ProtocolFailure(decoded.error()));
    }
    return std::move(*decoded);
}

template <typename WriteExact>
[[nodiscard]] std::expected<void, TransportError>
WriteProtocolMessage(const ProtocolMessage& message, WriteExact&& write_exact) {
    auto encoded = EncodeProtocolMessage(message);
    if (!encoded) {
        return std::unexpected(ProtocolFailure(encoded.error()));
    }
    return write_exact(std::span<const std::byte>{*encoded});
}

[[nodiscard]] inline std::vector<std::byte> ErrorPayload(const std::string_view message) {
    std::vector<std::byte> payload;
    const std::size_t size = std::min<std::size_t>(message.size(), MaximumProtocolPayloadSize);
    payload.reserve(size);
    for (std::size_t index = 0; index < size; ++index) {
        payload.push_back(static_cast<std::byte>(static_cast<unsigned char>(message[index])));
    }
    return payload;
}

struct ProcessedRequest {
    ProtocolMessage response;
    ServerAction action{ServerAction::Continue};
};

[[nodiscard]] inline ProcessedRequest ProcessRequest(const ProtocolMessage& request,
                                                     const RequestHandler& handler) {
    if (request.type != MessageType::Request) {
        return ProcessedRequest{
            MakeProtocolResponse(request, Status::InvalidMessage,
                                 ErrorPayload("The server accepts request frames only.")),
            ServerAction::Continue,
        };
    }
    if (!IsKnownCommand(request.command)) {
        return ProcessedRequest{
            MakeProtocolResponse(request, Status::UnsupportedCommand,
                                 ErrorPayload("The requested command is not supported.")),
            ServerAction::Continue,
        };
    }
    if (!request.payload.empty()) {
        return ProcessedRequest{
            MakeProtocolResponse(
                request, Status::InvalidPayload,
                ErrorPayload("Protocol v1 commands require an empty request payload.")),
            ServerAction::Continue,
        };
    }
    if (!handler) {
        return ProcessedRequest{
            MakeProtocolResponse(request, Status::InternalError,
                                 ErrorPayload("No request handler was supplied.")),
            ServerAction::Continue,
        };
    }

    try {
        auto handled = handler(request);
        if (!handled) {
            const Status status = handled.error().status == Status::Ok ? Status::InternalError
                                                                       : handled.error().status;
            return ProcessedRequest{
                MakeProtocolResponse(request, status, ErrorPayload(handled.error().message)),
                ServerAction::Continue,
            };
        }

        if (handled->payload.size() > MaximumProtocolPayloadSize) {
            return ProcessedRequest{
                MakeProtocolResponse(
                    request, Status::InternalError,
                    ErrorPayload("The handler produced a payload larger than 64 KiB.")),
                ServerAction::Continue,
            };
        }

        return ProcessedRequest{
            MakeProtocolResponse(request, handled->status, std::move(handled->payload)),
            handled->stop_server_after_response ? ServerAction::Stop : ServerAction::Continue,
        };
    } catch (...) {
        // An exception is contained at the application callback boundary. No
        // implementation-specific exception text is leaked over IPC.
        return ProcessedRequest{
            MakeProtocolResponse(request, Status::InternalError,
                                 ErrorPayload("The request handler raised an exception.")),
            ServerAction::Continue,
        };
    }
}

[[nodiscard]] inline std::expected<void, TransportError>
ValidateClientResponse(const ProtocolMessage& request, const ProtocolMessage& response) {
    if (response.type != MessageType::Response) {
        return std::unexpected(
            MakeTransportError(TransportErrorCode::ProtocolViolation,
                               "The peer returned a frame that is not a response."));
    }
    if (response.request_id != request.request_id || response.command != request.command) {
        return std::unexpected(
            MakeTransportError(TransportErrorCode::ProtocolViolation,
                               "The response correlation fields do not match the request."));
    }
    return {};
}

} // namespace blackframe::ipc::detail
