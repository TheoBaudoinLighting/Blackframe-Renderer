#pragma once

#include <Blackframe/IPC/Protocol.hpp>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace blackframe::ipc {

// Windows accepts either a simple pipe name or a local \\.\pipe\... path.
// POSIX requires an absolute AF_UNIX filesystem path whose parent directory
// already exists. A private (mode 0700) parent directory is recommended.
struct LocalEndpoint {
    std::string address;
};

enum class TransportErrorCode {
    InvalidArgument,
    InvalidEndpoint,
    EndpointInUse,
    ConnectionFailed,
    PeerClosed,
    ReadFailed,
    WriteFailed,
    ProtocolViolation,
    SystemFailure,
};

struct TransportError {
    TransportErrorCode code{TransportErrorCode::SystemFailure};
    std::int64_t native_error{};
    std::string message;
};

struct RequestHandlingError {
    Status status{Status::InternalError};
    std::string message;
};

// The server creates the actual protocol response, preserving command and
// request_id. A handler therefore cannot accidentally mismatch correlation
// fields. stop_server_after_response is observed only after the reply is sent.
struct ServerResponse {
    Status status{Status::Ok};
    std::vector<std::byte> payload;
    bool stop_server_after_response{};
};

using RequestHandler = std::function<std::expected<ServerResponse, RequestHandlingError>(
    const ProtocolMessage& request)>;

enum class ServerAction {
    Continue,
    Stop,
};

class LocalIpcClient final {
  public:
    // Opens one connection, sends one request, receives one response and
    // closes the connection. Correlation fields are validated before return.
    [[nodiscard]] static std::expected<ProtocolMessage, TransportError>
    Exchange(const LocalEndpoint& endpoint, const ProtocolMessage& request);
};

class LocalIpcServer final {
  public:
    LocalIpcServer(LocalIpcServer&& other) noexcept;
    LocalIpcServer& operator=(LocalIpcServer&& other) noexcept;
    LocalIpcServer(const LocalIpcServer&) = delete;
    LocalIpcServer& operator=(const LocalIpcServer&) = delete;
    ~LocalIpcServer();

    [[nodiscard]] static std::expected<LocalIpcServer, TransportError>
    Listen(const LocalEndpoint& endpoint);

    // Accepts exactly one connection and exactly one request. The connection
    // is closed after its response even when ServerAction::Continue is
    // returned.
    [[nodiscard]] std::expected<ServerAction, TransportError>
    ServeOne(const RequestHandler& handler);

    // Runs synchronously until a handler asks to stop after its response.
    // There is intentionally no detached worker or hidden global state.
    [[nodiscard]] std::expected<std::size_t, TransportError> Run(const RequestHandler& handler);

  private:
    struct Implementation;

    explicit LocalIpcServer(std::unique_ptr<Implementation> implementation) noexcept;

    std::unique_ptr<Implementation> implementation_;
};

} // namespace blackframe::ipc
