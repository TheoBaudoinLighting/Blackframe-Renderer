#include "LocalTransportFraming.hpp"

#include <Blackframe/IPC/LocalTransport.hpp>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <expected>
#include <fcntl.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>

namespace blackframe::ipc {
namespace {

inline constexpr auto ClientConnectTimeout = std::chrono::seconds{5};
inline constexpr auto ClientConnectPollInterval = std::chrono::milliseconds{10};

[[nodiscard]] TransportError PosixError(const TransportErrorCode code,
                                        const std::string_view operation, const int native_error) {
    std::string message{operation};
    message += ": ";
    message += std::generic_category().message(native_error);
    return detail::MakeTransportError(code, std::move(message),
                                      static_cast<std::int64_t>(native_error));
}

class UniqueFileDescriptor final {
  public:
    UniqueFileDescriptor() noexcept = default;
    explicit UniqueFileDescriptor(const int descriptor) noexcept : descriptor_{descriptor} {}

    UniqueFileDescriptor(UniqueFileDescriptor&& other) noexcept
        : descriptor_{std::exchange(other.descriptor_, -1)} {}

    UniqueFileDescriptor& operator=(UniqueFileDescriptor&& other) noexcept {
        if (this != &other) {
            Reset();
            descriptor_ = std::exchange(other.descriptor_, -1);
        }
        return *this;
    }

    UniqueFileDescriptor(const UniqueFileDescriptor&) = delete;
    UniqueFileDescriptor& operator=(const UniqueFileDescriptor&) = delete;

    ~UniqueFileDescriptor() {
        Reset();
    }

    [[nodiscard]] int Get() const noexcept {
        return descriptor_;
    }

    [[nodiscard]] bool IsValid() const noexcept {
        return descriptor_ >= 0;
    }

    void Reset() noexcept {
        if (IsValid()) {
            ::close(descriptor_);
        }
        descriptor_ = -1;
    }

  private:
    int descriptor_{-1};
};

struct SocketAddress {
    sockaddr_un native{};
    socklen_t length{};
};

[[nodiscard]] std::expected<SocketAddress, TransportError>
BuildSocketAddress(const LocalEndpoint& endpoint) {
    if (endpoint.address.empty() || endpoint.address.front() != '/') {
        return std::unexpected(
            detail::MakeTransportError(TransportErrorCode::InvalidEndpoint,
                                       "An AF_UNIX endpoint must be an absolute filesystem path."));
    }
    if (endpoint.address.find('\0') != std::string::npos) {
        return std::unexpected(
            detail::MakeTransportError(TransportErrorCode::InvalidEndpoint,
                                       "An AF_UNIX endpoint cannot contain a null byte."));
    }

    SocketAddress address;
    if (endpoint.address.size() >= sizeof(address.native.sun_path)) {
        return std::unexpected(
            detail::MakeTransportError(TransportErrorCode::InvalidEndpoint,
                                       "The AF_UNIX endpoint exceeds sun_path capacity."));
    }

    address.native.sun_family = AF_UNIX;
    for (std::size_t index = 0; index < endpoint.address.size(); ++index) {
        address.native.sun_path[index] = endpoint.address[index];
    }
    address.native.sun_path[endpoint.address.size()] = '\0';
    address.length =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + endpoint.address.size() + 1);
#if defined(__APPLE__) || defined(__FreeBSD__)
    address.native.sun_len = static_cast<unsigned char>(address.length);
#endif
    return address;
}

[[nodiscard]] std::expected<void, TransportError> ConfigureSocket(const int descriptor) {
    const int descriptor_flags = ::fcntl(descriptor, F_GETFD);
    if (descriptor_flags < 0 || ::fcntl(descriptor, F_SETFD, descriptor_flags | FD_CLOEXEC) < 0) {
        return std::unexpected(PosixError(TransportErrorCode::SystemFailure,
                                          "Marking the local socket close-on-exec", errno));
    }

#if defined(SO_NOSIGPIPE)
    const int enabled = 1;
    if (::setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) < 0) {
        return std::unexpected(PosixError(TransportErrorCode::SystemFailure,
                                          "Disabling SIGPIPE on the local socket", errno));
    }
#endif
    return {};
}

[[nodiscard]] std::expected<UniqueFileDescriptor, TransportError>
OpenClientSocket(const SocketAddress& address) {
    const auto deadline = std::chrono::steady_clock::now() + ClientConnectTimeout;
    for (;;) {
        UniqueFileDescriptor connection{::socket(AF_UNIX, SOCK_STREAM, 0)};
        if (!connection.IsValid()) {
            return std::unexpected(PosixError(TransportErrorCode::ConnectionFailed,
                                              "Creating an AF_UNIX client socket", errno));
        }
        if (auto configured = ConfigureSocket(connection.Get()); !configured) {
            return std::unexpected(std::move(configured.error()));
        }
        if (::connect(connection.Get(), reinterpret_cast<const sockaddr*>(&address.native),
                      address.length) == 0) {
            return connection;
        }

        const int error = errno;
        if (error != ENOENT && error != ECONNREFUSED) {
            return std::unexpected(PosixError(TransportErrorCode::ConnectionFailed,
                                              "Connecting to the AF_UNIX endpoint", error));
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return std::unexpected(PosixError(TransportErrorCode::ConnectionFailed,
                                              "Waiting for the AF_UNIX endpoint", error));
        }
        std::this_thread::sleep_for(ClientConnectPollInterval);
    }
}

[[nodiscard]] std::expected<void, TransportError> ReadExactly(const int socket,
                                                              std::span<std::byte> output) {
    while (!output.empty()) {
        const ssize_t received = ::recv(socket, output.data(), output.size(), 0);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            const int error = errno;
            const TransportErrorCode code = error == ECONNRESET ? TransportErrorCode::PeerClosed
                                                                : TransportErrorCode::ReadFailed;
            return std::unexpected(PosixError(code, "Reading from the local socket", error));
        }
        if (received == 0) {
            return std::unexpected(detail::MakeTransportError(
                TransportErrorCode::PeerClosed,
                "The local-socket peer closed before the frame was complete."));
        }
        output = output.subspan(static_cast<std::size_t>(received));
    }
    return {};
}

[[nodiscard]] std::expected<void, TransportError> WriteExactly(const int socket,
                                                               std::span<const std::byte> input) {
    while (!input.empty()) {
        int flags = 0;
#if defined(MSG_NOSIGNAL)
        flags |= MSG_NOSIGNAL;
#endif
        const ssize_t written = ::send(socket, input.data(), input.size(), flags);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            const int error = errno;
            const TransportErrorCode code = error == EPIPE || error == ECONNRESET
                                                ? TransportErrorCode::PeerClosed
                                                : TransportErrorCode::WriteFailed;
            return std::unexpected(PosixError(code, "Writing to the local socket", error));
        }
        if (written == 0) {
            return std::unexpected(detail::MakeTransportError(
                TransportErrorCode::WriteFailed, "Writing to the local socket made no progress."));
        }
        input = input.subspan(static_cast<std::size_t>(written));
    }
    return {};
}

[[nodiscard]] std::expected<UniqueFileDescriptor, TransportError>
AcceptConnection(const int listener) {
    for (;;) {
        const int connection = ::accept(listener, nullptr, nullptr);
        if (connection < 0) {
            if (errno == EINTR) {
                continue;
            }
            return std::unexpected(PosixError(TransportErrorCode::ConnectionFailed,
                                              "Accepting an AF_UNIX connection", errno));
        }

        UniqueFileDescriptor result{connection};
        if (auto configured = ConfigureSocket(connection); !configured) {
            return std::unexpected(std::move(configured.error()));
        }
        return result;
    }
}

[[nodiscard]] bool IsSameSocketNode(const std::string& path, const dev_t device,
                                    const ino_t inode) noexcept {
    struct stat current{};
    return ::lstat(path.c_str(), &current) == 0 && S_ISSOCK(current.st_mode) &&
           current.st_dev == device && current.st_ino == inode;
}

} // namespace

struct LocalIpcServer::Implementation {
    std::string socket_path;
    UniqueFileDescriptor listener;
    dev_t socket_device{};
    ino_t socket_inode{};
    bool owns_socket_node{};

    ~Implementation() {
        listener.Reset();
        // Do not unlink a path that was replaced while the server ran.
        if (owns_socket_node && IsSameSocketNode(socket_path, socket_device, socket_inode)) {
            ::unlink(socket_path.c_str());
        }
    }
};

LocalIpcServer::LocalIpcServer(std::unique_ptr<Implementation> implementation) noexcept
    : implementation_{std::move(implementation)} {}

LocalIpcServer::LocalIpcServer(LocalIpcServer&& other) noexcept = default;

LocalIpcServer& LocalIpcServer::operator=(LocalIpcServer&& other) noexcept = default;

LocalIpcServer::~LocalIpcServer() = default;

std::expected<LocalIpcServer, TransportError>
LocalIpcServer::Listen(const LocalEndpoint& endpoint) {
    auto address = BuildSocketAddress(endpoint);
    if (!address) {
        return std::unexpected(std::move(address.error()));
    }

    struct stat existing{};
    if (::lstat(endpoint.address.c_str(), &existing) == 0) {
        return std::unexpected(detail::MakeTransportError(
            TransportErrorCode::EndpointInUse, "The AF_UNIX endpoint path already exists."));
    }
    if (errno != ENOENT) {
        return std::unexpected(PosixError(TransportErrorCode::SystemFailure,
                                          "Inspecting the AF_UNIX endpoint path", errno));
    }

    UniqueFileDescriptor listener{::socket(AF_UNIX, SOCK_STREAM, 0)};
    if (!listener.IsValid()) {
        return std::unexpected(
            PosixError(TransportErrorCode::SystemFailure, "Creating the AF_UNIX listener", errno));
    }
    if (auto configured = ConfigureSocket(listener.Get()); !configured) {
        return std::unexpected(std::move(configured.error()));
    }

    if (::bind(listener.Get(), reinterpret_cast<const sockaddr*>(&address->native),
               address->length) < 0) {
        const int error = errno;
        const TransportErrorCode code = error == EADDRINUSE ? TransportErrorCode::EndpointInUse
                                                            : TransportErrorCode::SystemFailure;
        return std::unexpected(PosixError(code, "Binding the AF_UNIX endpoint", error));
    }

    struct stat bound_node{};
    if (::lstat(endpoint.address.c_str(), &bound_node) < 0) {
        const int error = errno;
        listener.Reset();
        return std::unexpected(PosixError(TransportErrorCode::SystemFailure,
                                          "Inspecting the bound AF_UNIX endpoint", error));
    }
    if (!S_ISSOCK(bound_node.st_mode)) {
        listener.Reset();
        return std::unexpected(
            detail::MakeTransportError(TransportErrorCode::SystemFailure,
                                       "The bound AF_UNIX endpoint was unexpectedly replaced."));
    }

    const auto remove_bound_node = [&endpoint, &bound_node]() noexcept {
        if (IsSameSocketNode(endpoint.address, bound_node.st_dev, bound_node.st_ino)) {
            ::unlink(endpoint.address.c_str());
        }
    };
    if (::chmod(endpoint.address.c_str(), S_IRUSR | S_IWUSR) < 0) {
        const int error = errno;
        listener.Reset();
        remove_bound_node();
        return std::unexpected(PosixError(TransportErrorCode::SystemFailure,
                                          "Restricting the AF_UNIX endpoint permissions", error));
    }
    if (::listen(listener.Get(), 8) < 0) {
        const int error = errno;
        listener.Reset();
        remove_bound_node();
        return std::unexpected(PosixError(TransportErrorCode::SystemFailure,
                                          "Listening on the AF_UNIX endpoint", error));
    }

    struct stat verified_node{};
    if (::lstat(endpoint.address.c_str(), &verified_node) < 0) {
        const int error = errno;
        listener.Reset();
        remove_bound_node();
        return std::unexpected(PosixError(TransportErrorCode::SystemFailure,
                                          "Verifying the bound AF_UNIX endpoint", error));
    }
    if (!S_ISSOCK(verified_node.st_mode) || verified_node.st_dev != bound_node.st_dev ||
        verified_node.st_ino != bound_node.st_ino) {
        listener.Reset();
        remove_bound_node();
        return std::unexpected(detail::MakeTransportError(
            TransportErrorCode::SystemFailure,
            "The AF_UNIX endpoint changed while it was being configured."));
    }

    auto implementation = std::make_unique<Implementation>();
    implementation->socket_path = endpoint.address;
    implementation->listener = std::move(listener);
    implementation->socket_device = verified_node.st_dev;
    implementation->socket_inode = verified_node.st_ino;
    implementation->owns_socket_node = true;
    return LocalIpcServer{std::move(implementation)};
}

std::expected<ServerAction, TransportError>
LocalIpcServer::ServeOne(const RequestHandler& handler) {
    if (!implementation_ || !implementation_->listener.IsValid()) {
        return std::unexpected(detail::MakeTransportError(
            TransportErrorCode::InvalidArgument, "The IPC server has no valid listening socket."));
    }

    auto connection = AcceptConnection(implementation_->listener.Get());
    if (!connection) {
        return std::unexpected(std::move(connection.error()));
    }

    auto request = detail::ReadProtocolMessage([&connection](const std::span<std::byte> bytes) {
        return ReadExactly(connection->Get(), bytes);
    });
    if (!request) {
        return std::unexpected(std::move(request.error()));
    }

    detail::ProcessedRequest processed = detail::ProcessRequest(*request, handler);
    if (auto write_result =
            detail::WriteProtocolMessage(processed.response,
                                         [&connection](const std::span<const std::byte> bytes) {
                                             return WriteExactly(connection->Get(), bytes);
                                         });
        !write_result) {
        return std::unexpected(std::move(write_result.error()));
    }
    return processed.action;
}

std::expected<std::size_t, TransportError> LocalIpcServer::Run(const RequestHandler& handler) {
    std::size_t response_count = 0;
    for (;;) {
        auto action = ServeOne(handler);
        if (!action) {
            return std::unexpected(std::move(action.error()));
        }
        ++response_count;
        if (*action == ServerAction::Stop) {
            return response_count;
        }
    }
}

std::expected<ProtocolMessage, TransportError>
LocalIpcClient::Exchange(const LocalEndpoint& endpoint, const ProtocolMessage& request) {
    if (request.type != MessageType::Request) {
        return std::unexpected(
            detail::MakeTransportError(TransportErrorCode::InvalidArgument,
                                       "LocalIpcClient::Exchange requires a request frame."));
    }

    auto address = BuildSocketAddress(endpoint);
    if (!address) {
        return std::unexpected(std::move(address.error()));
    }

    auto connection = OpenClientSocket(*address);
    if (!connection) {
        return std::unexpected(std::move(connection.error()));
    }

    if (auto write_result =
            detail::WriteProtocolMessage(request,
                                         [&connection](const std::span<const std::byte> bytes) {
                                             return WriteExactly(connection->Get(), bytes);
                                         });
        !write_result) {
        return std::unexpected(std::move(write_result.error()));
    }

    auto response = detail::ReadProtocolMessage([&connection](const std::span<std::byte> bytes) {
        return ReadExactly(connection->Get(), bytes);
    });
    if (!response) {
        return std::unexpected(std::move(response.error()));
    }
    if (auto validation = detail::ValidateClientResponse(request, *response); !validation) {
        return std::unexpected(std::move(validation.error()));
    }
    return response;
}

} // namespace blackframe::ipc
