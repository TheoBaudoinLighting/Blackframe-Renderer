#include "LocalTransportFraming.hpp"

#include <Blackframe/IPC/LocalTransport.hpp>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <cstddef>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <windows.h>

namespace blackframe::ipc {
namespace {

inline constexpr DWORD ClientPipeWaitMilliseconds = 5'000;
inline constexpr DWORD ClientPipePollMilliseconds = 10;

[[nodiscard]] TransportError WindowsError(const TransportErrorCode code,
                                          const std::string_view operation,
                                          const DWORD native_error) {
    std::string message{operation};
    message += ": ";
    message += std::system_category().message(static_cast<int>(native_error));
    return detail::MakeTransportError(code, std::move(message),
                                      static_cast<std::int64_t>(native_error));
}

[[nodiscard]] std::expected<std::wstring, TransportError>
ConvertUtf8ToWide(const std::string_view value) {
    if (value.empty() || value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return std::unexpected(detail::MakeTransportError(
            TransportErrorCode::InvalidEndpoint, "The named-pipe endpoint is empty or too long."));
    }

    const int required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                             static_cast<int>(value.size()), nullptr, 0);
    if (required == 0) {
        return std::unexpected(WindowsError(TransportErrorCode::InvalidEndpoint,
                                            "Converting the endpoint from UTF-8", GetLastError()));
    }

    std::wstring result(static_cast<std::size_t>(required), L'\0');
    const int converted =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                            static_cast<int>(value.size()), result.data(), required);
    if (converted != required) {
        return std::unexpected(WindowsError(TransportErrorCode::InvalidEndpoint,
                                            "Converting the endpoint from UTF-8", GetLastError()));
    }
    return result;
}

[[nodiscard]] std::expected<std::wstring, TransportError>
BuildLocalPipeName(const LocalEndpoint& endpoint) {
    constexpr std::string_view Utf8Prefix = R"(\\.\pipe\)";
    constexpr std::wstring_view WidePrefix = L"\\\\.\\pipe\\";

    if (endpoint.address.starts_with(Utf8Prefix)) {
        auto full_name = ConvertUtf8ToWide(endpoint.address);
        if (!full_name) {
            return std::unexpected(std::move(full_name.error()));
        }
        if (full_name->size() <= WidePrefix.size() || full_name->size() >= 256) {
            return std::unexpected(
                detail::MakeTransportError(TransportErrorCode::InvalidEndpoint,
                                           "The local named-pipe path has an invalid length."));
        }
        return full_name;
    }

    if (endpoint.address.empty()) {
        return std::unexpected(detail::MakeTransportError(
            TransportErrorCode::InvalidEndpoint, "A named-pipe endpoint must not be empty."));
    }
    for (const char character : endpoint.address) {
        const auto code_unit = static_cast<unsigned char>(character);
        if (code_unit < 0x20U || code_unit == static_cast<unsigned char>('\\') ||
            code_unit == static_cast<unsigned char>('/')) {
            return std::unexpected(detail::MakeTransportError(
                TransportErrorCode::InvalidEndpoint,
                "A simple pipe name cannot contain separators or control characters."));
        }
    }

    auto simple_name = ConvertUtf8ToWide(endpoint.address);
    if (!simple_name) {
        return std::unexpected(std::move(simple_name.error()));
    }
    std::wstring full_name{WidePrefix};
    full_name += *simple_name;
    if (full_name.size() >= 256) {
        return std::unexpected(
            detail::MakeTransportError(TransportErrorCode::InvalidEndpoint,
                                       "The expanded local named-pipe path is too long."));
    }
    return full_name;
}

class UniqueHandle final {
  public:
    UniqueHandle() noexcept = default;
    explicit UniqueHandle(const HANDLE handle) noexcept : handle_{handle} {}

    UniqueHandle(UniqueHandle&& other) noexcept
        : handle_{std::exchange(other.handle_, INVALID_HANDLE_VALUE)} {}

    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            Reset();
            handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
        }
        return *this;
    }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    ~UniqueHandle() {
        Reset();
    }

    [[nodiscard]] HANDLE Get() const noexcept {
        return handle_;
    }

    [[nodiscard]] bool IsValid() const noexcept {
        return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
    }

  private:
    void Reset() noexcept {
        if (IsValid()) {
            CloseHandle(handle_);
        }
        handle_ = INVALID_HANDLE_VALUE;
    }

    HANDLE handle_{INVALID_HANDLE_VALUE};
};

class ConnectedPipeScope final {
  public:
    explicit ConnectedPipeScope(const HANDLE pipe) noexcept : pipe_{pipe} {}

    ConnectedPipeScope(const ConnectedPipeScope&) = delete;
    ConnectedPipeScope& operator=(const ConnectedPipeScope&) = delete;

    ~ConnectedPipeScope() {
        DisconnectNamedPipe(pipe_);
    }

  private:
    HANDLE pipe_;
};

[[nodiscard]] std::expected<void, TransportError> ReadExactly(const HANDLE pipe,
                                                              std::span<std::byte> output) {
    while (!output.empty()) {
        const DWORD requested = static_cast<DWORD>(
            std::min<std::size_t>(output.size(), std::numeric_limits<DWORD>::max()));
        DWORD received = 0;
        if (!ReadFile(pipe, output.data(), requested, &received, nullptr)) {
            const DWORD error = GetLastError();
            const TransportErrorCode code = error == ERROR_BROKEN_PIPE
                                                ? TransportErrorCode::PeerClosed
                                                : TransportErrorCode::ReadFailed;
            return std::unexpected(WindowsError(code, "Reading from the named pipe", error));
        }
        if (received == 0) {
            return std::unexpected(detail::MakeTransportError(
                TransportErrorCode::PeerClosed,
                "The named-pipe peer closed before the frame was complete."));
        }
        output = output.subspan(received);
    }
    return {};
}

[[nodiscard]] std::expected<void, TransportError> WriteExactly(const HANDLE pipe,
                                                               std::span<const std::byte> input) {
    while (!input.empty()) {
        const DWORD requested = static_cast<DWORD>(
            std::min<std::size_t>(input.size(), std::numeric_limits<DWORD>::max()));
        DWORD written = 0;
        if (!WriteFile(pipe, input.data(), requested, &written, nullptr)) {
            const DWORD error = GetLastError();
            const TransportErrorCode code = error == ERROR_BROKEN_PIPE
                                                ? TransportErrorCode::PeerClosed
                                                : TransportErrorCode::WriteFailed;
            return std::unexpected(WindowsError(code, "Writing to the named pipe", error));
        }
        if (written == 0) {
            return std::unexpected(detail::MakeTransportError(
                TransportErrorCode::WriteFailed, "Writing to the named pipe made no progress."));
        }
        input = input.subspan(written);
    }
    return {};
}

[[nodiscard]] std::expected<UniqueHandle, TransportError> OpenClientPipe(const std::wstring& name) {
    const ULONGLONG start_time = GetTickCount64();
    for (;;) {
        const HANDLE pipe =
            CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                        SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
        if (pipe != INVALID_HANDLE_VALUE) {
            DWORD mode = PIPE_READMODE_BYTE;
            if (!SetNamedPipeHandleState(pipe, &mode, nullptr, nullptr)) {
                const DWORD error = GetLastError();
                CloseHandle(pipe);
                return std::unexpected(WindowsError(TransportErrorCode::ConnectionFailed,
                                                    "Selecting byte mode for the named pipe",
                                                    error));
            }
            return UniqueHandle{pipe};
        }

        const DWORD error = GetLastError();
        if (error != ERROR_PIPE_BUSY && error != ERROR_FILE_NOT_FOUND) {
            return std::unexpected(WindowsError(TransportErrorCode::ConnectionFailed,
                                                "Opening the named-pipe endpoint", error));
        }

        const ULONGLONG elapsed = GetTickCount64() - start_time;
        if (elapsed >= ClientPipeWaitMilliseconds) {
            return std::unexpected(detail::MakeTransportError(
                TransportErrorCode::ConnectionFailed,
                "Timed out waiting for the named-pipe endpoint to become available."));
        }
        const DWORD remaining = static_cast<DWORD>(ClientPipeWaitMilliseconds - elapsed);

        if (error == ERROR_PIPE_BUSY) {
            if (!WaitNamedPipeW(name.c_str(), remaining)) {
                const DWORD wait_error = GetLastError();
                if (wait_error != ERROR_FILE_NOT_FOUND) {
                    return std::unexpected(WindowsError(TransportErrorCode::ConnectionFailed,
                                                        "Waiting for the named-pipe endpoint",
                                                        wait_error));
                }
            }
        } else {
            Sleep(std::min(ClientPipePollMilliseconds, remaining));
        }
    }
}

} // namespace

struct LocalIpcServer::Implementation {
    std::wstring pipe_name;
    UniqueHandle pipe;
};

LocalIpcServer::LocalIpcServer(std::unique_ptr<Implementation> implementation) noexcept
    : implementation_{std::move(implementation)} {}

LocalIpcServer::LocalIpcServer(LocalIpcServer&& other) noexcept = default;

LocalIpcServer& LocalIpcServer::operator=(LocalIpcServer&& other) noexcept = default;

LocalIpcServer::~LocalIpcServer() = default;

std::expected<LocalIpcServer, TransportError>
LocalIpcServer::Listen(const LocalEndpoint& endpoint) {
    auto pipe_name = BuildLocalPipeName(endpoint);
    if (!pipe_name) {
        return std::unexpected(std::move(pipe_name.error()));
    }

    const HANDLE pipe = CreateNamedPipeW(
        pipe_name->c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1,
        static_cast<DWORD>(MaximumEncodedProtocolMessageSize),
        static_cast<DWORD>(MaximumEncodedProtocolMessageSize), 0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        const TransportErrorCode code = error == ERROR_ACCESS_DENIED
                                            ? TransportErrorCode::EndpointInUse
                                            : TransportErrorCode::SystemFailure;
        return std::unexpected(WindowsError(code, "Creating the local named-pipe endpoint", error));
    }

    auto implementation = std::make_unique<Implementation>();
    implementation->pipe_name = std::move(*pipe_name);
    implementation->pipe = UniqueHandle{pipe};
    return LocalIpcServer{std::move(implementation)};
}

std::expected<ServerAction, TransportError>
LocalIpcServer::ServeOne(const RequestHandler& handler) {
    if (!implementation_ || !implementation_->pipe.IsValid()) {
        return std::unexpected(detail::MakeTransportError(
            TransportErrorCode::InvalidArgument, "The IPC server has no valid listening pipe."));
    }

    const HANDLE pipe = implementation_->pipe.Get();
    if (!ConnectNamedPipe(pipe, nullptr)) {
        const DWORD error = GetLastError();
        if (error != ERROR_PIPE_CONNECTED) {
            return std::unexpected(WindowsError(TransportErrorCode::ConnectionFailed,
                                                "Accepting a named-pipe connection", error));
        }
    }
    const ConnectedPipeScope connection_scope{pipe};

    auto request = detail::ReadProtocolMessage(
        [pipe](const std::span<std::byte> bytes) { return ReadExactly(pipe, bytes); });
    if (!request) {
        return std::unexpected(std::move(request.error()));
    }

    detail::ProcessedRequest processed = detail::ProcessRequest(*request, handler);
    if (auto write_result = detail::WriteProtocolMessage(
            processed.response,
            [pipe](const std::span<const std::byte> bytes) { return WriteExactly(pipe, bytes); });
        !write_result) {
        return std::unexpected(std::move(write_result.error()));
    }
    if (!FlushFileBuffers(pipe)) {
        return std::unexpected(WindowsError(TransportErrorCode::WriteFailed,
                                            "Flushing the named-pipe response", GetLastError()));
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

    auto pipe_name = BuildLocalPipeName(endpoint);
    if (!pipe_name) {
        return std::unexpected(std::move(pipe_name.error()));
    }
    auto pipe = OpenClientPipe(*pipe_name);
    if (!pipe) {
        return std::unexpected(std::move(pipe.error()));
    }

    if (auto write_result =
            detail::WriteProtocolMessage(request,
                                         [&pipe](const std::span<const std::byte> bytes) {
                                             return WriteExactly(pipe->Get(), bytes);
                                         });
        !write_result) {
        return std::unexpected(std::move(write_result.error()));
    }

    auto response = detail::ReadProtocolMessage(
        [&pipe](const std::span<std::byte> bytes) { return ReadExactly(pipe->Get(), bytes); });
    if (!response) {
        return std::unexpected(std::move(response.error()));
    }
    if (auto validation = detail::ValidateClientResponse(request, *response); !validation) {
        return std::unexpected(std::move(validation.error()));
    }
    return response;
}

} // namespace blackframe::ipc
