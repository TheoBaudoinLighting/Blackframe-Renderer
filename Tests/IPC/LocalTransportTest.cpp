#include <Blackframe/IPC/LocalTransport.hpp>
#include <Blackframe/IPC/ProtocolCodec.hpp>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <thread>

namespace blackframe::ipc {
namespace {

[[nodiscard]] LocalEndpoint unique_endpoint() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(_WIN32)
    return LocalEndpoint{.address = "Blackframe-Test-" + std::to_string(nonce)};
#else
    return LocalEndpoint{
        .address = (std::filesystem::temp_directory_path() /
                    ("blackframe-test-" + std::to_string(nonce) + ".sock"))
                       .string(),
    };
#endif
}

TEST(LocalTransportTest, ExchangesOneCorrelatedRequestAndResponse) {
    const auto endpoint = unique_endpoint();
    auto listener = LocalIpcServer::Listen(endpoint);
    ASSERT_TRUE(listener.has_value()) << listener.error().message;

    std::optional<std::expected<std::size_t, TransportError>> server_result;
    auto server_thread = std::jthread{
        [server = std::move(*listener), &server_result]() mutable {
            server_result = server.Run([](const ProtocolMessage& request)
                                           -> std::expected<ServerResponse, RequestHandlingError> {
                EXPECT_EQ(request.command, Command::Ping);
                return ServerResponse{
                    .status = Status::Ok,
                    .payload = {},
                    .stop_server_after_response = true,
                };
            });
        },
    };

    const auto response =
        LocalIpcClient::Exchange(endpoint, MakeProtocolRequest(Command::Ping, 42));
    server_thread.join();

    ASSERT_TRUE(response.has_value()) << response.error().message;
    EXPECT_EQ(response->type, MessageType::Response);
    EXPECT_EQ(response->command, Command::Ping);
    EXPECT_EQ(response->request_id, 42U);
    EXPECT_EQ(response->status, Status::Ok);

    ASSERT_TRUE(server_result.has_value());
    ASSERT_TRUE(server_result->has_value()) << server_result->error().message;
    EXPECT_EQ(**server_result, 1U);
}

TEST(LocalTransportTest, WaitsForAStartingServer) {
    const auto endpoint = unique_endpoint();
    std::optional<std::expected<ProtocolMessage, TransportError>> client_result;
    auto client_thread = std::jthread{[&endpoint, &client_result] {
        client_result = LocalIpcClient::Exchange(endpoint, MakeProtocolRequest(Command::Ping, 99));
    }};

    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    auto listener = LocalIpcServer::Listen(endpoint);
    ASSERT_TRUE(listener.has_value()) << listener.error().message;

    const auto server_result = listener->ServeOne(
        [](const ProtocolMessage&) -> std::expected<ServerResponse, RequestHandlingError> {
            return ServerResponse{
                .status = Status::Ok,
                .payload = {},
                .stop_server_after_response = true,
            };
        });
    client_thread.join();

    ASSERT_TRUE(server_result.has_value()) << server_result.error().message;
    ASSERT_TRUE(client_result.has_value());
    ASSERT_TRUE(client_result->has_value()) << client_result->error().message;
    EXPECT_EQ(client_result->value().request_id, 99U);
}

} // namespace
} // namespace blackframe::ipc
