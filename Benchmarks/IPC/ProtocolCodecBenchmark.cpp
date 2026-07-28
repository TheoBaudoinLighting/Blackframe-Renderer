#include <Blackframe/IPC/ProtocolCodec.hpp>
#include <benchmark/benchmark.h>
#include <cstdint>

namespace blackframe::ipc {
namespace {

void encode_ping(benchmark::State& state) {
    const auto request = MakeProtocolRequest(Command::Ping, 1);
    for (auto _ : state) {
        auto encoded = EncodeProtocolMessage(request);
        benchmark::DoNotOptimize(encoded);
    }
}

void decode_ping(benchmark::State& state) {
    const auto encoded = EncodeProtocolMessage(MakeProtocolRequest(Command::Ping, 1)).value();
    for (auto _ : state) {
        auto decoded = DecodeProtocolMessage(encoded);
        benchmark::DoNotOptimize(decoded);
    }
}

BENCHMARK(encode_ping);
BENCHMARK(decode_ping);

} // namespace
} // namespace blackframe::ipc
