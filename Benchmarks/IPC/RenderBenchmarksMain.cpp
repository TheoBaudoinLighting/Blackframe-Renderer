#include <benchmark/benchmark.h>

#include <iostream>
#include <string_view>

namespace {

constexpr auto usage = std::string_view{
    "Blackframe benchmark runner\n"
    "Usage:\n"
    "  render_benchmarks [Google Benchmark options]\n"
    "  render_benchmarks --help\n"
    "JSON output:\n"
    "  render_benchmarks --benchmark_out=<path> --benchmark_out_format=json\n",
};

} // namespace

int main(int argument_count, char** arguments) {
    for (auto argument_index = 1; argument_index < argument_count; ++argument_index) {
        const auto argument = std::string_view{arguments[argument_index]};
        if (argument == "--help" || argument == "-h") {
            std::cout << usage;
            return 0;
        }
    }

    benchmark::Initialize(&argument_count, arguments);
    if (benchmark::ReportUnrecognizedArguments(argument_count, arguments)) {
        return 2;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
