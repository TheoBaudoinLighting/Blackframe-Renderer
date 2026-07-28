add_executable(
    BlackframeIpcBenchmarks
    ProtocolCodecBenchmark.cpp
)

target_link_libraries(
    BlackframeIpcBenchmarks
    PRIVATE
        Blackframe::IPC
        Blackframe::ProjectOptions
        benchmark::benchmark_main
)
