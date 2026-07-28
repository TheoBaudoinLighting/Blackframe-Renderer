add_executable(
    BlackframeIpcBenchmarks
    ProtocolCodecBenchmark.cpp
)
blackframe_set_target_role(BlackframeIpcBenchmarks benchmark)

target_link_libraries(
    BlackframeIpcBenchmarks
    PRIVATE
        Blackframe::IPC
        Blackframe::ProjectOptions
        benchmark::benchmark_main
)
