add_executable(
    BlackframeRenderBenchmarks
    ProtocolCodecBenchmark.cpp
    RenderBenchmarksMain.cpp
)
blackframe_set_target_role(BlackframeRenderBenchmarks benchmark)

target_link_libraries(
    BlackframeRenderBenchmarks
    PRIVATE
        Blackframe::IPC
        Blackframe::ProjectOptions
        benchmark::benchmark
)

set_target_properties(
    BlackframeRenderBenchmarks
    PROPERTIES
        OUTPUT_NAME render_benchmarks
        DEBUG_POSTFIX ""
)

if(BUILD_TESTING)
    add_test(
        NAME Blackframe.Executables.RenderBenchmarks.Help
        COMMAND "$<TARGET_FILE:BlackframeRenderBenchmarks>" --help
    )
    add_test(
        NAME Blackframe.Benchmarks.ProtocolCodec.Json
        COMMAND
            "${CMAKE_COMMAND}"
            "-DBENCHMARK_EXECUTABLE=$<TARGET_FILE:BlackframeRenderBenchmarks>"
            "-DBINARY_DIRECTORY=${CMAKE_BINARY_DIR}"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/render_benchmarks.json"
            -P
            "${CMAKE_CURRENT_SOURCE_DIR}/ValidateBenchmarkJson.cmake"
    )
    set_tests_properties(
        Blackframe.Executables.RenderBenchmarks.Help
        PROPERTIES
            LABELS "smoke;cpu"
            PASS_REGULAR_EXPRESSION "render_benchmarks --help"
    )
    set_tests_properties(
        Blackframe.Benchmarks.ProtocolCodec.Json
        PROPERTIES
            LABELS "smoke;cpu"
            TIMEOUT 30
    )
endif()
