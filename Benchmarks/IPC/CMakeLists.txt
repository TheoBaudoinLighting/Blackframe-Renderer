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
    set(
        blackframe_microbenchmark_filter
        "^(encode_ping|decode_ping|build_light_tree/.*|sample_light_tree/.*|query_light_tree_probability/.*|sample_power_sampler/.*)$"
    )
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
            "-DBENCHMARK_FILTER=${blackframe_microbenchmark_filter}"
            -P
            "${CMAKE_CURRENT_SOURCE_DIR}/ValidateBenchmarkJson.cmake"
    )
    add_test(
        NAME Blackframe.Benchmarks.CornellPowerMisConvergence.Json
        COMMAND
            "${CMAKE_COMMAND}"
            "-DBENCHMARK_EXECUTABLE=$<TARGET_FILE:BlackframeRenderBenchmarks>"
            "-DBINARY_DIRECTORY=${CMAKE_BINARY_DIR}"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/cornell_power_mis_convergence.json"
            "-DPNG_PATH=${CMAKE_BINARY_DIR}/Benchmarks/Renderer/artifacts/cornell-power-mis-256spp.png"
            -P
            "${CMAKE_CURRENT_SOURCE_DIR}/ValidateConvergenceBenchmarkJson.cmake"
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
    set_tests_properties(
        Blackframe.Benchmarks.CornellPowerMisConvergence.Json
        PROPERTIES
            LABELS "regression;nightly;cpu"
            TIMEOUT 1800
    )
endif()
