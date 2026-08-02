target_sources(
    BlackframeRenderBenchmarks
    PRIVATE
        CornellConvergenceBenchmark.cpp
        LightTreeBenchmark.cpp
        WavefrontCompactionBenchmark.cpp
)

foreach(blackframe_cornell_target IN ITEMS
        Blackframe::Renderer
        Blackframe::Engine
        Blackframe::CpuEmbree
        Blackframe::Png)
    if(NOT TARGET ${blackframe_cornell_target})
        message(FATAL_ERROR
            "The Cornell convergence benchmark requires ${blackframe_cornell_target}.")
    endif()
endforeach()

target_link_libraries(
    BlackframeRenderBenchmarks
    PRIVATE
        Blackframe::Renderer
        Blackframe::Engine
        Blackframe::CpuEmbree
        Blackframe::Png
)

target_compile_definitions(
    BlackframeRenderBenchmarks
    PRIVATE
        "BLACKFRAME_CORNELL_BENCHMARK_ARTIFACT_PATH=\"${CMAKE_BINARY_DIR}/Benchmarks/Renderer/artifacts/cornell-power-mis-256spp.png\""
)
