target_sources(
    BlackframeRenderBenchmarks
    PRIVATE
        LightTreeBenchmark.cpp
)

target_link_libraries(
    BlackframeRenderBenchmarks
    PRIVATE
        Blackframe::Renderer
)
