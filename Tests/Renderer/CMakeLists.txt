add_executable(
    BlackframeRendererTests
    BoundsTest.cpp
    CapabilityRegistryTest.cpp
    ExrWriterTest.cpp
    FilmTest.cpp
    FilmTileTest.cpp
    GeometryTypesTest.cpp
    LocalFrameTest.cpp
    NumericPrecisionTest.cpp
    RayTest.cpp
    RayOriginOffsetTest.cpp
    RenderConfigurationTest.cpp
    RenderSessionStateTest.cpp
    SpectrumTest.cpp
    TransformsTest.cpp
)
blackframe_set_target_role(BlackframeRendererTests test)

target_link_libraries(
    BlackframeRendererTests
    PRIVATE
        Blackframe::Renderer
        Blackframe::Exr
        Blackframe::OpenExr
        Blackframe::ProjectOptions
        GTest::gtest_main
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
target_compile_definitions(
    BlackframeRendererTests
    PRIVATE
        BLACKFRAME_RENDERER_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
)

gtest_discover_tests(
    BlackframeRendererTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)
