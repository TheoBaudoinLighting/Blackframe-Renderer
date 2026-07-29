add_executable(
    BlackframeRendererTests
    BoundsTest.cpp
    CapabilityRegistryTest.cpp
    DisplayPsnrTest.cpp
    ExrWriterTest.cpp
    FilmTest.cpp
    FilmTileTest.cpp
    GeometryTypesTest.cpp
    HdrMetricsTest.cpp
    LinearMetricsTest.cpp
    LocalFrameTest.cpp
    NumericPrecisionTest.cpp
    PinholeCameraTest.cpp
    PngWriterTest.cpp
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

if(TARGET Blackframe::Png)
    target_link_libraries(
        BlackframeRendererTests
        PRIVATE
            Blackframe::Png
            Blackframe::Stb
    )
    target_compile_definitions(BlackframeRendererTests PRIVATE BLACKFRAME_HAS_PNG_PREVIEW=1)
else()
    target_compile_definitions(BlackframeRendererTests PRIVATE BLACKFRAME_HAS_PNG_PREVIEW=0)
endif()

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

if(TARGET Blackframe::Png)
    add_test(
        NAME Blackframe.PngPreviewChecksum
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeRendererTests>"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-preview.png"
            "-DEXPECTED_SHA256=3ff3ba54ed3d83a806b7e51583bb64ff106a6e51c09d589ad6115f56a8bcf168"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.PngPreviewChecksum
        PROPERTIES
            LABELS "regression;cpu"
    )
endif()
