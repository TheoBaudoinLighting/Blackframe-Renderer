add_executable(
    BlackframeRendererTests
    BasicSpectraTest.cpp
    BoundsTest.cpp
    CapabilityRegistryTest.cpp
    Cie1931SensorTest.cpp
    DebugIntegratorGoldenTest.cpp
    DebugIntegratorsTest.cpp
    DisplayPsnrTest.cpp
    ExrWriterTest.cpp
    FilmTest.cpp
    FilmTileTest.cpp
    GeometryTypesTest.cpp
    HdrMetricsTest.cpp
    IndependentSamplerTest.cpp
    LambertianReflectionTest.cpp
    LinearMetricsTest.cpp
    LocalFrameTest.cpp
    LocalPcg32Test.cpp
    NumericPrecisionTest.cpp
    OwenScramblingTest.cpp
    PinholeCameraTest.cpp
    PixelJitterTest.cpp
    PlanarPrimitivesTest.cpp
    PngWriterTest.cpp
    RayTest.cpp
    RayOriginOffsetTest.cpp
    RenderConfigurationTest.cpp
    RenderSessionStateTest.cpp
    SampleDimensionMapTest.cpp
    SampleStreamTest.cpp
    SamplingMappingsTest.cpp
    SobolSamplingTest.cpp
    SphereTest.cpp
    SpectrumTest.cpp
    StratifiedSamplingTest.cpp
    SurfaceInteractionTest.cpp
    TransformsTest.cpp
    TriangleTest.cpp
    WavelengthSamplingTest.cpp
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

function(blackframe_add_png_checksum_test test_name gtest_filter output_filename expected_sha256)
    add_test(
        NAME "${test_name}"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeRendererTests>"
            "-DGTEST_FILTER=${gtest_filter}"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/${output_filename}"
            "-DEXPECTED_SHA256=${expected_sha256}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/PngChecksumTest.cmake"
    )
    set_tests_properties(
        "${test_name}"
        PROPERTIES
            LABELS "regression;cpu"
    )
endfunction()

if(TARGET Blackframe::Png)
    blackframe_add_png_checksum_test(
        Blackframe.PngPreviewChecksum
        PngWriterTest.WritesFixedDisplayTransformForSyntheticCrop
        checksum-preview.png
        3ff3ba54ed3d83a806b7e51583bb64ff106a6e51c09d589ad6115f56a8bcf168
    )
    blackframe_add_png_checksum_test(
        Blackframe.OwenScramblingVisualizer
        OwenScramblingTest.WritesStableVisualizer
        checksum-owen-scrambling-visualizer.png
        7f92475259a487f9f49e94706b4a44cb5198d6c60806ebd6827672659547c7b2
    )
    blackframe_add_png_checksum_test(
        Blackframe.DebugGolden.Normal
        DebugIntegratorGoldenTest.Normal64x64
        checksum-debug-normal-64x64.png
        1f73b6edf88198d52709acbe5aaeecbf4b8f6d61cd5ce71671ca6a6632339f9d
    )
    blackframe_add_png_checksum_test(
        Blackframe.DebugGolden.Depth
        DebugIntegratorGoldenTest.Depth64x64
        checksum-debug-depth-64x64.png
        5a4cd5a42464e04293eec3017c52a2ce4936169fcd454b22599748b654413821
    )
    blackframe_add_png_checksum_test(
        Blackframe.DebugGolden.Uv
        DebugIntegratorGoldenTest.Uv64x64
        checksum-debug-uv-64x64.png
        29011e89b06232f139b122ac18542c4528aa7850eed4a9dc27e14299a493f893
    )
    blackframe_add_png_checksum_test(
        Blackframe.DebugGolden.Barycentrics
        DebugIntegratorGoldenTest.Barycentrics64x64
        checksum-debug-barycentrics-64x64.png
        0bca6a41b30e6c415731b58f816526f16e82ea250d317320077874d806283adf
    )
    blackframe_add_png_checksum_test(
        Blackframe.DebugGolden.Identifiers
        DebugIntegratorGoldenTest.Identifiers64x64
        checksum-debug-identifiers-64x64.png
        7451a7fc96bcde1205464c2e20eff7029ff75c0cd98f60666a52a1d6790aa240
    )
endif()
