add_executable(
    BlackframeRendererTests
    BasicSpectraTest.cpp
    BoundsTest.cpp
    BsdfOnlyPathLoopTest.cpp
    CapabilityRegistryTest.cpp
    Cie1931SensorTest.cpp
    ClosureMixtureTest.cpp
    ClosureSetTest.cpp
    DebugIntegratorGoldenTest.cpp
    DebugIntegratorsTest.cpp
    DisplayPsnrTest.cpp
    DirectLightingTest.cpp
    DepthFilteredClosureMixtureTest.cpp
    EmissionTest.cpp
    ExrWriterTest.cpp
    FilmTest.cpp
    FilmTileTest.cpp
    FresnelTest.cpp
    GeometryTypesTest.cpp
    GgxMicrofacetTest.cpp
    HdrMetricsTest.cpp
    IndependentSamplerTest.cpp
    LambertianReflectionTest.cpp
    LinearMetricsTest.cpp
    LocalFrameTest.cpp
    LocalPcg32Test.cpp
    NumericPrecisionTest.cpp
    OwenScramblingTest.cpp
    PathDepthLimitsTest.cpp
    PathStateTest.cpp
    PathStateSoATest.cpp
    PinholeCameraTest.cpp
    PixelJitterTest.cpp
    PlanarPrimitivesTest.cpp
    PngWriterTest.cpp
    RayTest.cpp
    RayDifferentialTest.cpp
    RayOriginOffsetTest.cpp
    RenderConfigurationTest.cpp
    RenderSessionStateTest.cpp
    RussianRouletteTest.cpp
    RoughConductorReflectionTest.cpp
    RoughDielectricTest.cpp
    RoughDiffuseReflectionTest.cpp
    SampleDimensionMapTest.cpp
    SampleStreamTest.cpp
    SamplingMappingsTest.cpp
    ShadingNormalCorrectionTest.cpp
    SobolSamplingTest.cpp
    SphereTest.cpp
    SpecularDeltaTest.cpp
    SpectrumTest.cpp
    StratifiedSamplingTest.cpp
    SurfaceInteractionTest.cpp
    TransformsTest.cpp
    TransportConventionsTest.cpp
    TriangleTest.cpp
    WavefrontQueuesTest.cpp
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

add_executable(
    BlackframeRendererRegressionTests
    BsdfOnlyCornellConvergenceTest.cpp
)
blackframe_set_target_role(BlackframeRendererRegressionTests test)

target_link_libraries(
    BlackframeRendererRegressionTests
    PRIVATE
        Blackframe::Renderer
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeRendererRegressionTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "regression;cpu"
)

set(blackframe_cornell_scene_dir "${PROJECT_SOURCE_DIR}/scenes")
set(blackframe_cornell_reference_dir "${blackframe_cornell_scene_dir}/references")
set(
    blackframe_cornell_reference_index
    "${blackframe_cornell_reference_dir}/S02_CornellDiffuse.references.json"
)
set(blackframe_cornell_scene_64
    "${blackframe_cornell_scene_dir}/S02_CornellDiffuse_64x64.json"
)
set(blackframe_cornell_scene_256
    "${blackframe_cornell_scene_dir}/S02_CornellDiffuse_256x256.json"
)
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${blackframe_cornell_reference_index}"
        "${blackframe_cornell_scene_64}"
        "${blackframe_cornell_scene_256}"
)
file(SHA256 "${blackframe_cornell_scene_64}" blackframe_cornell_scene_64_sha256)
file(SHA256 "${blackframe_cornell_scene_256}" blackframe_cornell_scene_256_sha256)
set(
    blackframe_cornell_expected_scene_64_sha256
    "1bf6ccd4c740cdbad4509426e7faa9797950db3dd3e34c2209ed67d22b8fd635"
)
set(
    blackframe_cornell_expected_scene_256_sha256
    "b7791857eae37db887b9bc23ba820cbfae1bc727b5d9fe2366a7a2e48ba86972"
)
if(NOT blackframe_cornell_scene_64_sha256 STREQUAL
       blackframe_cornell_expected_scene_64_sha256 OR
   NOT blackframe_cornell_scene_256_sha256 STREQUAL
       blackframe_cornell_expected_scene_256_sha256)
    message(FATAL_ERROR "A closed Cornell scene fixture changed without an explicit revision.")
endif()
file(READ "${blackframe_cornell_reference_index}" blackframe_cornell_reference_manifest)
string(
    JSON blackframe_cornell_source_base_commit
    GET "${blackframe_cornell_reference_manifest}" source_snapshot base_commit
)

function(
    blackframe_add_cornell_reference_test
    target_name
    test_prefix
    variant_is_64
    scene_sha256
    test_labels
)
    add_executable(
        "${target_name}"
        CornellDiffuseReferenceTest.cpp
    )
    blackframe_set_target_role("${target_name}" test)
    target_link_libraries(
        "${target_name}"
        PRIVATE
            Blackframe::Renderer
            Blackframe::OpenExr
            Blackframe::ProjectOptions
            GTest::gtest_main
    )
    target_compile_definitions(
        "${target_name}"
        PRIVATE
            BLACKFRAME_CORNELL_REFERENCE_DIR="${blackframe_cornell_reference_dir}"
            BLACKFRAME_CORNELL_SCENE_SHA256="${scene_sha256}"
            BLACKFRAME_CORNELL_SOURCE_BASE_COMMIT="${blackframe_cornell_source_base_commit}"
            BLACKFRAME_CORNELL_VARIANT_64=${variant_is_64}
    )
    gtest_discover_tests(
        "${target_name}"
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
        TEST_PREFIX "${test_prefix}"
        PROPERTIES
            LABELS "${test_labels}"
            PROCESSORS 8
    )
endfunction()

blackframe_add_cornell_reference_test(
    BlackframeCornellReference64Tests
    "Cornell64."
    1
    "${blackframe_cornell_scene_64_sha256}"
    "regression;cpu"
)
blackframe_add_cornell_reference_test(
    BlackframeCornellReference256Tests
    "Cornell256."
    0
    "${blackframe_cornell_scene_256_sha256}"
    "regression;nightly;cpu"
)

add_test(
    NAME Blackframe.CornellReferences.Manifest
    COMMAND
        "${CMAKE_COMMAND}"
        "-DREPOSITORY_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/ValidateCornellReferences.cmake"
)
set_tests_properties(
    Blackframe.CornellReferences.Manifest
    PROPERTIES
        LABELS "regression;cpu"
)

add_executable(
    BlackframeCornellReferenceGenerator
    EXCLUDE_FROM_ALL
    CornellDiffuseReferenceGenerator.cpp
)
blackframe_set_target_role(BlackframeCornellReferenceGenerator tool)
target_link_libraries(
    BlackframeCornellReferenceGenerator
    PRIVATE
        Blackframe::Exr
        Blackframe::ProjectOptions
)
target_compile_definitions(
    BlackframeCornellReferenceGenerator
    PRIVATE
        BLACKFRAME_CORNELL_SCENE_64_SHA256="${blackframe_cornell_expected_scene_64_sha256}"
        BLACKFRAME_CORNELL_SCENE_256_SHA256="${blackframe_cornell_expected_scene_256_sha256}"
)
set_target_properties(
    BlackframeCornellReferenceGenerator
    PROPERTIES
        OUTPUT_NAME blackframe_cornell_reference_generator
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
        Blackframe.RayDifferentialSpecularPreview
        RayDifferentialTest.WritesStableMirrorAndTransmissionFootprintPreview
        checksum-ray-differential-specular-800x800.png
        fe65ffb737101b249142418bf109bf9b5e36a9945f00dfe80c38c8cac573dd63
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
