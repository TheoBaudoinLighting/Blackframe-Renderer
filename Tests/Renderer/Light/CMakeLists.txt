add_executable(
    BlackframeRendererLightTests
    AreaLightImageTest.cpp
    AreaLightsTest.cpp
    EnvironmentLightImageTest.cpp
    EnvironmentLightsTest.cpp
    LightSamplerImageTest.cpp
    LightSamplerTest.cpp
    LightTreeImageTest.cpp
    LightTreeTest.cpp
    LightTest.cpp
    PunctualLightImageTest.cpp
    PunctualLightsTest.cpp
)
blackframe_set_target_role(BlackframeRendererLightTests test)

target_link_libraries(
    BlackframeRendererLightTests
    PRIVATE
        Blackframe::Renderer
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET Blackframe::Png)
    target_link_libraries(
        BlackframeRendererLightTests
        PRIVATE
            Blackframe::Png
            Blackframe::Stb
    )
    target_compile_definitions(BlackframeRendererLightTests PRIVATE BLACKFRAME_HAS_PNG_PREVIEW=1)
else()
    target_compile_definitions(BlackframeRendererLightTests PRIVATE BLACKFRAME_HAS_PNG_PREVIEW=0)
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
target_compile_definitions(
    BlackframeRendererLightTests
    PRIVATE
        BLACKFRAME_LIGHT_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
)

gtest_discover_tests(
    BlackframeRendererLightTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

if(TARGET Blackframe::Png)
    add_test(
        NAME Blackframe.AreaLightPdfSidednessAtlas
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeRendererLightTests>"
            "-DGTEST_FILTER=AreaLightImageTest.WritesStablePdfAndSidednessAtlas"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-area-light-pdf-sides.png"
            "-DEXPECTED_SHA256=744d033adce3da1b10e29326b1e6675964f9a7313aa71aedf60bf62d85b90d98"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.AreaLightPdfSidednessAtlas
        PROPERTIES
            LABELS "regression;cpu"
    )

    add_test(
        NAME Blackframe.PunctualLightResponseAtlas
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeRendererLightTests>"
            "-DGTEST_FILTER=PunctualLightImageTest.WritesStableAnalyticResponseAtlas"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-punctual-light-response.png"
            "-DEXPECTED_SHA256=8d030d7ba0495e36adba87464d9c856515679cfcd318881c833580bb78f5e09f"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.PunctualLightResponseAtlas
        PROPERTIES
            LABELS "regression;cpu"
    )

    set(
        blackframe_environment_light_atlas_sha256
        "b72020187bd1f2a8f0fb6bfcfd8dcc3b0510239ace32d3c754d870f80e6620bc"
    )
    add_test(
        NAME Blackframe.EnvironmentLightRadiancePdfAtlas
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeRendererLightTests>"
            "-DGTEST_FILTER=EnvironmentLightImageTest.WritesStableRadianceAndPdfAtlas"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-environment-light-radiance-pdf.png"
            "-DEXPECTED_SHA256=${blackframe_environment_light_atlas_sha256}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.EnvironmentLightRadiancePdfAtlas
        PROPERTIES
            LABELS "regression;cpu"
    )

    set(
        blackframe_light_sampler_atlas_sha256
        "ad19ac06124fc74cda0edfc4a983b35174815da72f5f1bd0dd242b369df9fa2a"
    )
    add_test(
        NAME Blackframe.LightSamplerSelectionAtlas
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeRendererLightTests>"
            "-DGTEST_FILTER=LightSamplerImageTest.WritesStableUniformAndPowerSelectionAtlas"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-light-sampler-selection.png"
            "-DEXPECTED_SHA256=${blackframe_light_sampler_atlas_sha256}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.LightSamplerSelectionAtlas
        PROPERTIES
            LABELS "regression;cpu"
    )

    set(
        blackframe_light_tree_atlas_sha256
        "34e7f57593259e738a90a8de0c2040b25d37051616e49eb7d528e7a669498982"
    )
    add_test(
        NAME Blackframe.LightTreeDistributionAtlas
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeRendererLightTests>"
            "-DGTEST_FILTER=LightTreeImageTest.WritesStableNormalizedDistributionAtlas"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-light-tree-distribution.png"
            "-DEXPECTED_SHA256=${blackframe_light_tree_atlas_sha256}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.LightTreeDistributionAtlas
        PROPERTIES
            LABELS "regression;cpu"
    )
endif()
