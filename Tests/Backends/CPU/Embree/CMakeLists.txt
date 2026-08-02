add_executable(
    BlackframeEmbreeSmokeTests
    EmbreeSmokeTest.cpp
    ShadowRayImageTest.cpp
)
blackframe_set_target_role(BlackframeEmbreeSmokeTests test)

target_link_libraries(
    BlackframeEmbreeSmokeTests
    PRIVATE
        Blackframe::CpuEmbree
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET Blackframe::Png)
    target_link_libraries(
        BlackframeEmbreeSmokeTests
        PRIVATE
            Blackframe::Png
            Blackframe::Stb
    )
    target_compile_definitions(
        BlackframeEmbreeSmokeTests
        PRIVATE
            BLACKFRAME_HAS_PNG_PREVIEW=1
    )
else()
    target_compile_definitions(
        BlackframeEmbreeSmokeTests
        PRIVATE
            BLACKFRAME_HAS_PNG_PREVIEW=0
    )
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
target_compile_definitions(
    BlackframeEmbreeSmokeTests
    PRIVATE
        BLACKFRAME_EMBREE_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
)

gtest_discover_tests(
    BlackframeEmbreeSmokeTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;smoke;regression;cpu"
)

if(TARGET Blackframe::Png)
    add_test(
        NAME Blackframe.EmbreeSharpShadowParityAtlas
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeEmbreeSmokeTests>"
            "-DGTEST_FILTER=ShadowRayImageTest.WritesStableAnalyticAndEmbreeSharpShadowAtlas"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-shadow-ray-analytic-embree.png"
            "-DEXPECTED_SHA256=f0eaa0bdc913057cc6bb54f9371027624907cc9a08a76696f2191088677590eb"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../../../Renderer/PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.EmbreeSharpShadowParityAtlas
        PROPERTIES
            LABELS "regression;cpu"
    )
endif()

add_executable(
    BlackframeEmbreeNeeRegressionTests
    NextEventEstimationImageTest.cpp
)
blackframe_set_target_role(BlackframeEmbreeNeeRegressionTests test)

target_link_libraries(
    BlackframeEmbreeNeeRegressionTests
    PRIVATE
        Blackframe::CpuEmbree
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET Blackframe::Png)
    target_link_libraries(
        BlackframeEmbreeNeeRegressionTests
        PRIVATE
            Blackframe::Png
            Blackframe::Stb
    )
    target_compile_definitions(
        BlackframeEmbreeNeeRegressionTests
        PRIVATE
            BLACKFRAME_HAS_PNG_PREVIEW=1
    )
else()
    target_compile_definitions(
        BlackframeEmbreeNeeRegressionTests
        PRIVATE
            BLACKFRAME_HAS_PNG_PREVIEW=0
    )
endif()

target_compile_definitions(
    BlackframeEmbreeNeeRegressionTests
    PRIVATE
        BLACKFRAME_EMBREE_NEE_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
)

gtest_discover_tests(
    BlackframeEmbreeNeeRegressionTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "regression;cpu"
        TIMEOUT 180
)

if(TARGET Blackframe::Png)
    add_test(
        NAME Blackframe.EmbreePointLightNee64SppImage
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeEmbreeNeeRegressionTests>"
            "-DGTEST_FILTER=NextEventEstimationImageTest.WritesStableEmbreePointLightImageAt64Spp"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-point-light-nee-64spp.png"
            "-DEXPECTED_SHA256=7913b44f675b2db9742ab9aee4bcaecc641bc6c698f064f7b786f93e9e9ff18e"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../../../Renderer/PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.EmbreePointLightNee64SppImage
        PROPERTIES
            LABELS "regression;cpu"
            TIMEOUT 180
    )
endif()

add_executable(
    BlackframeEmbreeMisRegressionTests
    VeachMisImageTest.cpp
)
blackframe_set_target_role(BlackframeEmbreeMisRegressionTests test)

target_link_libraries(
    BlackframeEmbreeMisRegressionTests
    PRIVATE
        Blackframe::CpuEmbree
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET Blackframe::Png)
    target_link_libraries(
        BlackframeEmbreeMisRegressionTests
        PRIVATE
            Blackframe::Png
            Blackframe::Stb
    )
    target_compile_definitions(
        BlackframeEmbreeMisRegressionTests
        PRIVATE
            BLACKFRAME_HAS_PNG_PREVIEW=1
    )
else()
    target_compile_definitions(
        BlackframeEmbreeMisRegressionTests
        PRIVATE
            BLACKFRAME_HAS_PNG_PREVIEW=0
    )
endif()

target_compile_definitions(
    BlackframeEmbreeMisRegressionTests
    PRIVATE
        BLACKFRAME_EMBREE_MIS_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
)

gtest_discover_tests(
    BlackframeEmbreeMisRegressionTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "regression;cpu"
        TIMEOUT 300
)

if(TARGET Blackframe::Png)
    add_test(
        NAME Blackframe.EmbreeVeachMisPower128SppImage
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:BlackframeEmbreeMisRegressionTests>"
            "-DGTEST_FILTER=VeachMisImageTest.WritesNeutralStablePowerHeuristicPreview"
            "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/artifacts/checksum-veach-mis-power-128spp.png"
            "-DEXPECTED_SHA256=76ff012ade3552e0ef58a725c8d478c8993cf1c0300bc968be6ee42ebb59b63c"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../../../Renderer/PngChecksumTest.cmake"
    )
    set_tests_properties(
        Blackframe.EmbreeVeachMisPower128SppImage
        PROPERTIES
            LABELS "regression;cpu"
            TIMEOUT 300
    )
endif()

add_executable(
    BlackframeEmbreeCornellRegressionTests
    CornellFrameSceneIntegrationTest.cpp
    CornellWavefrontParityTest.cpp
)
blackframe_set_target_role(BlackframeEmbreeCornellRegressionTests test)

target_link_libraries(
    BlackframeEmbreeCornellRegressionTests
    PRIVATE
        Blackframe::CpuEmbree
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeEmbreeCornellRegressionTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "smoke;regression;cpu"
        TIMEOUT 120
)

add_executable(
    BlackframeEmbreeWavefrontMisTests
    WavefrontMisTransportTest.cpp
)
blackframe_set_target_role(BlackframeEmbreeWavefrontMisTests test)

target_link_libraries(
    BlackframeEmbreeWavefrontMisTests
    PRIVATE
        Blackframe::CpuEmbree
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeEmbreeWavefrontMisTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;regression;cpu"
        TIMEOUT 120
)

add_test(
    NAME Blackframe.EmbreeScalarWavefrontParity.Json
    COMMAND
        "${CMAKE_COMMAND}"
        "-DBINARY_DIRECTORY=${PROJECT_BINARY_DIR}"
        "-DSOURCE_DIRECTORY=${PROJECT_SOURCE_DIR}"
        "-DBUILD_CONFIGURATION=$<CONFIG>"
        "-DEMBREE_ENABLED=${BLACKFRAME_ENABLE_EMBREE}"
        "-DCUDA_ENABLED=${BLACKFRAME_ENABLE_CUDA}"
        "-DOUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/reports/scalar-wavefront-parity.json"
        "-DWAVEFRONT_TEST_EXECUTABLE=$<TARGET_FILE:BlackframeEmbreeWavefrontMisTests>"
        "-DNEE_TEST_EXECUTABLE=$<TARGET_FILE:BlackframeEmbreeNeeRegressionTests>"
        "-DVEACH_TEST_EXECUTABLE=$<TARGET_FILE:BlackframeEmbreeMisRegressionTests>"
        "-DCORNELL_TEST_EXECUTABLE=$<TARGET_FILE:BlackframeEmbreeCornellRegressionTests>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/ValidateCpuWavefrontParityJson.cmake"
)
set_tests_properties(
    Blackframe.EmbreeScalarWavefrontParity.Json
    PROPERTIES
        LABELS "regression;cpu"
        TIMEOUT 180
)
