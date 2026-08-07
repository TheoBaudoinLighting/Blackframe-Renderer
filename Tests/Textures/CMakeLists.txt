add_executable(
    BlackframeConstantTextureTests
    ConstantTextureTest.cpp
    TextureWrapTest.cpp
)
blackframe_set_target_role(BlackframeConstantTextureTests test)

target_link_libraries(
    BlackframeConstantTextureTests
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::Renderer
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeConstantTextureTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES LABELS "unit"
)

if(TARGET Blackframe::HostImage)
    set(blackframe_host_image_fixture
        "${CMAKE_CURRENT_SOURCE_DIR}/Fixtures/host-image-rgb8.ppm"
    )
    set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${blackframe_host_image_fixture}"
    )
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/artifacts")

    add_executable(
        BlackframeHostImageCacheTests
        HostImageCacheTest.cpp
        HostImageFilterTest.cpp
    )
    blackframe_set_target_role(BlackframeHostImageCacheTests test)
    target_link_libraries(
        BlackframeHostImageCacheTests
        PRIVATE
            Blackframe::HostImage
            Blackframe::Exr
            Blackframe::ProjectOptions
            GTest::gtest_main
    )
    target_compile_definitions(
        BlackframeHostImageCacheTests
        PRIVATE
            BLACKFRAME_HOST_IMAGE_TEST_FIXTURE="${blackframe_host_image_fixture}"
            BLACKFRAME_HOST_IMAGE_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
    )

    if(TARGET Blackframe::Png)
        target_link_libraries(
            BlackframeHostImageCacheTests
            PRIVATE
                Blackframe::Png
        )
        target_compile_definitions(
            BlackframeHostImageCacheTests
            PRIVATE
                BLACKFRAME_HOST_IMAGE_FILTER_PNG=1
        )
    endif()

    gtest_discover_tests(
        BlackframeHostImageCacheTests
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
        PROPERTIES LABELS "unit;smoke;cpu"
    )
endif()
