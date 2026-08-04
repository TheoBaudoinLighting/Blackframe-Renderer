add_executable(
    BlackframeCudaSceneTests
    SceneBvhTest.cpp
    SceneSoATest.cpp
)
blackframe_set_target_role(BlackframeCudaSceneTests test)

target_link_libraries(
    BlackframeCudaSceneTests
    PRIVATE
        Blackframe::CudaScene
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaSceneTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
)
