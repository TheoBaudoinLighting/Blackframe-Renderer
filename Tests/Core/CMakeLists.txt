add_executable(
    BlackframeCoreTests
    StatusTest.cpp
    VersionTest.cpp
)
blackframe_set_target_role(BlackframeCoreTests test)

target_link_libraries(
    BlackframeCoreTests
    PRIVATE
        Blackframe::Core
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCoreTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
)
