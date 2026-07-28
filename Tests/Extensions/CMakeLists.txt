add_executable(
    BlackframeExtensionTests
    ExtensionModuleTest.cpp
)
blackframe_set_target_role(BlackframeExtensionTests test)

add_library(BlackframeMissingQueryExtension MODULE Fixtures/MissingQueryExtension.cpp)
blackframe_set_target_role(BlackframeMissingQueryExtension test)
target_link_libraries(BlackframeMissingQueryExtension PRIVATE Blackframe::ProjectOptions)
set_target_properties(
    BlackframeMissingQueryExtension
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME BlackframeMissingQueryExtension
)

add_library(BlackframeIncompatibleExtension MODULE Fixtures/IncompatibleExtension.cpp)
blackframe_set_target_role(BlackframeIncompatibleExtension test)
target_link_libraries(
    BlackframeIncompatibleExtension
    PRIVATE
        Blackframe::ExtensionAbi
        Blackframe::ProjectOptions
)
target_compile_definitions(
    BlackframeIncompatibleExtension
    PRIVATE
        BLACKFRAME_EXTENSION_IMPLEMENTATION=1
)
set_target_properties(
    BlackframeIncompatibleExtension
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME BlackframeIncompatibleExtension
)

add_library(BlackframeLifecycleExtension MODULE Fixtures/LifecycleExtension.cpp)
blackframe_set_target_role(BlackframeLifecycleExtension test)
target_link_libraries(
    BlackframeLifecycleExtension
    PRIVATE
        Blackframe::ExtensionAbi
        Blackframe::ProjectOptions
)
target_compile_definitions(
    BlackframeLifecycleExtension
    PRIVATE
        BLACKFRAME_EXTENSION_IMPLEMENTATION=1
)
set_target_properties(
    BlackframeLifecycleExtension
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME BlackframeLifecycleExtension
)

target_link_libraries(
    BlackframeExtensionTests
    PRIVATE
        Blackframe::ExtensionLoader
        Blackframe::ProjectOptions
        GTest::gtest_main
)

add_dependencies(
    BlackframeExtensionTests
    BlackframeMissingQueryExtension
    BlackframeIncompatibleExtension
    BlackframeLifecycleExtension
)
target_compile_definitions(
    BlackframeExtensionTests
    PRIVATE
        BLACKFRAME_MISSING_QUERY_EXTENSION_PATH="$<TARGET_FILE:BlackframeMissingQueryExtension>"
        BLACKFRAME_INCOMPATIBLE_EXTENSION_PATH="$<TARGET_FILE:BlackframeIncompatibleExtension>"
        BLACKFRAME_LIFECYCLE_EXTENSION_PATH="$<TARGET_FILE:BlackframeLifecycleExtension>"
)

gtest_discover_tests(
    BlackframeExtensionTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "smoke;regression;nightly;cpu"
)

add_executable(
    BlackframeExtensionAbiCTest
    ExtensionAbiCCompilation.c
)
blackframe_set_target_role(BlackframeExtensionAbiCTest test)

target_link_libraries(
    BlackframeExtensionAbiCTest
    PRIVATE
        Blackframe::ExtensionAbi
        Blackframe::ProjectOptions
)
set_target_properties(
    BlackframeExtensionAbiCTest
    PROPERTIES
        C_STANDARD 17
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
)

add_test(NAME Blackframe.ExtensionAbi.CCompilation COMMAND BlackframeExtensionAbiCTest)
set_tests_properties(
    Blackframe.ExtensionAbi.CCompilation
    PROPERTIES
        LABELS "unit;cpu"
)
