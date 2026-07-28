add_library(
    BlackframeCore
    STATIC
        Include/Blackframe/Core/Status.hpp
        Include/Blackframe/Core/Version.hpp
        Source/Status.cpp
        Source/Version.cpp
)
add_library(Blackframe::Core ALIAS BlackframeCore)

target_include_directories(
    BlackframeCore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(BlackframeCore PUBLIC Blackframe::ProjectOptions)

target_compile_definitions(
    BlackframeCore
    PRIVATE
        BLACKFRAME_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        BLACKFRAME_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        BLACKFRAME_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

set_target_properties(
    BlackframeCore
    PROPERTIES
        OUTPUT_NAME BlackframeCore
        DEBUG_POSTFIX d
)
