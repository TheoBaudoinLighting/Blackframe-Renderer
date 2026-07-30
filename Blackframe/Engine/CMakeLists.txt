add_library(
    BlackframeEngine
    STATIC
        Include/Blackframe/Engine/Engine.hpp
        Include/Blackframe/Engine/FrameScene.hpp
        Source/Engine.cpp
        Source/FrameScene.cpp
)
add_library(Blackframe::Engine ALIAS BlackframeEngine)
blackframe_set_target_role(BlackframeEngine core)

target_include_directories(
    BlackframeEngine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    BlackframeEngine
    PUBLIC
        Blackframe::Core
        Blackframe::IPC
        Blackframe::Renderer
        Blackframe::XpuHost
    PRIVATE
        Blackframe::ProjectOptions
)

set_target_properties(
    BlackframeEngine
    PROPERTIES
        OUTPUT_NAME BlackframeEngine
        DEBUG_POSTFIX d
)
