add_library(
    BlackframeRenderer
    STATIC
        Include/Blackframe/Renderer/RenderConfiguration.hpp
        Include/Blackframe/Renderer/RenderSessionState.hpp
        Include/Blackframe/Renderer/TransportConventions.hpp
        Include/Blackframe/Renderer/TransportInterfaces.hpp
        Source/RenderConfiguration.cpp
        Source/RenderSessionState.cpp
)
add_library(Blackframe::Renderer ALIAS BlackframeRenderer)
blackframe_set_target_role(BlackframeRenderer core)

target_include_directories(
    BlackframeRenderer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    BlackframeRenderer
    PUBLIC
        Blackframe::Core
    PRIVATE
        Blackframe::ProjectOptions
)

set_target_properties(
    BlackframeRenderer
    PROPERTIES
        OUTPUT_NAME BlackframeRenderer
        DEBUG_POSTFIX d
)
