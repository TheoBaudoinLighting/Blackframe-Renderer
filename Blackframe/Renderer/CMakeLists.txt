add_library(
    BlackframeRenderer
    STATIC
        Include/Blackframe/Renderer/CapabilityRegistry.hpp
        Include/Blackframe/Renderer/GeometryOperations.hpp
        Include/Blackframe/Renderer/GeometryTypes.hpp
        Include/Blackframe/Renderer/LocalFrame.hpp
        Include/Blackframe/Renderer/MatrixOperations.hpp
        Include/Blackframe/Renderer/MatrixTypes.hpp
        Include/Blackframe/Renderer/NumericConversion.hpp
        Include/Blackframe/Renderer/NumericPrecision.hpp
        Include/Blackframe/Renderer/Quaternion.hpp
        Include/Blackframe/Renderer/RenderConfiguration.hpp
        Include/Blackframe/Renderer/RenderSessionState.hpp
        Include/Blackframe/Renderer/Transforms.hpp
        Include/Blackframe/Renderer/TransportConventions.hpp
        Include/Blackframe/Renderer/TransportInterfaces.hpp
        Source/CapabilityRegistry.cpp
        Source/NumericConversion.cpp
        Source/RenderConfiguration.cpp
        Source/RenderConfigurationLimits.hpp
        Source/RenderConfigurationSchema.cpp
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

target_compile_definitions(
    BlackframeRenderer
    PRIVATE
        BLACKFRAME_CAPABILITY_REFERENCE_CPU=$<BOOL:${BLACKFRAME_BUILD_REFERENCE_PLUGINS}>
        BLACKFRAME_CAPABILITY_EMBREE=$<BOOL:${BLACKFRAME_ENABLE_EMBREE}>
        BLACKFRAME_CAPABILITY_CUDA=$<BOOL:${BLACKFRAME_ENABLE_CUDA}>
)

set_target_properties(
    BlackframeRenderer
    PROPERTIES
        OUTPUT_NAME BlackframeRenderer
        DEBUG_POSTFIX d
)
