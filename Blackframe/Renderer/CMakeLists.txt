add_library(
    BlackframeRenderer
    STATIC
        Include/Blackframe/Renderer/Bounds.hpp
        Include/Blackframe/Renderer/CapabilityRegistry.hpp
        Include/Blackframe/Renderer/Color.hpp
        Include/Blackframe/Renderer/DebugIntegrators.hpp
        Include/Blackframe/Renderer/Disk.hpp
        Include/Blackframe/Renderer/DisplayPsnr.hpp
        Include/Blackframe/Renderer/DisplayTransform.hpp
        Include/Blackframe/Renderer/Film.hpp
        Include/Blackframe/Renderer/FilmTile.hpp
        Include/Blackframe/Renderer/GeometryOperations.hpp
        Include/Blackframe/Renderer/GeometryTypes.hpp
        Include/Blackframe/Renderer/HdrMetrics.hpp
        Include/Blackframe/Renderer/IndependentSampler.hpp
        Include/Blackframe/Renderer/Interval.hpp
        Include/Blackframe/Renderer/LocalFrame.hpp
        Include/Blackframe/Renderer/LocalPcg32.hpp
        Include/Blackframe/Renderer/LinearMetrics.hpp
        Include/Blackframe/Renderer/MatrixOperations.hpp
        Include/Blackframe/Renderer/MatrixTypes.hpp
        Include/Blackframe/Renderer/NumericConversion.hpp
        Include/Blackframe/Renderer/NumericPrecision.hpp
        Include/Blackframe/Renderer/OwenScrambling.hpp
        Include/Blackframe/Renderer/PinholeCamera.hpp
        Include/Blackframe/Renderer/PixelJitter.hpp
        Include/Blackframe/Renderer/Plane.hpp
        Include/Blackframe/Renderer/Quaternion.hpp
        Include/Blackframe/Renderer/Ray.hpp
        Include/Blackframe/Renderer/RayDiagnostics.hpp
        Include/Blackframe/Renderer/RayOriginOffset.hpp
        Include/Blackframe/Renderer/RenderConfiguration.hpp
        Include/Blackframe/Renderer/RenderSessionState.hpp
        Include/Blackframe/Renderer/SampleDimensionMap.hpp
        Include/Blackframe/Renderer/SampleStream.hpp
        Include/Blackframe/Renderer/SamplingMappings.hpp
        Include/Blackframe/Renderer/SobolSampling.hpp
        Include/Blackframe/Renderer/Spectrum.hpp
        Include/Blackframe/Renderer/Sphere.hpp
        Include/Blackframe/Renderer/StratifiedSampling.hpp
        Include/Blackframe/Renderer/SurfaceInteraction.hpp
        Include/Blackframe/Renderer/Transforms.hpp
        Include/Blackframe/Renderer/TransportConventions.hpp
        Include/Blackframe/Renderer/TransportInterfaces.hpp
        Include/Blackframe/Renderer/Triangle.hpp
        Source/CapabilityRegistry.cpp
        Source/NumericConversion.cpp
        Source/OwenScrambling.cpp
        Source/RayDiagnostics.cpp
        Source/RenderConfiguration.cpp
        Source/RenderConfigurationLimits.hpp
        Source/RenderConfigurationSchema.cpp
        Source/RenderSessionState.cpp
        Source/SampleDimensionMap.cpp
        Source/SobolDirectionNumbers.inc
        Source/SobolSampling.cpp
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

add_library(
    BlackframeExr
    STATIC
        Include/Blackframe/Renderer/ExrWriter.hpp
        Source/ExrWriter.cpp
)
add_library(Blackframe::Exr ALIAS BlackframeExr)
blackframe_set_target_role(BlackframeExr renderer_io)

target_include_directories(
    BlackframeExr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    BlackframeExr
    PUBLIC
        Blackframe::Renderer
    PRIVATE
        Blackframe::OpenExr
        Blackframe::ProjectOptions
)

set_target_properties(
    BlackframeExr
    PROPERTIES
        OUTPUT_NAME BlackframeExr
        DEBUG_POSTFIX d
)

if(TARGET Blackframe::Stb)
    add_library(
        BlackframePng
        STATIC
            Include/Blackframe/Renderer/PngWriter.hpp
            Source/PngWriter.cpp
    )
    add_library(Blackframe::Png ALIAS BlackframePng)
    blackframe_set_target_role(BlackframePng renderer_io)

    target_include_directories(
        BlackframePng
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_link_libraries(
        BlackframePng
        PUBLIC
            Blackframe::Renderer
        PRIVATE
            Blackframe::Stb
            Blackframe::ProjectOptions
    )

    set_target_properties(
        BlackframePng
        PROPERTIES
            OUTPUT_NAME BlackframePng
            DEBUG_POSTFIX d
    )
endif()
