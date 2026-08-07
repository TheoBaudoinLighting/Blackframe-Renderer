set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_library(
    BlackframeRenderer
    STATIC
        Include/Blackframe/Renderer/AreaLights.hpp
        Include/Blackframe/Renderer/BasicSpectra.hpp
        Include/Blackframe/Renderer/Bounds.hpp
        Include/Blackframe/Renderer/BsdfOnlyPathLoop.hpp
        Include/Blackframe/Renderer/CapabilityRegistry.hpp
        Include/Blackframe/Renderer/Cie1931Sensor.hpp
        Include/Blackframe/Renderer/ClosureMixture.hpp
        Include/Blackframe/Renderer/ClosureSet.hpp
        Include/Blackframe/Renderer/Color.hpp
        Include/Blackframe/Renderer/ConvergenceMetrics.hpp
        Include/Blackframe/Renderer/ConstantTexture.hpp
        Include/Blackframe/Renderer/CpuWavefrontScheduler.hpp
        Include/Blackframe/Renderer/DebugIntegrators.hpp
        Include/Blackframe/Renderer/Detail/BsdfOnlyPathLoop.hpp
        Include/Blackframe/Renderer/Detail/DepthFilteredClosureMixture.hpp
        Include/Blackframe/Renderer/Detail/PacketLightSpectrum.hpp
        Include/Blackframe/Renderer/Disk.hpp
        Include/Blackframe/Renderer/DisplayPsnr.hpp
        Include/Blackframe/Renderer/DisplayTransform.hpp
        Include/Blackframe/Renderer/DirectLighting.hpp
        Include/Blackframe/Renderer/Emission.hpp
        Include/Blackframe/Renderer/EnvironmentLights.hpp
        Include/Blackframe/Renderer/Film.hpp
        Include/Blackframe/Renderer/FilmTile.hpp
        Include/Blackframe/Renderer/Fresnel.hpp
        Include/Blackframe/Renderer/GeometryOperations.hpp
        Include/Blackframe/Renderer/GeometryTypes.hpp
        Include/Blackframe/Renderer/GgxMicrofacet.hpp
        Include/Blackframe/Renderer/HdrMetrics.hpp
        Include/Blackframe/Renderer/IndependentSampler.hpp
        Include/Blackframe/Renderer/Interval.hpp
        Include/Blackframe/Renderer/LambertianReflection.hpp
        Include/Blackframe/Renderer/Light.hpp
        Include/Blackframe/Renderer/LightSampler.hpp
        Include/Blackframe/Renderer/LocalFrame.hpp
        Include/Blackframe/Renderer/LocalPcg32.hpp
        Include/Blackframe/Renderer/LinearMetrics.hpp
        Include/Blackframe/Renderer/MatrixOperations.hpp
        Include/Blackframe/Renderer/MatrixTypes.hpp
        Include/Blackframe/Renderer/MisHeuristics.hpp
        Include/Blackframe/Renderer/NumericConversion.hpp
        Include/Blackframe/Renderer/NumericPrecision.hpp
        Include/Blackframe/Renderer/OwenScrambling.hpp
        Include/Blackframe/Renderer/PathDepthLimits.hpp
        Include/Blackframe/Renderer/PathState.hpp
        Include/Blackframe/Renderer/PathStateDiagnostics.hpp
        Include/Blackframe/Renderer/PathStateSoA.hpp
        Include/Blackframe/Renderer/PinholeCamera.hpp
        Include/Blackframe/Renderer/PixelJitter.hpp
        Include/Blackframe/Renderer/Plane.hpp
        Include/Blackframe/Renderer/PunctualLights.hpp
        Include/Blackframe/Renderer/Quaternion.hpp
        Include/Blackframe/Renderer/Ray.hpp
        Include/Blackframe/Renderer/RayDiagnostics.hpp
        Include/Blackframe/Renderer/RayOriginOffset.hpp
        Include/Blackframe/Renderer/RenderConfiguration.hpp
        Include/Blackframe/Renderer/RenderSessionState.hpp
        Include/Blackframe/Renderer/RussianRoulette.hpp
        Include/Blackframe/Renderer/RoughConductorReflection.hpp
        Include/Blackframe/Renderer/RoughDielectric.hpp
        Include/Blackframe/Renderer/RoughDiffuseReflection.hpp
        Include/Blackframe/Renderer/SampleDimensionMap.hpp
        Include/Blackframe/Renderer/SampleStream.hpp
        Include/Blackframe/Renderer/SamplingMappings.hpp
        Include/Blackframe/Renderer/SceneIdentifiers.hpp
        Include/Blackframe/Renderer/ShadowRay.hpp
        Include/Blackframe/Renderer/ShadingNormalCorrection.hpp
        Include/Blackframe/Renderer/SobolSampling.hpp
        Include/Blackframe/Renderer/Spectrum.hpp
        Include/Blackframe/Renderer/SpecularDelta.hpp
        Include/Blackframe/Renderer/Sphere.hpp
        Include/Blackframe/Renderer/StratifiedSampling.hpp
        Include/Blackframe/Renderer/SurfaceInteraction.hpp
        Include/Blackframe/Renderer/TextureColorSpace.hpp
        Include/Blackframe/Renderer/TextureWrap.hpp
        Include/Blackframe/Renderer/Transforms.hpp
        Include/Blackframe/Renderer/TransportConventionDiagnostics.hpp
        Include/Blackframe/Renderer/TransportConventions.hpp
        Include/Blackframe/Renderer/TransportInterfaces.hpp
        Include/Blackframe/Renderer/Triangle.hpp
        Include/Blackframe/Renderer/WavefrontQueues.hpp
        Include/Blackframe/Renderer/WavelengthSampling.hpp
        Source/BsdfOnlyPathLoop.cpp
        Source/CapabilityRegistry.cpp
        Source/Cie1931ColorMatchingFunctions.inc
        Source/Cie1931Sensor.cpp
        Source/CpuWavefrontScheduler.cpp
        Source/LightSampler.cpp
        Source/NumericConversion.cpp
        Source/OwenScrambling.cpp
        Source/PathDepthLimits.cpp
        Source/PathStateDiagnostics.cpp
        Source/PathStateSoA.cpp
        Source/RayDiagnostics.cpp
        Source/RenderConfiguration.cpp
        Source/RenderConfigurationLimits.hpp
        Source/RenderConfigurationSchema.cpp
        Source/RenderSessionState.cpp
        Source/RussianRoulette.cpp
        Source/SampleDimensionMap.cpp
        Source/SobolDirectionNumbers.inc
        Source/SobolSampling.cpp
        Source/TransportConventionDiagnostics.cpp
        Source/WavefrontQueues.cpp
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
        Threads::Threads
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

if(TARGET Blackframe::OpenImageIo)
    add_library(
        BlackframeHostImage
        STATIC
            Include/Blackframe/Renderer/HostImageCache.hpp
            Include/Blackframe/Renderer/HostImageFilter.hpp
            Include/Blackframe/Renderer/HostImageMipChain.hpp
            Source/HostImageCache.cpp
            Source/HostImageFilter.cpp
            Source/HostImageMipChain.cpp
    )
    add_library(Blackframe::HostImage ALIAS BlackframeHostImage)
    blackframe_set_target_role(BlackframeHostImage renderer_io)

    target_include_directories(
        BlackframeHostImage
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_link_libraries(
        BlackframeHostImage
        PUBLIC
            Blackframe::Renderer
        PRIVATE
            Blackframe::OpenImageIo
            Blackframe::ProjectOptions
    )

    set_target_properties(
        BlackframeHostImage
        PROPERTIES
            OUTPUT_NAME BlackframeHostImage
            DEBUG_POSTFIX d
    )
endif()

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
