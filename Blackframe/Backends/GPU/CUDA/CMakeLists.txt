add_library(
    BlackframeCudaScene
    STATIC
        Source/ClosestHit.cpp
        Source/Occlusion.cpp
        Source/SceneBvh.cpp
        Source/SceneQuery.cpp
        Source/SceneSoA.cpp
)
add_library(Blackframe::CudaScene ALIAS BlackframeCudaScene)
blackframe_set_target_role(BlackframeCudaScene cuda)

target_sources(
    BlackframeCudaScene
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/Backends/GPU/CUDA/ClosestHit.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/Backends/GPU/CUDA/Occlusion.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/Backends/GPU/CUDA/SceneBvh.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/Backends/GPU/CUDA/SceneSoA.hpp"
)

target_include_directories(
    BlackframeCudaScene
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    BlackframeCudaScene
    PUBLIC
        Blackframe::CudaMemory
        Blackframe::CudaSceneKernel
        Blackframe::SceneGeometry
        Blackframe::XpuHostDeviceAbi
    PRIVATE
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
)

set_target_properties(
    BlackframeCudaScene
    PROPERTIES
        OUTPUT_NAME BlackframeCudaScene
        DEBUG_POSTFIX d
)

add_library(
    BlackframeCudaQueues
    STATIC
        Source/WavefrontQueues.cpp
)
add_library(Blackframe::CudaQueues ALIAS BlackframeCudaQueues)
blackframe_set_target_role(BlackframeCudaQueues cuda)

target_sources(
    BlackframeCudaQueues
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/Backends/GPU/CUDA/WavefrontQueues.hpp"
)

target_include_directories(
    BlackframeCudaQueues
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    BlackframeCudaQueues
    PUBLIC
        Blackframe::CudaMemory
        Blackframe::CudaQueueKernel
        Blackframe::Renderer
        Blackframe::XpuHostDeviceAbi
    PRIVATE
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
)

set_target_properties(
    BlackframeCudaQueues
    PROPERTIES
        OUTPUT_NAME BlackframeCudaQueues
        DEBUG_POSTFIX d
)

add_library(
    BlackframeCudaWavefrontTransport
    STATIC
        Source/WavefrontTransport.cpp
)
add_library(Blackframe::CudaWavefrontTransport ALIAS BlackframeCudaWavefrontTransport)
blackframe_set_target_role(BlackframeCudaWavefrontTransport cuda)

target_sources(
    BlackframeCudaWavefrontTransport
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/Backends/GPU/CUDA/WavefrontTransport.hpp"
)

target_include_directories(
    BlackframeCudaWavefrontTransport
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    BlackframeCudaWavefrontTransport
    PUBLIC
        Blackframe::CudaQueues
        Blackframe::CudaScene
        Blackframe::Renderer
    PRIVATE
        Blackframe::CudaAsyncRuntime
        Blackframe::CudaToolkit
        Blackframe::CudaWavefrontStageKernel
        Blackframe::ProjectOptions
        Blackframe::CudaNvtx3
)

set_target_properties(
    BlackframeCudaWavefrontTransport
    PROPERTIES
        OUTPUT_NAME BlackframeCudaWavefrontTransport
        DEBUG_POSTFIX d
)
