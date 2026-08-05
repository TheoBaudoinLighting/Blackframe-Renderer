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
