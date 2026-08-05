blackframe_find_cuda_toolkit()
add_subdirectory(Host)

enable_language(CUDA)
blackframe_require_cuda_cxx20()

if(WIN32 AND CMAKE_CUDA_SIMULATE_ID STREQUAL "MSVC")
    # NVCC emits MSVC librarian syntax; keep its compatible archiver local to
    # this CUDA-only directory.
    find_program(
        blackframe_cuda_librarian
        NAMES lib.exe
        REQUIRED
        NO_CACHE
    )
    set(CMAKE_AR "${blackframe_cuda_librarian}")
endif()

add_library(BlackframeCudaSharedHeaders INTERFACE)
add_library(Blackframe::CudaSharedHeaders ALIAS BlackframeCudaSharedHeaders)
blackframe_set_target_role(BlackframeCudaSharedHeaders cuda)

target_sources(
    BlackframeCudaSharedHeaders
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/XPU/CUDA/SceneClosestHit.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/XPU/CUDA/SceneSoaHash.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/XPU/CUDA/SmokeKernel.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/XPU/CUDA/SmokeKernelPayload.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Include/Blackframe/XPU/CUDA/TransportAbiProbe.hpp"
)

target_include_directories(
    BlackframeCudaSharedHeaders
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:Include>
)

target_link_libraries(
    BlackframeCudaSharedHeaders
    INTERFACE
        Blackframe::ProjectOptions
        Blackframe::XpuHostDeviceAbi
)

add_library(
    BlackframeCudaSmokeKernel
    STATIC
        Source/CudaSharedHeaderContract.cu
        Source/CudaSmokeKernel.cu
        Source/CudaTransportAbiProbe.cu
)
add_library(Blackframe::CudaSmokeKernel ALIAS BlackframeCudaSmokeKernel)
blackframe_set_target_role(BlackframeCudaSmokeKernel cuda)

blackframe_configure_cuda_target(BlackframeCudaSmokeKernel)

target_link_libraries(
    BlackframeCudaSmokeKernel
    PUBLIC
        Blackframe::CudaSharedHeaders
    PRIVATE
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
)

add_library(
    BlackframeCudaSceneKernel
    STATIC
        Source/CudaSceneClosestHit.cu
        Source/CudaSceneBvhAbiContract.cu
        Source/CudaSceneSoaHash.cu
)
add_library(Blackframe::CudaSceneKernel ALIAS BlackframeCudaSceneKernel)
blackframe_set_target_role(BlackframeCudaSceneKernel cuda)

blackframe_configure_cuda_target(BlackframeCudaSceneKernel)

target_link_libraries(
    BlackframeCudaSceneKernel
    PUBLIC
        Blackframe::CudaSharedHeaders
    PRIVATE
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
)

add_library(BlackframeCuda INTERFACE)
add_library(Blackframe::Cuda ALIAS BlackframeCuda)

target_link_libraries(
    BlackframeCuda
    INTERFACE
        Blackframe::CudaMemory
        Blackframe::CudaSmokeKernel
        Blackframe::ProjectOptions
)

blackframe_set_target_role(BlackframeCuda cuda)
