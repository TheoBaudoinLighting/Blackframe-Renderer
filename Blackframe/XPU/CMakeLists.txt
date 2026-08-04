add_library(BlackframeXpuHostDeviceAbi INTERFACE)
add_library(Blackframe::XpuHostDeviceAbi ALIAS BlackframeXpuHostDeviceAbi)
blackframe_set_target_role(BlackframeXpuHostDeviceAbi core)

target_sources(
    BlackframeXpuHostDeviceAbi
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Shared/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Shared/Include/Blackframe/XPU/Shared/SceneBvhAbi.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Shared/Include/Blackframe/XPU/Shared/SceneSoaAbi.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Shared/Include/Blackframe/XPU/Shared/TransportAbi.hpp"
)

target_include_directories(
    BlackframeXpuHostDeviceAbi
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Shared/Include>
        $<INSTALL_INTERFACE:Include>
)

target_link_libraries(
    BlackframeXpuHostDeviceAbi
    INTERFACE
        Blackframe::ProjectOptions
)

add_library(BlackframeXpuBackendAbi INTERFACE)
add_library(Blackframe::XpuBackendAbi ALIAS BlackframeXpuBackendAbi)
blackframe_set_target_role(BlackframeXpuBackendAbi core)

target_include_directories(
    BlackframeXpuBackendAbi
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/BackendAbi/Include>
        $<INSTALL_INTERFACE:Include>
)

target_sources(
    BlackframeXpuBackendAbi
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/BackendAbi/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/BackendAbi/Include/Blackframe/XPU/XpuBackendAbi.h"
)

target_link_libraries(
    BlackframeXpuBackendAbi
    INTERFACE
        Blackframe::ExtensionAbi
)

add_library(BlackframeXpuHost STATIC)
add_library(Blackframe::XpuHost ALIAS BlackframeXpuHost)
blackframe_set_target_role(BlackframeXpuHost core)

target_sources(
    BlackframeXpuHost
    PRIVATE
        Host/Source/XpuBackend.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Host/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Host/Include/Blackframe/XPU/DeviceDescriptor.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Host/Include/Blackframe/XPU/XpuBackend.hpp"
)

target_include_directories(
    BlackframeXpuHost
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Host/Include>
        $<INSTALL_INTERFACE:Include>
)

target_link_libraries(
    BlackframeXpuHost
    PUBLIC
        Blackframe::ExtensionLoader
        Blackframe::XpuBackendAbi
    PRIVATE
        Blackframe::ProjectOptions
)

if(BLACKFRAME_ENABLE_CUDA)
    add_subdirectory(CUDA)
endif()
