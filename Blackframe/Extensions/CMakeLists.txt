add_library(BlackframeExtensionAbi INTERFACE)
add_library(Blackframe::ExtensionAbi ALIAS BlackframeExtensionAbi)
blackframe_set_target_role(BlackframeExtensionAbi core)

target_include_directories(
    BlackframeExtensionAbi
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Abi/Include>
        $<INSTALL_INTERFACE:Include>
)

target_sources(
    BlackframeExtensionAbi
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Abi/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Abi/Include/Blackframe/Extensions/ExtensionAbi.h"
)

add_library(BlackframeExtensionLoader STATIC)
add_library(Blackframe::ExtensionLoader ALIAS BlackframeExtensionLoader)
blackframe_set_target_role(BlackframeExtensionLoader core)

target_sources(
    BlackframeExtensionLoader
    PRIVATE
        Loader/Source/DynamicLibrary.cpp
        Loader/Source/ExtensionModule.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/Loader/Include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/Loader/Include/Blackframe/Extensions/DynamicLibrary.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Loader/Include/Blackframe/Extensions/ExtensionModule.hpp"
)

target_include_directories(
    BlackframeExtensionLoader
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Loader/Include>
        $<INSTALL_INTERFACE:Include>
)

target_link_libraries(
    BlackframeExtensionLoader
    PUBLIC
        Blackframe::ExtensionAbi
    PRIVATE
        Blackframe::ProjectOptions
        ${CMAKE_DL_LIBS}
)

if(WIN32)
    target_compile_definitions(
        BlackframeExtensionLoader
        PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
    )
endif()
