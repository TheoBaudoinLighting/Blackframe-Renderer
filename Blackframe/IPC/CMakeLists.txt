add_library(BlackframeIpc STATIC)
add_library(Blackframe::IPC ALIAS BlackframeIpc)
blackframe_set_target_role(BlackframeIpc core)

target_sources(
    BlackframeIpc
    PRIVATE
        Source/ProtocolCodec.cpp
)

if(WIN32)
    target_sources(
        BlackframeIpc
        PRIVATE
            Source/LocalTransportWindows.cpp
    )
elseif(UNIX)
    target_sources(
        BlackframeIpc
        PRIVATE
            Source/LocalTransportPosix.cpp
    )
else()
    message(FATAL_ERROR "Blackframe IPC has no local transport for this platform.")
endif()

target_include_directories(
    BlackframeIpc
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
)

target_compile_features(BlackframeIpc PUBLIC cxx_std_26)
target_link_libraries(BlackframeIpc PUBLIC Blackframe::ProjectOptions)

set_target_properties(
    BlackframeIpc
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
