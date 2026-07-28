add_executable(
    BlackframeRender
    Source/CommandLine.cpp
    Source/DefaultLocalEndpointPosix.cpp
    Source/DefaultLocalEndpointWindows.cpp
    Source/HeadlessMain.cpp
)
blackframe_set_target_role(BlackframeRender tool)

if(WIN32)
    set_source_files_properties(Source/DefaultLocalEndpointPosix.cpp PROPERTIES HEADER_FILE_ONLY YES)
else()
    set_source_files_properties(Source/DefaultLocalEndpointWindows.cpp PROPERTIES HEADER_FILE_ONLY YES)
endif()

target_include_directories(BlackframeRender PRIVATE Source)

target_link_libraries(
    BlackframeRender
    PRIVATE
        Blackframe::Core
        Blackframe::Engine
        Blackframe::IPC
        Blackframe::ProjectOptions
        Blackframe::Renderer
)

set_target_properties(
    BlackframeRender
    PROPERTIES
        OUTPUT_NAME render
        DEBUG_POSTFIX ""
)

add_custom_target(BlackframeTools)
add_dependencies(BlackframeTools BlackframeRender)
blackframe_set_target_role(BlackframeTools tool)
