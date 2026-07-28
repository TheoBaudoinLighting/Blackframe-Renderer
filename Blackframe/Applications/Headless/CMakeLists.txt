add_executable(
    BlackframeHeadless
    Source/CommandLine.cpp
    Source/DefaultLocalEndpointPosix.cpp
    Source/DefaultLocalEndpointWindows.cpp
    Source/HeadlessMain.cpp
)
blackframe_set_target_role(BlackframeHeadless tool)

if(WIN32)
    set_source_files_properties(Source/DefaultLocalEndpointPosix.cpp PROPERTIES HEADER_FILE_ONLY YES)
else()
    set_source_files_properties(Source/DefaultLocalEndpointWindows.cpp PROPERTIES HEADER_FILE_ONLY YES)
endif()

target_include_directories(BlackframeHeadless PRIVATE Source)

target_link_libraries(
    BlackframeHeadless
    PRIVATE
        Blackframe::Core
        Blackframe::Engine
        Blackframe::IPC
        Blackframe::ProjectOptions
)

set_target_properties(
    BlackframeHeadless
    PROPERTIES
        OUTPUT_NAME Blackframe
        DEBUG_POSTFIX ""
)

add_custom_target(BlackframeTools)
add_dependencies(BlackframeTools BlackframeHeadless)
blackframe_set_target_role(BlackframeTools tool)
