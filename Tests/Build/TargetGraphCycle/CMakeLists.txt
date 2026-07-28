cmake_minimum_required(VERSION 3.30)

project(BlackframeTargetGraphCycleProbe LANGUAGES NONE)

if(NOT DEFINED TARGET_GRAPH_MODULE)
    message(FATAL_ERROR "TARGET_GRAPH_MODULE is required.")
endif()

include("${TARGET_GRAPH_MODULE}")

add_library(BlackframeCycleA INTERFACE)
add_library(BlackframeCycleB INTERFACE)
target_link_libraries(BlackframeCycleA INTERFACE BlackframeCycleB)
target_link_libraries(BlackframeCycleB INTERFACE BlackframeCycleA)
blackframe_set_target_role(BlackframeCycleA core)
blackframe_set_target_role(BlackframeCycleB core)

blackframe_validate_target_graph(
    JSON_OUTPUT "${CMAKE_BINARY_DIR}/target-graph.json"
    DOT_OUTPUT "${CMAKE_BINARY_DIR}/target-graph.dot"
)
