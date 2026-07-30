cmake_minimum_required(VERSION 3.30)

function(require_target_role expected_target expected_role)
    string(JSON node_count LENGTH "${target_graph}" nodes)
    if(node_count EQUAL 0)
        message(FATAL_ERROR "Target graph contains no nodes.")
    endif()

    math(EXPR last_node_index "${node_count} - 1")
    foreach(node_index RANGE 0 "${last_node_index}")
        string(JSON node_name GET "${target_graph}" nodes "${node_index}" name)
        if(node_name STREQUAL expected_target)
            string(JSON node_role GET "${target_graph}" nodes "${node_index}" role)
            if(NOT node_role STREQUAL expected_role)
                message(
                    FATAL_ERROR
                    "Target '${expected_target}' has role '${node_role}', "
                    "expected '${expected_role}'."
                )
            endif()
            return()
        endif()
    endforeach()

    message(FATAL_ERROR "Target graph does not contain '${expected_target}'.")
endfunction()

function(target_graph_has_edge source_target dependency_target dependency_kind output_variable)
    string(JSON edge_count LENGTH "${target_graph}" edges)
    if(edge_count EQUAL 0)
        set("${output_variable}" FALSE PARENT_SCOPE)
        return()
    endif()

    math(EXPR last_edge_index "${edge_count} - 1")
    foreach(edge_index RANGE 0 "${last_edge_index}")
        string(JSON edge_source GET "${target_graph}" edges "${edge_index}" from)
        string(JSON edge_dependency GET "${target_graph}" edges "${edge_index}" to)
        string(JSON edge_kind GET "${target_graph}" edges "${edge_index}" kind)
        if(edge_source STREQUAL source_target AND
           edge_dependency STREQUAL dependency_target AND
           edge_kind STREQUAL dependency_kind)
            set("${output_variable}" TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set("${output_variable}" FALSE PARENT_SCOPE)
endfunction()

function(require_target_edge source_target dependency_target dependency_kind)
    target_graph_has_edge(
        "${source_target}"
        "${dependency_target}"
        "${dependency_kind}"
        edge_exists
    )
    if(NOT edge_exists)
        message(
            FATAL_ERROR
            "Target graph does not contain ${dependency_kind} edge "
            "'${source_target}' -> '${dependency_target}'."
        )
    endif()
endfunction()

function(forbid_target_edge source_target dependency_target)
    foreach(dependency_kind IN ITEMS link order)
        target_graph_has_edge(
            "${source_target}"
            "${dependency_target}"
            "${dependency_kind}"
            edge_exists
        )
        if(edge_exists)
            message(
                FATAL_ERROR
                "Forbidden ${dependency_kind} edge '${source_target}' -> "
                "'${dependency_target}' exists."
            )
        endif()
    endforeach()
endfunction()

if(NOT DEFINED TARGET_GRAPH_PATH OR NOT EXISTS "${TARGET_GRAPH_PATH}")
    message(FATAL_ERROR "Target graph does not exist: ${TARGET_GRAPH_PATH}")
endif()

file(READ "${TARGET_GRAPH_PATH}" target_graph)
string(JSON graph_schema_version GET "${target_graph}" schema_version)
if(NOT graph_schema_version EQUAL 1)
    message(FATAL_ERROR "Unsupported target graph schema version '${graph_schema_version}'.")
endif()

string(JSON graph_is_acyclic GET "${target_graph}" acyclic)
if(NOT graph_is_acyclic)
    message(FATAL_ERROR "Configured target graph is not acyclic.")
endif()

require_target_role(BlackframeCore core)
require_target_role(BlackframeProjectOptions core)
require_target_role(BlackframeExr renderer_io)
require_target_role(BlackframeOpenExr dependency)
require_target_role(BlackframeTests test)
require_target_role(BlackframeSceneTests test)
require_target_edge(BlackframeExr BlackframeOpenExr link)
require_target_edge(BlackframeExr BlackframeRenderer link)
forbid_target_edge(BlackframeRenderer BlackframeOpenExr)
if(STB_ENABLED)
    require_target_role(BlackframePng renderer_io)
    require_target_role(BlackframeStb dependency)
    require_target_edge(BlackframePng BlackframeRenderer link)
    require_target_edge(BlackframePng BlackframeStb link)
    forbid_target_edge(BlackframeRenderer BlackframeStb)
endif()
require_target_edge(BlackframeTests BlackframeCoreTests order)
require_target_edge(BlackframeTests BlackframeSceneTests order)
require_target_edge(BlackframeSceneTests BlackframeEngine link)
require_target_role(BlackframeRenderTests test)
require_target_edge(BlackframeTests BlackframeRenderTests order)

if(CPU_EMBREE_ENABLED)
    require_target_role(BlackframeCpuEmbree cpu_embree)
    require_target_edge(BlackframeCpuEmbree BlackframeEmbree link)
    require_target_edge(BlackframeCpuEmbree BlackframeRenderer link)
endif()
if(CUDA_ENABLED)
    require_target_role(BlackframeCuda cuda)
    require_target_edge(BlackframeCuda BlackframeCudaSmokeKernel link)
    forbid_target_edge(BlackframeCuda BlackframeCpuEmbree)
    forbid_target_edge(BlackframeCpuEmbree BlackframeCuda)
endif()
if(TOOLS_ENABLED)
    require_target_role(BlackframeRender tool)
    require_target_role(BlackframeTools tool)
    require_target_edge(BlackframeTools BlackframeRender order)
endif()
if(BENCHMARKS_ENABLED)
    require_target_role(BlackframeBenchmarks benchmark)
    require_target_role(BlackframeRenderBenchmarks benchmark)
    require_target_edge(BlackframeBenchmarks BlackframeRenderBenchmarks order)
endif()
