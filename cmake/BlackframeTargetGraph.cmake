include_guard(GLOBAL)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(
    BLACKFRAME_TARGET_ROLES
    benchmark
    core
    cpu_embree
    cuda
    dependency
    reference
    renderer_io
    test
    tool
)

function(blackframe_set_target_role target role)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Cannot classify unknown target '${target}'.")
    endif()

    if(NOT role IN_LIST BLACKFRAME_TARGET_ROLES)
        message(FATAL_ERROR "Unknown Blackframe target role '${role}' for '${target}'.")
    endif()

    if(role STREQUAL "core")
        set(target_folder "Blackframe/Core")
    elseif(role STREQUAL "cpu_embree")
        set(target_folder "Blackframe/Backends/CPU/Embree")
    elseif(role STREQUAL "cuda")
        set(target_folder "Blackframe/Backends/CUDA")
    elseif(role STREQUAL "dependency")
        set(target_folder "Blackframe/Dependencies")
    elseif(role STREQUAL "reference")
        set(target_folder "Blackframe/Reference")
    elseif(role STREQUAL "renderer_io")
        set(target_folder "Blackframe/Renderer")
    elseif(role STREQUAL "tool")
        set(target_folder "Blackframe/Tools")
    elseif(role STREQUAL "test")
        set(target_folder "Blackframe/Tests")
    elseif(role STREQUAL "benchmark")
        set(target_folder "Blackframe/Benchmarks")
    endif()

    set_property(TARGET "${target}" PROPERTY BLACKFRAME_TARGET_ROLE "${role}")
    set_property(TARGET "${target}" PROPERTY FOLDER "${target_folder}")
endfunction()

function(_blackframe_collect_directory_targets directory output_variable)
    get_property(directory_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(child_directories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)

    set(collected_targets ${directory_targets})
    foreach(child_directory IN LISTS child_directories)
        _blackframe_collect_directory_targets("${child_directory}" child_targets)
        list(APPEND collected_targets ${child_targets})
    endforeach()

    set("${output_variable}" "${collected_targets}" PARENT_SCOPE)
endfunction()

function(_blackframe_resolve_target candidate output_variable)
    set(resolved_candidate "${candidate}")

    if(resolved_candidate MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
        set(resolved_candidate "${CMAKE_MATCH_1}")
    elseif(resolved_candidate MATCHES "^\\$<BUILD_INTERFACE:([^>]+)>$")
        set(resolved_candidate "${CMAKE_MATCH_1}")
    endif()

    if(TARGET "${resolved_candidate}")
        get_target_property(aliased_target "${resolved_candidate}" ALIASED_TARGET)
        if(aliased_target)
            set(resolved_candidate "${aliased_target}")
        endif()
        set("${output_variable}" "${resolved_candidate}" PARENT_SCOPE)
    else()
        set("${output_variable}" "" PARENT_SCOPE)
    endif()
endfunction()

function(_blackframe_role_allows_dependency source_role dependency_role output_variable)
    if(source_role STREQUAL "core")
        set(allowed_roles core)
    elseif(source_role STREQUAL "cpu_embree")
        set(allowed_roles core cpu_embree dependency)
    elseif(source_role STREQUAL "cuda")
        set(allowed_roles core cuda dependency)
    elseif(source_role STREQUAL "dependency")
        set(allowed_roles dependency)
    elseif(source_role STREQUAL "reference")
        set(allowed_roles core reference)
    elseif(source_role STREQUAL "renderer_io")
        set(allowed_roles core dependency renderer_io)
    elseif(source_role STREQUAL "tool")
        set(allowed_roles core cpu_embree cuda dependency renderer_io tool)
    elseif(source_role STREQUAL "test")
        set(allowed_roles core cpu_embree cuda dependency reference renderer_io test)
    elseif(source_role STREQUAL "benchmark")
        set(allowed_roles benchmark core cpu_embree cuda dependency renderer_io)
    else()
        message(FATAL_ERROR "No dependency policy exists for target role '${source_role}'.")
    endif()

    if(dependency_role IN_LIST allowed_roles)
        set("${output_variable}" TRUE PARENT_SCOPE)
    else()
        set("${output_variable}" FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_blackframe_json_quote value output_variable)
    set(escaped_value "${value}")
    string(REPLACE "\\" "\\\\" escaped_value "${escaped_value}")
    string(REPLACE "\"" "\\\"" escaped_value "${escaped_value}")
    string(REPLACE "\n" "\\n" escaped_value "${escaped_value}")
    set("${output_variable}" "\"${escaped_value}\"" PARENT_SCOPE)
endfunction()

function(blackframe_validate_target_graph)
    cmake_parse_arguments(
        PARSE_ARGV
        0
        graph
        ""
        "JSON_OUTPUT;DOT_OUTPUT"
        ""
    )

    if(graph_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unexpected target graph arguments: ${graph_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT graph_JSON_OUTPUT OR NOT graph_DOT_OUTPUT)
        message(FATAL_ERROR "Target graph validation requires JSON_OUTPUT and DOT_OUTPUT.")
    endif()

    foreach(output_kind IN ITEMS JSON_OUTPUT DOT_OUTPUT)
        set(output_path "${graph_${output_kind}}")
        cmake_path(ABSOLUTE_PATH output_path NORMALIZE OUTPUT_VARIABLE absolute_output_path)
        cmake_path(
            IS_PREFIX
            PROJECT_BINARY_DIR
            "${absolute_output_path}"
            NORMALIZE
            output_is_in_build_tree
        )
        if(NOT output_is_in_build_tree)
            message(FATAL_ERROR "Target graph output must be below '${PROJECT_BINARY_DIR}'.")
        endif()
        set("graph_${output_kind}" "${absolute_output_path}")
    endforeach()

    _blackframe_collect_directory_targets("${PROJECT_SOURCE_DIR}" all_targets)
    list(REMOVE_DUPLICATES all_targets)

    set(blackframe_targets "")
    foreach(target IN LISTS all_targets)
        if(target MATCHES "^Blackframe")
            get_target_property(target_role "${target}" BLACKFRAME_TARGET_ROLE)
            if(NOT target_role)
                message(FATAL_ERROR "Blackframe target '${target}' has no declared graph role.")
            endif()
            list(APPEND blackframe_targets "${target}")
        endif()
    endforeach()
    list(SORT blackframe_targets)

    set(graph_edges "")
    foreach(target IN LISTS blackframe_targets)
        foreach(link_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(link_dependencies "${target}" "${link_property}")
            if(NOT link_dependencies)
                set(link_dependencies "")
            endif()

            foreach(link_dependency IN LISTS link_dependencies)
                _blackframe_resolve_target("${link_dependency}" resolved_dependency)
                if(resolved_dependency IN_LIST blackframe_targets)
                    list(APPEND graph_edges "${target}|${resolved_dependency}|link")
                endif()
            endforeach()
        endforeach()

        get_target_property(order_dependencies "${target}" MANUALLY_ADDED_DEPENDENCIES)
        if(NOT order_dependencies)
            set(order_dependencies "")
        endif()

        foreach(order_dependency IN LISTS order_dependencies)
            _blackframe_resolve_target("${order_dependency}" resolved_dependency)
            if(resolved_dependency IN_LIST blackframe_targets)
                list(APPEND graph_edges "${target}|${resolved_dependency}|order")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES graph_edges)
    list(SORT graph_edges)

    foreach(edge IN LISTS graph_edges)
        string(REPLACE "|" ";" edge_fields "${edge}")
        list(GET edge_fields 0 source_target)
        list(GET edge_fields 1 dependency_target)
        get_target_property(source_role "${source_target}" BLACKFRAME_TARGET_ROLE)
        get_target_property(dependency_role "${dependency_target}" BLACKFRAME_TARGET_ROLE)
        _blackframe_role_allows_dependency(
            "${source_role}"
            "${dependency_role}"
            dependency_is_allowed
        )
        if(NOT dependency_is_allowed)
            message(
                FATAL_ERROR
                "Target layer violation: '${source_target}' (${source_role}) depends on "
                "'${dependency_target}' (${dependency_role})."
            )
        endif()
    endforeach()

    set(remaining_targets ${blackframe_targets})
    while(remaining_targets)
        set(removable_target "")
        foreach(candidate IN LISTS remaining_targets)
            set(candidate_has_remaining_dependency FALSE)
            foreach(edge IN LISTS graph_edges)
                string(REPLACE "|" ";" edge_fields "${edge}")
                list(GET edge_fields 0 source_target)
                list(GET edge_fields 1 dependency_target)
                if(source_target STREQUAL candidate AND
                   dependency_target IN_LIST remaining_targets)
                    set(candidate_has_remaining_dependency TRUE)
                    break()
                endif()
            endforeach()

            if(NOT candidate_has_remaining_dependency)
                set(removable_target "${candidate}")
                break()
            endif()
        endforeach()

        if(NOT removable_target)
            list(JOIN remaining_targets ", " cyclic_targets)
            message(
                FATAL_ERROR
                "Blackframe target graph contains a dependency cycle among: ${cyclic_targets}"
            )
        endif()

        list(REMOVE_ITEM remaining_targets "${removable_target}")
    endwhile()

    set(json_nodes "")
    set(dot_contents "digraph BlackframeTargets {\n  rankdir=LR;\n")
    foreach(target IN LISTS blackframe_targets)
        get_target_property(target_role "${target}" BLACKFRAME_TARGET_ROLE)
        get_target_property(target_type "${target}" TYPE)
        _blackframe_json_quote("${target}" target_json)
        _blackframe_json_quote("${target_role}" role_json)
        _blackframe_json_quote("${target_type}" type_json)

        if(json_nodes)
            string(APPEND json_nodes ",\n")
        endif()
        string(
            APPEND
            json_nodes
            "    {\"name\": ${target_json}, \"role\": ${role_json}, \"type\": ${type_json}}"
        )
        string(APPEND dot_contents "  \"${target}\" [label=\"${target}\\n${target_role}\"];\n")
    endforeach()

    set(json_edges "")
    foreach(edge IN LISTS graph_edges)
        string(REPLACE "|" ";" edge_fields "${edge}")
        list(GET edge_fields 0 source_target)
        list(GET edge_fields 1 dependency_target)
        list(GET edge_fields 2 dependency_kind)
        _blackframe_json_quote("${source_target}" source_json)
        _blackframe_json_quote("${dependency_target}" dependency_json)
        _blackframe_json_quote("${dependency_kind}" kind_json)

        if(json_edges)
            string(APPEND json_edges ",\n")
        endif()
        string(
            APPEND
            json_edges
            "    {\"from\": ${source_json}, \"to\": ${dependency_json}, "
            "\"kind\": ${kind_json}}"
        )
        string(
            APPEND
            dot_contents
            "  \"${source_target}\" -> \"${dependency_target}\" "
            "[label=\"${dependency_kind}\"];\n"
        )
    endforeach()
    string(APPEND dot_contents "}\n")

    set(
        json_contents
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"acyclic\": true,\n"
        "  \"nodes\": [\n${json_nodes}\n  ],\n"
        "  \"edges\": [\n${json_edges}\n  ]\n"
        "}\n"
    )
    string(JOIN "" json_contents ${json_contents})

    file(WRITE "${graph_JSON_OUTPUT}" "${json_contents}")
    file(WRITE "${graph_DOT_OUTPUT}" "${dot_contents}")

    set(BLACKFRAME_TARGET_GRAPH_JSON "${graph_JSON_OUTPUT}" CACHE INTERNAL "" FORCE)
    set(BLACKFRAME_TARGET_GRAPH_DOT "${graph_DOT_OUTPUT}" CACHE INTERNAL "" FORCE)

    list(LENGTH blackframe_targets target_count)
    list(LENGTH graph_edges edge_count)
    message(STATUS "Blackframe target graph: ${target_count} targets, ${edge_count} edges, acyclic")
endfunction()
