add_library(
    BlackframeCpuEmbree
    STATIC
        Include/Blackframe/Backends/CPU/Embree/AccelBackend.hpp
        Source/AccelBackend.cpp
)
add_library(Blackframe::CpuEmbree ALIAS BlackframeCpuEmbree)
blackframe_set_target_role(BlackframeCpuEmbree cpu_embree)

target_include_directories(
    BlackframeCpuEmbree
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    BlackframeCpuEmbree
    PUBLIC
        Blackframe::SceneGeometry
    PRIVATE
        Blackframe::Embree
        Blackframe::ProjectOptions
)

set_target_properties(
    BlackframeCpuEmbree
    PROPERTIES
        OUTPUT_NAME BlackframeCpuEmbree
        DEBUG_POSTFIX d
)
