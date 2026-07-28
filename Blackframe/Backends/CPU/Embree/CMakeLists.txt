add_library(BlackframeCpuEmbree INTERFACE)
add_library(Blackframe::CpuEmbree ALIAS BlackframeCpuEmbree)

target_link_libraries(
    BlackframeCpuEmbree
    INTERFACE
        Blackframe::Embree
        Blackframe::ProjectOptions
        Blackframe::Renderer
)

blackframe_set_target_role(BlackframeCpuEmbree cpu_embree)
