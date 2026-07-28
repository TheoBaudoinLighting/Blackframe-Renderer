include(GoogleTest)

add_subdirectory(Core)
add_subdirectory(Build)
add_subdirectory(Renderer)
add_subdirectory(Extensions)
add_subdirectory(XPU)
add_subdirectory(IPC)
add_subdirectory(Engine)

if(BLACKFRAME_ENABLE_CUDA)
    add_subdirectory(CUDA)
endif()
