include(GoogleTest)

add_subdirectory(Applications)
add_subdirectory(Core)
add_subdirectory(Build)
add_subdirectory(Renderer)
add_subdirectory(Extensions)
add_subdirectory(XPU)
add_subdirectory(IPC)
add_subdirectory(Engine)

if(BLACKFRAME_ENABLE_EMBREE)
    add_subdirectory(Backends/CPU/Embree)
endif()

if(BLACKFRAME_ENABLE_CUDA)
    add_subdirectory(CUDA)
endif()

set(
    blackframe_test_targets
    BlackframeCoreTests
    BlackframeEngineTests
    BlackframeExtensionAbiCTest
    BlackframeExtensionTests
    BlackframeHostLanguageTests
    BlackframeIpcTests
    BlackframeRenderTests
    BlackframeCornellReference64Tests
    BlackframeCornellReference256Tests
    BlackframeRendererTests
    BlackframeXpuTests
)
if(BLACKFRAME_ENABLE_EMBREE)
    list(APPEND blackframe_test_targets BlackframeEmbreeSmokeTests)
endif()
if(BLACKFRAME_ENABLE_CUDA)
    list(APPEND blackframe_test_targets BlackframeCudaSmokeTests)
endif()

add_custom_target(BlackframeTests)
add_dependencies(BlackframeTests ${blackframe_test_targets})
blackframe_set_target_role(BlackframeTests test)
