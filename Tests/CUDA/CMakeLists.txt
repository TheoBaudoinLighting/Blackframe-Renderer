add_executable(
    BlackframeCudaSmokeTests
    CudaSmokeTest.cpp
    TransportAbiTest.cpp
)
blackframe_set_target_role(BlackframeCudaSmokeTests test)

target_link_libraries(
    BlackframeCudaSmokeTests
    PRIVATE
        Blackframe::Cuda
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaSmokeTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;smoke;cuda"
)

add_executable(
    BlackframeCudaTransportLobeTests
    TransportLobeTest.cpp
)
blackframe_set_target_role(BlackframeCudaTransportLobeTests test)

target_link_libraries(
    BlackframeCudaTransportLobeTests
    PRIVATE
        Blackframe::CudaSmokeKernel
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaTransportLobeTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
)

add_executable(
    BlackframeCudaMemoryTests
    DeviceMemoryTest.cpp
)
blackframe_set_target_role(BlackframeCudaMemoryTests test)

target_link_libraries(
    BlackframeCudaMemoryTests
    PRIVATE
        Blackframe::CudaMemory
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaMemoryTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
)

add_executable(
    BlackframeCudaAsyncRuntimeTests
    AsyncRuntimeTest.cpp
)
blackframe_set_target_role(BlackframeCudaAsyncRuntimeTests test)

target_link_libraries(
    BlackframeCudaAsyncRuntimeTests
    PRIVATE
        Blackframe::CudaAsyncRuntime
        Blackframe::CudaMemory
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaAsyncRuntimeTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
)

add_executable(
    BlackframeCudaSampleStreamTests
    SampleStreamTest.cpp
)
blackframe_set_target_role(BlackframeCudaSampleStreamTests test)

target_link_libraries(
    BlackframeCudaSampleStreamTests
    PRIVATE
        Blackframe::CudaMemory
        Blackframe::CudaSampleStreamKernel
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        Blackframe::Renderer
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaSampleStreamTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
)

add_subdirectory(Memcheck)
