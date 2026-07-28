#include <Blackframe/Extensions/ExtensionModule.hpp>
#include <gtest/gtest.h>

namespace blackframe::extensions {
namespace {

TEST(ExtensionModuleTest, RejectsRelativeLibraryPaths) {
    const auto module = ExtensionModule::load("BlackframeXpuReferenceCpu");

    ASSERT_FALSE(module.has_value());
    EXPECT_EQ(module.error().code, ExtensionLoadErrorCode::DynamicLibraryFailure);
}

TEST(ExtensionModuleTest, RejectsMissingAbsoluteLibraryPaths) {
    const auto path = std::filesystem::temp_directory_path() / "blackframe-missing-extension";
    const auto module = ExtensionModule::load(path);

    ASSERT_FALSE(module.has_value());
    EXPECT_EQ(module.error().code, ExtensionLoadErrorCode::DynamicLibraryFailure);
}

TEST(ExtensionModuleTest, RejectsLibrariesWithoutTheQuerySymbol) {
    const auto module =
        ExtensionModule::load(std::filesystem::path{BLACKFRAME_MISSING_QUERY_EXTENSION_PATH});

    ASSERT_FALSE(module.has_value());
    EXPECT_EQ(module.error().code, ExtensionLoadErrorCode::QuerySymbolMissing);
}

TEST(ExtensionModuleTest, RejectsAnIncompatibleExtensionApi) {
    const auto module =
        ExtensionModule::load(std::filesystem::path{BLACKFRAME_INCOMPATIBLE_EXTENSION_PATH});

    ASSERT_FALSE(module.has_value());
    EXPECT_EQ(module.error().code, ExtensionLoadErrorCode::MalformedExtensionApi);
}

TEST(ExtensionModuleTest, ShutsDownAnAcceptedExtensionBeforeUnloadingItsLibrary) {
    {
        auto module =
            ExtensionModule::load(std::filesystem::path{BLACKFRAME_LIFECYCLE_EXTENSION_PATH});

        ASSERT_TRUE(module.has_value()) << module.error().message;
        EXPECT_TRUE(module->is_active());
    }

    SUCCEED();
}

} // namespace
} // namespace blackframe::extensions
