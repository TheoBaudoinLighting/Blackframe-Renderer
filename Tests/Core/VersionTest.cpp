#include <Blackframe/Core/Version.hpp>
#include <gtest/gtest.h>

namespace blackframe::core {
namespace {

TEST(VersionTest, UsesBlackframeAsTheOnlyProductName) {
    EXPECT_EQ(product_name(), "Blackframe");
}

TEST(VersionTest, ExposesTheConfiguredSemanticVersion) {
    const auto version = current_version();
    EXPECT_EQ(version.major, 0U);
    EXPECT_EQ(version.minor, 1U);
    EXPECT_EQ(version.patch, 0U);
    EXPECT_EQ(version_string(), "0.1.0");
}

} // namespace
} // namespace blackframe::core
