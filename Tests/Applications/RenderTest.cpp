#include <Blackframe/Core/Version.hpp>
#include <gtest/gtest.h>

namespace blackframe::application {
namespace {

TEST(RenderTest, UsesTheBlackframeCoreIdentity) {
    EXPECT_EQ(core::product_name(), "Blackframe");
    EXPECT_FALSE(core::version_string().empty());
}

} // namespace
} // namespace blackframe::application
