#include <Blackframe/Core/Status.hpp>
#include <gtest/gtest.h>

namespace blackframe::core {
namespace {

TEST(StatusTest, NamesEveryDeclaredCode) {
    EXPECT_EQ(status_code_name(StatusCode::success), "success");
    EXPECT_EQ(status_code_name(StatusCode::invalid_argument), "invalid_argument");
    EXPECT_EQ(status_code_name(StatusCode::unavailable), "unavailable");
    EXPECT_EQ(status_code_name(StatusCode::not_found), "not_found");
    EXPECT_EQ(status_code_name(StatusCode::incompatible), "incompatible");
    EXPECT_EQ(status_code_name(StatusCode::resource_exhausted), "resource_exhausted");
    EXPECT_EQ(status_code_name(StatusCode::protocol_error), "protocol_error");
    EXPECT_EQ(status_code_name(StatusCode::platform_error), "platform_error");
    EXPECT_EQ(status_code_name(StatusCode::internal_error), "internal_error");
}

} // namespace
} // namespace blackframe::core
