#include <cstddef>
#include <gtest/gtest.h>

namespace blackframe::build {
namespace {

static_assert(__cplusplus > 202302L, "Blackframe host targets require C++26 language mode.");

template <std::size_t index, typename... Values>
consteval auto select_pack_value(Values... values) {
    return values...[index];
}

static_assert(select_pack_value<2>(11, 22, 33, 44) == 33);

TEST(Cxx26LanguageFeatureTest, CompilesPackIndexingInTheRequiredLanguageMode) {
    constexpr auto selected_value = select_pack_value<3>(10, 20, 30, 40, 50);

    EXPECT_EQ(selected_value, 40);
}

} // namespace
} // namespace blackframe::build
