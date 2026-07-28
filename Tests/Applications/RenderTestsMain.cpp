#include <Blackframe/Core/Version.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <string_view>

namespace {

constexpr auto usage = std::string_view{
    "Blackframe test runner\n"
    "Usage:\n"
    "  render_tests [GoogleTest options]\n"
    "  render_tests --version\n"
    "  render_tests --help\n",
};

} // namespace

int main(int argument_count, char** arguments) {
    const auto argument = argument_count > 1 ? std::string_view{arguments[1]} : std::string_view{};
    if (argument_count == 2 && (argument == "--help" || argument == "-h")) {
        std::cout << usage;
        return 0;
    }
    if (argument_count == 2 && argument == "--version") {
        std::cout << blackframe::core::product_name() << ' ' << blackframe::core::version_string()
                  << '\n';
        return 0;
    }

    testing::InitGoogleTest(&argument_count, arguments);
    if (argument_count > 1) {
        std::cerr << "Blackframe: unknown render_tests option '" << arguments[1] << "'\n" << usage;
        return 2;
    }
    return RUN_ALL_TESTS();
}
