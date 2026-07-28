#include <Blackframe/Core/Version.hpp>

#include <iostream>
#include <string_view>

namespace {

constexpr auto usage = std::string_view{
    "Blackframe test runner\n"
    "Usage:\n"
    "  render_tests --version\n"
    "  render_tests --help\n",
};

} // namespace

int main(const int argument_count, const char* const* const arguments) {
    if (argument_count <= 1) {
        std::cout << usage;
        return 0;
    }

    const auto argument = std::string_view{arguments[1]};
    if (argument_count == 2 && (argument == "--help" || argument == "-h")) {
        std::cout << usage;
        return 0;
    }
    if (argument_count == 2 && argument == "--version") {
        std::cout << blackframe::core::product_name() << ' '
                  << blackframe::core::version_string() << '\n';
        return 0;
    }

    std::cerr << "Blackframe: unknown render_tests option\n" << usage;
    return 2;
}
