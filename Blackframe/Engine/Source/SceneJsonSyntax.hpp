#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/SceneDescriptionJson.hpp>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine::scene_json_syntax {

// Syntax limits are shared by decoding and canonical encoding. Every field must be non-zero and
// no field may exceed MaximumSceneDescriptionJsonLimits.
[[nodiscard]] core::Status validate_limits(SceneDescriptionJsonLimits limits);

class Reader final {
  public:
    [[nodiscard]] static core::Result<Reader> create(std::string_view encoded_scene,
                                                     SceneDescriptionJsonLimits limits = {});

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    Reader(Reader&&) noexcept = default;
    Reader& operator=(Reader&&) noexcept = default;
    ~Reader() noexcept = default;

    // The callback receives one decoded key and must consume exactly one value from this reader.
    // Unknown-key policy belongs to the schema callback. Duplicate decoded keys are rejected here.
    template <typename Callback> [[nodiscard]] core::Status read_object(Callback&& callback) {
        auto callable = std::ref(callback);
        using CallbackType = decltype(callable);
        return read_object_impl(&callable,
                                [](void* opaque, const std::string_view key) -> core::Status {
                                    return std::invoke(*static_cast<CallbackType*>(opaque), key);
                                });
    }

    // The callback receives the zero-based element index and must consume exactly one value.
    template <typename Callback> [[nodiscard]] core::Status read_array(Callback&& callback) {
        auto callable = std::ref(callback);
        using CallbackType = decltype(callable);
        return read_array_impl(&callable,
                               [](void* opaque, const std::uint64_t index) -> core::Status {
                                   return std::invoke(*static_cast<CallbackType*>(opaque), index);
                               });
    }

    [[nodiscard]] core::Result<std::string> read_string();
    [[nodiscard]] core::Result<std::uint64_t> read_u64();
    [[nodiscard]] core::Result<std::int64_t> read_i64();
    [[nodiscard]] core::Result<float> read_float();
    [[nodiscard]] core::Status read_null();
    [[nodiscard]] bool peek_null() noexcept;
    [[nodiscard]] bool peek_array() noexcept;

    // Succeeds only after exactly one complete root value followed by JSON whitespace.
    [[nodiscard]] core::Status finish();

  private:
    using ObjectCallback = core::Status (*)(void*, std::string_view);
    using ArrayCallback = core::Status (*)(void*, std::uint64_t);

    Reader(std::string_view encoded_scene, SceneDescriptionJsonLimits limits,
           std::vector<std::uint8_t> value_slots) noexcept;

    [[nodiscard]] core::Status read_object_impl(void* context, ObjectCallback callback);
    [[nodiscard]] core::Status read_array_impl(void* context, ArrayCallback callback);
    [[nodiscard]] core::Result<std::string> read_string_token();
    [[nodiscard]] core::Result<std::string_view> read_number_token(bool integer_only,
                                                                   bool unsigned_only);
    [[nodiscard]] core::Status claim_value();
    [[nodiscard]] core::Status begin_container(char opening);
    void end_container() noexcept;
    void skip_whitespace() noexcept;
    [[nodiscard]] bool consume(char expected) noexcept;
    [[nodiscard]] bool consume_literal(std::string_view literal) noexcept;
    [[nodiscard]] bool value_boundary() const noexcept;
    [[nodiscard]] core::Error syntax_error(std::string message) const;
    [[nodiscard]] core::Error resource_error(std::string message) const;

    std::string_view input_;
    SceneDescriptionJsonLimits limits_{};
    std::size_t offset_{};
    std::uint64_t value_count_{};
    std::uint64_t string_byte_count_{};
    std::uint32_t depth_{};
    std::vector<std::uint8_t> value_slots_{};
    bool failed_{};
    bool finished_{};
};

class Writer final {
  public:
    [[nodiscard]] static core::Result<Writer> create(SceneDescriptionJsonLimits limits = {});

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    Writer(Writer&&) noexcept = default;
    Writer& operator=(Writer&&) noexcept = default;
    ~Writer() noexcept = default;

    [[nodiscard]] core::Status begin_object();
    [[nodiscard]] core::Status end_object();
    [[nodiscard]] core::Status begin_array();
    [[nodiscard]] core::Status end_array();
    [[nodiscard]] core::Status key(std::string_view value);
    [[nodiscard]] core::Status write_null();
    [[nodiscard]] core::Status write_string(std::string_view value);
    [[nodiscard]] core::Status write_u64(std::uint64_t value);
    [[nodiscard]] core::Status write_i64(std::int64_t value);
    [[nodiscard]] core::Status write_float(float value);

    // Returns the unique compact UTF-8 representation after one complete root value.
    [[nodiscard]] core::Result<std::string> finish();

  private:
    enum class ContainerKind : std::uint8_t {
        object,
        array,
    };

    struct Frame final {
        ContainerKind kind{};
        std::uint64_t element_count{};
        bool awaiting_value{};
        std::set<std::string, std::less<>> keys{};
    };

    explicit Writer(SceneDescriptionJsonLimits limits) noexcept;

    [[nodiscard]] core::Status begin_value();
    [[nodiscard]] core::Status begin_container(ContainerKind kind, char opening);
    [[nodiscard]] core::Status end_container(ContainerKind kind, char closing);
    [[nodiscard]] core::Status append(std::string_view bytes);
    [[nodiscard]] core::Status append_character(char value);
    [[nodiscard]] core::Status append_escaped_string(std::string_view value,
                                                     bool count_string_bytes);
    [[nodiscard]] core::Status count_value();
    [[nodiscard]] core::Status count_string(std::size_t byte_count);
    [[nodiscard]] core::Error state_error(std::string message) const;
    [[nodiscard]] core::Error resource_error(std::string message) const;

    SceneDescriptionJsonLimits limits_{};
    std::string output_{};
    std::vector<Frame> stack_{};
    std::uint64_t value_count_{};
    std::uint64_t string_byte_count_{};
    bool root_written_{};
    bool failed_{};
    bool finished_{};
};

} // namespace blackframe::engine::scene_json_syntax
