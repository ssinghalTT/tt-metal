// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <span>

#include "small_vector.hpp"

namespace tt::tt_metal {

// Container wrapper that allows negative indexing
class ShapeBase {
public:
    using Container = SmallVector<uint32_t>;

    ShapeBase() { init(); };
    explicit ShapeBase(const Container& shape) : value_(shape) { init(); }
    explicit ShapeBase(Container&& shape) : value_(std::move(shape)) { init(); }
    explicit ShapeBase(std::initializer_list<uint32_t> ilist) : value_(ilist) { init(); }
    template <std::size_t N>
    explicit ShapeBase(const std::array<uint32_t, N>& arr) : value_(arr.begin(), arr.end()) {
        init();
    }
    explicit ShapeBase(std::span<const uint32_t> span) : value_(span.begin(), span.end()) { init(); }

    ShapeBase(const ShapeBase& other) = default;
    ShapeBase& operator=(const ShapeBase& other) = default;

    ShapeBase(ShapeBase&& other) noexcept {
        other.moved_from = false;
        this->value_ = other.value_;
        this->original_size_ = other.original_size_;
    }
    ShapeBase& operator=(ShapeBase&& other) noexcept {
        this->moved_from = other.moved_from;
        other.moved_from = false;
        this->value_ = other.value_;
        this->original_size_ = other.original_size_;
        return *this;
    }

    template <std::size_t N>
    bool operator==(const std::array<uint32_t, N>& other) const {
        bool same_size = value_.size() == N;
        return same_size && std::equal(value_.begin(), value_.end(), other.begin());
    }

    bool operator==(const ShapeBase& other) const;
    bool operator==(const Container& other) const;
    bool operator==(const std::vector<uint32_t>& other) const;

    uint32_t operator[](int32_t index) const;
    uint32_t& operator[](int32_t index);

    Container::const_iterator cbegin() const;
    Container::const_iterator cend() const;

    std::span<const uint32_t> view() const;

    bool empty() const;

protected:
    void init();
    size_t size() const;

    Container value_;

private:
    bool moved_from = false;
    size_t original_size_ = 0;
};

}  // namespace tt::tt_metal
