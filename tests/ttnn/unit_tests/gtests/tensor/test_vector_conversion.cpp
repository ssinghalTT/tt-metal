// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include "ttnn/tensor/xtensor/xtensor_all_includes.hpp"

namespace ttnn {
namespace {

using ::testing::Eq;
using ::testing::Pointwise;

const std::vector<ttnn::SimpleShape>& get_shapes_for_test() {
    static auto* shapes = new std::vector<ttnn::SimpleShape>{
        ttnn::SimpleShape{1},
        ttnn::SimpleShape{1, 1, 1, 1},
        ttnn::SimpleShape{1, 1, 1, 10},
        ttnn::SimpleShape{1, 32, 32, 16},
        ttnn::SimpleShape{1, 40, 3, 128},
        ttnn::SimpleShape{2, 2},
        ttnn::SimpleShape{1, 1, 1, 1, 10},
    };
    return *shapes;
}

TensorSpec get_tensor_spec(const ttnn::SimpleShape& shape, DataType dtype, Layout layout = Layout::ROW_MAJOR) {
    return TensorSpec(shape, TensorLayout(dtype, layout, MemoryConfig{}));
}

template <typename T>
std::vector<T> arange(int64_t start, int64_t end, int64_t step) {
    std::vector<T> result;
    for (int el : xt::arange<int64_t>(start, end, step)) {
        if constexpr (std::is_same_v<T, ::bfloat16>) {
            result.push_back(T(static_cast<float>(el)));
        } else {
            result.push_back(static_cast<T>(el));
        }
    }
    return result;
}

template <typename T>
class VectorConversionTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, bfloat16, uint32_t, int32_t>;
TYPED_TEST_SUITE(VectorConversionTest, TestTypes);

TYPED_TEST(VectorConversionTest, Roundtrip) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input = arange<TypeParam>(0, static_cast<int64_t>(shape.volume()), 1);
        auto output = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>()))
                          .template to_vector<TypeParam>();
        EXPECT_THAT(output, Pointwise(Eq(), input)) << "for shape: " << shape;
    }
}

TYPED_TEST(VectorConversionTest, InvalidSize) {
    ttnn::SimpleShape shape{32, 32};
    auto input = arange<TypeParam>(0, 42, 1);

    ASSERT_NE(input.size(), shape.volume());
    EXPECT_ANY_THROW(Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>())));
}

TYPED_TEST(VectorConversionTest, RoundtripTilezedLayout) {
    ttnn::SimpleShape shape{128, 128};

    auto input = arange<TypeParam>(0, shape.volume(), 1);
    // TODO: Support this.
    EXPECT_ANY_THROW(
        Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>(), Layout::TILE)));

    auto output = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>()))
                      .to(Layout::TILE)
                      .template to_vector<TypeParam>();
    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TYPED_TEST(VectorConversionTest, InvalidDtype) {
    ttnn::SimpleShape shape{32, 32};
    auto input = arange<TypeParam>(0, 42, 1);

    ASSERT_NE(input.size(), shape.volume());
    EXPECT_ANY_THROW(Tensor::from_vector(
        input,
        get_tensor_spec(
            shape,
            // Use INT32 for verification, except for when the actual type is int32_t.
            (std::is_same_v<TypeParam, int32_t> ? DataType::FLOAT32 : DataType::INT32))));
}

TEST(FloatVectorConversionTest, RoundtripBfloat16Representation) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input_bf16 = arange<bfloat16>(0, static_cast<int64_t>(shape.volume()), 1);
        std::vector<float> input_ft;
        input_ft.reserve(input_bf16.size());
        std::transform(input_bf16.begin(), input_bf16.end(), std::back_inserter(input_ft), [](bfloat16 bf) {
            return bf.to_float();
        });

        auto output_bf16 =
            Tensor::from_vector(input_ft, get_tensor_spec(shape, DataType::BFLOAT16)).to_vector<bfloat16>();
        EXPECT_THAT(output_bf16, Pointwise(Eq(), input_bf16)) << "for shape: " << shape;

        auto output_ft = Tensor::from_vector(input_bf16, get_tensor_spec(shape, DataType::BFLOAT16)).to_vector<float>();
        EXPECT_THAT(output_ft, Pointwise(Eq(), input_ft)) << "for shape: " << shape;
    }
}

}  // namespace
}  // namespace ttnn
