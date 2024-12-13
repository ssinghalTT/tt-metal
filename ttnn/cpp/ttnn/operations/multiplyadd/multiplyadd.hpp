
#pragma once

#include "ttnn/decorators.hpp"
namespace ttnn {

namespace operations {
namespace mac {

struct MulAddOperation {
    static Tensor invoke(
        const ttnn::Tensor& input_tensor1, const ttnn::Tensor& input_tensor2, const ttnn::Tensor& input_tensor3);
};

}  // namespace mac
}  // namespace operations

}  // namespace ttnn

namespace ttnn {
constexpr auto multiplyadd =
    ttnn::register_operation_with_auto_launch_op<"ttnn::multiplyadd", ttnn::operations::mac::MulAddOperation>();
}  // namespace ttnn
