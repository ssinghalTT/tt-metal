
#pragma once

#include "ttnn/decorators.hpp"
namespace ttnn {

namespace operations {
namespace multiplyadd {

struct MulAddOperation {
    static Tensor invoke(
        const ttnn::Tensor& input_tensor1, const ttnn::Tensor& input_tensor2, const ttnn::Tensor& input_tensor3);
};

}  // namespace multiplyadd
}  // namespace operations
constexpr auto multiplyadd =
    ttnn::register_operation<"ttnn::multiplyadd", ttnn::operations::multiplyadd::MulAddOperation>();
}  // namespace ttnn
