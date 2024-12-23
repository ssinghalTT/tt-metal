
#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/multiplyadd/device/multiplyadd_device_operation.hpp"
namespace ttnn {
namespace operations {
namespace multiplyadd {

struct MulAddOperation {
    static Tensor invoke(
        const ttnn::Tensor& input_tensor1, const ttnn::Tensor& input_tensor2, const ttnn::Tensor& input_tensor3) {
        return ttnn::prim::multiplyadd(input_tensor1, input_tensor2, input_tensor3);
    }
};

}  // namespace multiplyadd
}  // namespace operations
constexpr auto multiplyadd = register_operation<"ttnn::multiplyadd", ttnn::operations::multiplyadd::MulAddOperation>();
}  // namespace ttnn
