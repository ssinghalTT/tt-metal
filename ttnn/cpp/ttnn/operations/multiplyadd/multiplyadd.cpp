#include "multiplyadd.hpp"
#include "ttnn/operations/multiplyadd/device/multiplyadd_device_operation.hpp"
using namespace tt::tt_metal;
using namespace ttnn;

namespace ttnn::operations::multiplyadd {

ttnn::Tensor MulAddOperation::invoke(
    const ttnn::Tensor& input_tensor1, const ttnn::Tensor& input_tensor2, const ttnn::Tensor& input_tensor3) {
    return ttnn::prim::multiplyadd(input_tensor1, input_tensor2, input_tensor3);
}
}  // namespace ttnn::operations::multiplyadd
