import torch
import ttnn
from models.utility_functions import torch_random, comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_maxpool(device, input_shape, kernel_size, stride, padding, dilation):
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    batch_size, in_c, in_h, in_w = input_shape

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, in_c))
    input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.max_pool2d(
        input_tensor,
        batch_size,
        in_h,
        in_w,
        in_c,
        kernel_size,
        stride,
        padding,
        dilation,
    )

    expected_output = torch.nn.functional.max_pool2d(torch_input, kernel_size, stride, padding)

    output_tensor = ttnn.to_torch(output_tensor)
    _, out_c, out_h, out_w = expected_output.shape
    output_tensor = torch.reshape(output_tensor, (batch_size, out_h, out_w, out_c))
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    print(output_tensor.shape)
    # COMMENTED OUT FOR INIT DEBUGGING
    assert_with_pcc(output_tensor, expected_output)


def test_add(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    # torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_input_tensor = torch.linspace(0, 1023, steps=batch_size * h * w, dtype=torch.bfloat16).reshape(
        batch_size, h, w
    )
    torch_zero_tensor = torch.empty((batch_size, h, w), dtype=torch.bfloat16).uniform_(-1, 1)
    torch_zero_tensor = torch.zeros_like(torch_input_tensor, dtype=torch.bfloat16)
    # torch output
    torch_output_tensor = torch_input_tensor + torch_zero_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    zero_tensor = ttnn.from_torch(torch_zero_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    # ttnn output
    output_tensor = ttnn.add(input_tensor, zero_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


def test_mean(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    # torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_input_tensor = torch.linspace(0, 1023, steps=batch_size * h * w, dtype=torch.bfloat16).reshape(
        batch_size, h, w
    )
    # torch output
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=True, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    # ttnn output
    output_tensor = ttnn.mean(input_tensor, dim=dim)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


if __name__ == "__main__":
    try:
        device = ttnn.open_device(device_id=0, l1_small_size=4096)
        test_maxpool(device, (1, 32, 2, 2), (2, 2), (2, 2), (0, 0), (1, 1))
        # test_add(device, 1, 32, 32, -2)
        test_mean(device, 1, 32, 32, -2)
    finally:
        ttnn.close_device(device)
