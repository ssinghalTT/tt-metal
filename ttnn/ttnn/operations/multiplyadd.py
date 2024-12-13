import ttnn


def golden_function(input_tensor_a, input_tensor_b, input_tensor_c, *args, **kwargs):
    import torch

    return torch.add(torch.mul(input_tensor_a, input_tensor_b), input_tensor_c)


ttnn.attach_golden_function(ttnn.multiplyadd, golden_function=golden_function)
