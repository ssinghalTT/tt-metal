## This commit contains Conv1d unit test for SqueezeBert model.

1. `test_squeezebert_conv1d` method in `tests/ttnn/unit_tests/operations/test_conv1d.py` file will contain unit test for ttnn.conv2d.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_conv1d.py::test_squeezebert_conv1d`

## Expected Behaviour / Error(s):

    Expected to pass the testcase with bias.

## Details:

#### On WH(n300):


1. Input shape: [8, 384, 3072] in (batch_size, input_length, in_channels) format and out_channels: 768, groups: 4, bias = True
    ```
    E       RuntimeError: TT_THROW @ ../tt_metal/impl/program/program.cpp:511: tt::exception
    E       info:
    E       Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=5)] grow to 3400864 B which is beyond max L1 size of 1499136 B
    ```
2. Input shape: [8, 384, 768] in (batch_size, input_length, in_channels) format and out_channels: 3072, groups: 4, bias = True
    ```
    E       RuntimeError: TT_THROW @ ../tt_metal/impl/program/program.cpp:511: tt::exception
    E       info:
    E       Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=5)] grow to 2811040 B which is beyond max L1 size of 1499136 B

    ```
