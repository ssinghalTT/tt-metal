## This commit contains Conv1d unit test for Whisper model.

1. `test_whisper_conv1d` method in `tests/ttnn/unit_tests/operations/test_conv1d.py` file will contain unit test for ttnn.conv2d.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_conv1d.py::test_whisper_conv1d`

## Expected Behaviour / Error(s):

    Expected to pass the testcase with bias.

## Details:

#### On WH(n300):


1. Input shape: [8, 3000, 512] in (batch_size, input_length, in_channels) format and out_channels: 512, groups: 1, stride: 2, bias = True
    ```
    E       RuntimeError: TT_THROW @ ../tt_metal/impl/allocator/allocator.cpp:143: tt::exception
    E       info:
    E       Out of Memory: Not enough space to allocate 24652800 B L1 buffer across 25 banks, where each bank needs to store 986112 B
    ```
