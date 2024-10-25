On unit testing the input configurations of MaxPool2d op in PETR VovNetCP submodule,
- Among 3 input configurations, 2 passed. Passed after commenting the skip conditions from run_max_pool. Commented the following lines:
```
if (
        (kernel_h == 13 and pad_h != 6)
        or (kernel_h == 9 and pad_h != 4)
        or (kernel_h == 5 and pad_h != 2)
        or (kernel_h == 3 and pad_h != 1)
        or (kernel_h == 2 and pad_h != 0)
    ):
        pytest.skip("kernel size and padding combination not supported")
```
- A input configuration got failed in Bfloat16 with the error:
```
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/pool/maxpool/device/max_pool2d_device_op.cpp:26: is_pow2
E       info:
E       Row size (nchannels * bytes = 1536) should be power of 2 (false).
```
The same configuration got skipped in dtype Bfloat8_b with the information: `For BFP8_B datatype, input height * width should be multiple of 32`.

Run the command to unit test the failing case: `pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_petr_vovnetcp`
