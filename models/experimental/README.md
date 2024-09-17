On unit testing Conv2d op of Yolov7 model, 2 convs failed with the error:
```
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:962: a_shape[-1] == b_shape[-2]
E       info:
E       The width of the first tensor must be equal to the height of the second tensor. Mismatch: width={} height={}
```

Run the command to test: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_yolov7_failing_convs`
