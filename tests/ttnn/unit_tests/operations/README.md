# YOLOv8m Ops Unit Tests

This commit contains unit tests for the ops in the YOLOv8n model. The tests include test for layers such as `Conv2d`, `MaxPool2d`,`Silu` .


### To run the Conv2d unit test, use the following command:

```
pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_conv_yolov8m
```
### To run the Silu unit test, use the following command:

```
pytest tests/ttnn/unit_tests/operations/test_silu.py::test_silu_yolov8m
```
