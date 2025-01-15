# YOLOv8n Ops Unit Tests

This commit contains unit tests for the ops in the YOLOv8n model. The tests include test for layers such as `Conv2d`, `MaxPool2d`,`Silu` .

### To run the maxpool unit test, use the following command:

```
pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_run_max_pool_yolov8n
```
### To run the Conv2d unit test, use the following command:

```
pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_conv_yolov8n
```
### To run the Silu unit test, use the following command:

```
pytest tests/ttnn/unit_tests/operations/test_silu.py::test_silu_yolov8n
```
