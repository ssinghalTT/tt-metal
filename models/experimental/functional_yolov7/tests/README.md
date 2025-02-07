## Trace+2cq implementation fails with Out of Memoy issue in DRAM in Conv34

- The inputs to the Yolov7 model pipeline are in DRAM, as the second Conv gives Circular buffers issue while input is passed in L1 memory to the model. To note, the output of all conv ops are in L1 sharded memory config.
- As the input to Yolov7 trace implementation can't be kept in l1 sharded, having the input in DRAM memory config.
- Currently, facing the following OOM issue from Conv34:

```
Out of Memory: Not enough space to allocate 6400 B L1_SMALL buffer across 25 banks, where each bank needs to store 256 B
```

Note: The trace implementation is referred from Yolov4 model.


### Run the following test to reproduce the issue

```
pytest models/experimental/functional_yolov7/tests/test_yolov7_performant.py::test_run_yolov7_inference
```
