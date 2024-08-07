There are 7 conv's in model_k model.

**model_k model: 256x256**
- To test the unit_test, run `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_256x256_failing_convs`
- Among 7 convs, 4 convs passed. Among 3 failing convs, 2 fail with L1 issues and 1 fail with low PCC.

**model_k model: 128x128**
- The command to test the unit test: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_128x128_failing_convs`
- Among 7 convs, 6 convs passed. 1 convs fail with low PCC.

Note: For conv checking purpose batch_size=1 is used even though 32 is used in the model.
