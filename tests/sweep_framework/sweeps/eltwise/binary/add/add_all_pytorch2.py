# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "test_bcast_1d_13": {
        "input_shape": [
            # {"self": [0, 1], "other": [0, 1]}, #0 is not a valid shape
            # {"self": [0], "other": [0]}, #0 is not a valid shape
            {"self": [1, 1, 1024], "other": [1, 1, 1024]},
            {"self": [1, 1, 16, 32], "other": [1, 1, 16, 32]},
            {"self": [1, 1, 3072], "other": [1, 1, 3072]},
            {"self": [1, 1, 4096], "other": [1, 1, 4096]},
            {"self": [1, 1, 512], "other": [1, 1, 512]},
            {"self": [1, 1, 7, 64], "other": [1, 1, 7, 64]},
            {"self": [1, 1, 768], "other": [1, 1, 768]},
            {"self": [1, 1, 768], "other": [1, 768]},
            {"self": [1, 10, 1024], "other": [1, 10, 1024]},
            {"self": [1, 10, 512], "other": [1, 10, 512]},
            {"self": [1, 10, 768], "other": [1, 10, 768]},
            {"self": [1, 1008, 7, 7], "other": [1, 1008, 7, 7]},
            {"self": [1, 1024, 10, 10], "other": [1, 1024, 10, 10]},
            {"self": [1, 1024, 14, 14], "other": [1, 1024, 14, 14]},
            {"self": [1, 1024, 16, 16], "other": [1, 1024, 16, 16]},
            {"self": [1, 1024, 160], "other": [1, 1024, 160]},
            {"self": [1, 1024, 256], "other": [256]},
            {"self": [1, 1024, 45, 80], "other": [1, 1024, 1, 1]},
            {"self": [1, 1024, 45, 80], "other": [1, 1024, 45, 80]},
            {"self": [1, 1024, 50, 68], "other": [1, 1024, 1, 1]},
            {"self": [1, 1024, 50, 68], "other": [1, 1024, 50, 68]},
            {"self": [1, 1024, 640], "other": [1, 1024, 640]},
            {"self": [1, 1024, 7, 7], "other": [1, 1024, 7, 7]},
            {"self": [1, 1024], "other": [1, 1024]},
            {"self": [1, 104, 28, 28], "other": [1, 104, 28, 28]},
            {"self": [1, 1056, 48, 48], "other": [1, 1056, 48, 48]},
            {"self": [1, 112, 14, 14], "other": [1, 112, 14, 14]},
            {"self": [1, 112, 15, 15], "other": [1, 112, 15, 15]},
            {"self": [1, 112, 20, 20], "other": [1, 112, 20, 20]},
            {"self": [1, 112, 24, 24], "other": [1, 112, 24, 24]},
            {"self": [1, 12, 1, 10], "other": [1, 1, 1, 10]},
            {"self": [1, 12, 1, 10], "other": [1, 12, 1, 10]},
            {"self": [1, 12, 1, 1], "other": [1, 1, 1, 1]},
            {"self": [1, 12, 1, 1], "other": [1, 12, 1, 1]},
            {"self": [1, 12, 1, 24], "other": [1, 1, 1, 24]},
            {"self": [1, 12, 1, 2], "other": [1, 1, 1, 2]},
            {"self": [1, 12, 1, 2], "other": [1, 12, 1, 2]},
            {"self": [1, 12, 1, 46], "other": [1, 1, 1, 46]},
            {"self": [1, 12, 10, 10], "other": [1, 1, 1, 10]},
            {"self": [1, 12, 10, 10], "other": [1, 12, 10, 10]},
            {"self": [1, 12, 12, 12], "other": [1, 1, 1, 12]},
            {"self": [1, 12, 128], "other": [1, 12, 128]},
            {"self": [1, 12, 14, 14], "other": [1, 1, 1, 14]},
            {"self": [1, 12, 197, 197], "other": [1, 12, 197, 197]},
            {"self": [1, 12, 201, 201], "other": [1, 1, 1, 201]},
            {"self": [1, 12, 24, 24], "other": [1, 1, 24, 24]},
            {"self": [1, 12, 25, 25], "other": [1, 1, 1, 25]},
            {"self": [1, 12, 3072], "other": [1, 12, 3072]},
            {"self": [1, 12, 45, 45], "other": [1, 1, 45, 45]},
            {"self": [1, 12, 7, 7], "other": [1, 1, 1, 7]},
            {"self": [1, 12, 768], "other": [1, 12, 768]},
            {"self": [1, 12, 9, 9], "other": [1, 1, 1, 9]},
            {"self": [1, 120, 17, 17], "other": [1, 120, 17, 17]},
            {"self": [1, 120, 28, 28], "other": [1, 120, 28, 28]},
            {"self": [1, 1200, 320], "other": [1, 1200, 320]},
            {"self": [1, 1232, 14, 14], "other": [1, 1232, 14, 14]},
            {"self": [1, 128, 100, 136], "other": [1, 128, 1, 1]},
            {"self": [1, 128, 128, 128], "other": [1, 128, 128, 128]},
            {"self": [1, 128, 1536], "other": [1, 128, 1536]},
            {"self": [1, 128, 180, 320], "other": [1, 128, 1, 1]},
            {"self": [1, 128, 200, 272], "other": [1, 128, 1, 1]},
            {"self": [1, 128, 28, 28], "other": [1, 128, 28, 28]},
            {"self": [1, 128, 56, 56], "other": [1, 128, 56, 56]},
            {"self": [1, 128, 75, 75], "other": [1, 128, 75, 75]},
            {"self": [1, 128, 90, 160], "other": [1, 128, 1, 1]},
            {"self": [1, 1280, 16, 16], "other": [1, 1280, 1, 1]},
            {"self": [1, 1280, 16, 16], "other": [1, 1280, 16, 16]},
            {"self": [1, 1280, 8, 8], "other": [1, 1280, 1, 1]},
            {"self": [1, 1280, 8, 8], "other": [1, 1280, 8, 8]},
            {"self": [1, 1344, 14, 14], "other": [1, 1344, 14, 14]},
            {"self": [1, 136, 19, 19], "other": [1, 136, 19, 19]},
            {"self": [1, 1370, 1280], "other": [1, 1370, 1280]},
            {"self": [1, 1392, 14, 14], "other": [1, 1392, 14, 14]},
            {"self": [1, 14, 128], "other": [1, 14, 128]},
            {"self": [1, 14, 14, 384], "other": [1, 14, 14, 384]},
            {"self": [1, 14, 14, 512], "other": [1, 14, 14, 512]},
            {"self": [1, 14, 3072], "other": [1, 14, 3072]},
            {"self": [1, 14, 768], "other": [1, 14, 768]},
            {"self": [1, 144, 28, 28], "other": [1, 144, 28, 28]},
            {"self": [1, 144, 7, 7], "other": [1, 144, 7, 7]},
            {"self": [1, 1445, 192], "other": [1, 1445, 192]},
            {"self": [1, 15, 1024], "other": [1, 15, 1024]},
            {"self": [1, 15, 512], "other": [1, 15, 512]},
            {"self": [1, 1500, 768], "other": [1, 1500, 768]},
            {"self": [1, 1500, 768], "other": [1500, 768]},
            {"self": [1, 1512, 7, 7], "other": [1, 1512, 7, 7]},
            {"self": [1, 16, 1, 10], "other": [1, 1, 1, 10]},
            {"self": [1, 16, 1, 10], "other": [1, 16, 1, 10]},
            {"self": [1, 16, 1, 1], "other": [1, 1, 1, 1]},
            {"self": [1, 16, 1, 1], "other": [1, 16, 1, 1]},
            {"self": [1, 16, 1, 2], "other": [1, 1, 1, 2]},
            {"self": [1, 16, 1, 2], "other": [1, 16, 1, 2]},
            {"self": [1, 16, 1, 60], "other": [1, 1, 1, 60]},
            {"self": [1, 16, 1, 6], "other": [1, 1, 1, 6]},
            {"self": [1, 16, 10, 10], "other": [1, 1, 1, 10]},
            {"self": [1, 16, 10, 10], "other": [1, 16, 10, 10]},
            {"self": [1, 16, 112, 112], "other": [1, 16, 112, 112]},
            {"self": [1, 16, 16, 384], "other": [1, 16, 16, 384]},
            {"self": [1, 16, 16, 512], "other": [1, 16, 16, 512]},
            {"self": [1, 16, 160, 160], "other": [1, 16, 160, 160]},
            {"self": [1, 16, 19, 19], "other": [1, 1, 19, 19]},
            {"self": [1, 16, 197, 197], "other": [1, 16, 197, 197]},
            {"self": [1, 16, 256, 256], "other": [1, 1, 1, 256]},
            {"self": [1, 16, 5, 5], "other": [1, 1, 1, 5]},
            {"self": [1, 16, 59, 59], "other": [1, 1, 59, 59]},
            {"self": [1, 16, 6, 49, 49], "other": [1, 16, 1, 49, 49]},
            {"self": [1, 16, 6, 64, 64], "other": [1, 16, 1, 64, 64]},
            {"self": [1, 16, 768], "other": [1, 16, 768]},
            {"self": [1, 16, 8, 49, 49], "other": [1, 16, 1, 49, 49]},
            {"self": [1, 16, 8, 64, 64], "other": [1, 16, 1, 64, 64]},
            {"self": [1, 16, 9, 9], "other": [1, 1, 1, 9]},
            {"self": [1, 160, 14, 14], "other": [1, 160, 14, 14]},
            {"self": [1, 160, 24, 24], "other": [1, 160, 24, 24]},
            {"self": [1, 160, 7, 7], "other": [1, 160, 7, 7]},
            {"self": [1, 16384, 256], "other": [256]},
            {"self": [1, 16384, 32], "other": [1, 16384, 32]},
            {"self": [1, 168, 28, 28], "other": [1, 168, 28, 28]},
            {"self": [1, 18, 56, 56], "other": [1, 18, 56, 56]},
            {"self": [1, 19, 1024], "other": [1, 19, 1024]},
            {"self": [1, 192, 28, 28], "other": [1, 192, 28, 28]},
            {"self": [1, 192, 32, 42], "other": [1, 192, 32, 42]},
            {"self": [1, 192, 7, 7], "other": [1, 192, 7, 7]},
            {"self": [1, 192, 8, 8], "other": [1, 192, 8, 8]},
            {"self": [1, 1920, 7, 7], "other": [1, 1920, 7, 7]},
            {"self": [1, 19200, 64], "other": [1, 19200, 64]},
            {"self": [1, 193, 768], "other": [1, 193, 768]},
            {"self": [1, 196, 768], "other": [1, 196, 768]},
            {"self": [1, 197, 1024], "other": [1, 197, 1024]},
            {"self": [1, 197, 768], "other": [1, 197, 768]},
            {"self": [1, 201, 768], "other": [1, 201, 768]},
            {"self": [1, 2016, 7, 7], "other": [1, 2016, 7, 7]},
            {"self": [1, 2048, 23, 40], "other": [1, 2048, 1, 1]},
            {"self": [1, 2048, 23, 40], "other": [1, 2048, 23, 40]},
            {"self": [1, 2048, 25, 34], "other": [1, 2048, 1, 1]},
            {"self": [1, 2048, 25, 34], "other": [1, 2048, 25, 34]},
            {"self": [1, 2048, 7, 7], "other": [1, 2048, 7, 7]},
            {"self": [1, 2048, 768], "other": [1, 2048, 768]},
            {"self": [1, 2048, 768], "other": [2048, 768]},
            {"self": [1, 208, 14, 14], "other": [1, 208, 14, 14]},
            {"self": [1, 208, 9, 9], "other": [1, 208, 9, 9]},
            {"self": [1, 216, 28, 28], "other": [1, 216, 28, 28]},
            {"self": [1, 224, 56, 56], "other": [1, 224, 56, 56]},
            {"self": [1, 232, 10, 10], "other": [1, 232, 10, 10]},
            {"self": [1, 232, 56, 56], "other": [1, 232, 56, 56]},
            {"self": [1, 24, 28, 28], "other": [1, 24, 28, 28]},
            {"self": [1, 24, 49, 49], "other": [1, 24, 49, 49]},
            {"self": [1, 24, 56, 56], "other": [1, 24, 56, 56]},
            {"self": [1, 24, 60, 60], "other": [1, 24, 60, 60]},
            {"self": [1, 24, 64, 64], "other": [1, 24, 64, 64]},
            {"self": [1, 24, 65, 65], "other": [1, 24, 65, 65]},
            {"self": [1, 24, 768], "other": [1, 24, 768]},
            {"self": [1, 24, 80, 80], "other": [1, 24, 80, 80]},
            {"self": [1, 240, 28, 28], "other": [1, 240, 28, 28]},
            {"self": [1, 25, 768], "other": [1, 25, 768]},
            {"self": [1, 2520, 7, 7], "other": [1, 2520, 7, 7]},
            {"self": [1, 256, 100, 136], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 100, 136], "other": [1, 256, 100, 136]},
            {"self": [1, 256, 1024], "other": [1, 256, 1024]},
            {"self": [1, 256, 128, 128], "other": [1, 256, 128, 128]},
            {"self": [1, 256, 1280], "other": [1, 256, 1280]},
            {"self": [1, 256, 14, 14], "other": [1, 256, 14, 14]},
            {"self": [1, 256, 180, 320], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 180, 320], "other": [1, 256, 180, 320]},
            {"self": [1, 256, 200, 272], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 200, 272], "other": [1, 256, 200, 272]},
            {"self": [1, 256, 256], "other": [1, 256, 256]},
            {"self": [1, 256, 256], "other": [256]},
            {"self": [1, 256, 28, 28], "other": [1, 256, 28, 28]},
            {"self": [1, 256, 38, 38], "other": [1, 256, 38, 38]},
            {"self": [1, 256, 384], "other": [1, 256, 384]},
            {"self": [1, 256, 45, 80], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 50, 68], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 50, 68], "other": [1, 256, 50, 68]},
            {"self": [1, 256, 512], "other": [1, 256, 512]},
            {"self": [1, 256, 56, 56], "other": [1, 256, 56, 56]},
            {"self": [1, 256, 64, 64], "other": [1, 256, 64, 64]},
            {"self": [1, 256, 75, 75], "other": [1, 256, 75, 75]},
            {"self": [1, 256, 90, 160], "other": [1, 256, 1, 1]},
            {"self": [1, 272, 12, 12], "other": [1, 272, 12, 12]},
            {"self": [1, 28, 28, 192], "other": [1, 28, 28, 192]},
            {"self": [1, 28, 28, 256], "other": [1, 28, 28, 256]},
            {"self": [1, 288, 14, 14], "other": [1, 288, 14, 14]},
            {"self": [1, 2904, 24, 24], "other": [1, 2904, 24, 24]},
            {"self": [1, 3, 16, 16, 2], "other": [1, 3, 16, 16, 2]},
            {"self": [1, 3, 300, 300], "other": [1, 3, 300, 300]},
            {"self": [1, 3, 32, 32, 2], "other": [1, 3, 32, 32, 2]},
            {"self": [1, 3, 320, 320], "other": [1, 3, 320, 320]},
            {"self": [1, 3, 64, 64, 2], "other": [1, 3, 64, 64, 2]},
            {"self": [1, 3, 800, 1066], "other": [1, 3, 800, 1066]},
            {"self": [1, 300, 512], "other": [1, 300, 512]},
            {"self": [1, 3024, 7, 7], "other": [1, 3024, 7, 7]},
            {"self": [1, 32, 1536], "other": [1, 32, 1536]},
            {"self": [1, 32, 24576], "other": [1, 32, 24576]},
            {"self": [1, 32, 28, 28], "other": [1, 32, 28, 28]},
            {"self": [1, 32, 32, 192], "other": [1, 32, 32, 192]},
            {"self": [1, 32, 32, 256], "other": [1, 32, 32, 256]},
            {"self": [1, 32, 49, 49], "other": [1, 32, 49, 49]},
            {"self": [1, 32, 56, 56], "other": [1, 32, 56, 56]},
            {"self": [1, 32, 64, 64], "other": [1, 32, 64, 64]},
            {"self": [1, 32, 75, 75], "other": [1, 32, 75, 75]},
            {"self": [1, 32, 95, 95], "other": [1, 32, 95, 95]},
            {"self": [1, 320, 14, 14], "other": [1, 320, 14, 14]},
            {"self": [1, 320, 64, 64], "other": [1, 320, 1, 1]},
            {"self": [1, 320, 64, 64], "other": [1, 320, 64, 64]},
            {"self": [1, 336, 14, 14], "other": [1, 336, 14, 14]},
            {"self": [1, 336, 56, 56], "other": [1, 336, 56, 56]},
            {"self": [1, 36, 28, 28], "other": [1, 36, 28, 28]},
            {"self": [1, 3712, 7, 7], "other": [1, 3712, 7, 7]},
            {"self": [1, 4, 12, 49, 49], "other": [1, 4, 1, 49, 49]},
            {"self": [1, 4, 12, 64, 64], "other": [1, 4, 1, 64, 64]},
            {"self": [1, 4, 16, 49, 49], "other": [1, 4, 1, 49, 49]},
            {"self": [1, 4, 16, 64, 64], "other": [1, 4, 1, 64, 64]},
            {"self": [1, 4, 768], "other": [1, 4, 768]},
            {"self": [1, 4, 768], "other": [4, 768]},
            {"self": [1, 40, 14, 14], "other": [1, 40, 14, 14]},
            {"self": [1, 40, 28, 28], "other": [1, 40, 28, 28]},
            {"self": [1, 40, 30, 30], "other": [1, 40, 30, 30]},
            {"self": [1, 40, 40, 40], "other": [1, 40, 40, 40]},
            {"self": [1, 400, 7, 7], "other": [1, 400, 7, 7]},
            {"self": [1, 408, 14, 14], "other": [1, 408, 14, 14]},
            {"self": [1, 4096, 256], "other": [256]},
            {"self": [1, 4096, 320], "other": [1, 4096, 320]},
            {"self": [1, 4096, 64], "other": [1, 4096, 64]},
            {"self": [1, 432, 14, 14], "other": [1, 432, 14, 14]},
            {"self": [1, 440, 7, 7], "other": [1, 440, 7, 7]},
            {"self": [1, 448, 28, 28], "other": [1, 448, 28, 28]},
            {"self": [1, 45, 3072], "other": [1, 45, 3072]},
            {"self": [1, 45, 768], "other": [1, 45, 768]},
            {"self": [1, 48, 14, 14], "other": [1, 48, 14, 14]},
            {"self": [1, 48, 33, 33], "other": [1, 48, 33, 33]},
            {"self": [1, 48, 38, 38], "other": [1, 48, 38, 38]},
            {"self": [1, 48, 56, 56], "other": [1, 48, 56, 56]},
            {"self": [1, 4800, 128], "other": [1, 4800, 128]},
            {"self": [1, 5, 1024], "other": [1, 5, 1024]},
            {"self": [1, 5, 16, 32], "other": [1, 5, 16, 32]},
            {"self": [1, 5, 4096], "other": [1, 5, 4096]},
            {"self": [1, 50, 1024], "other": [1, 50, 1024]},
            {"self": [1, 50, 768], "other": [1, 50, 768]},
            {"self": [1, 512, 100, 136], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 100, 136], "other": [1, 512, 100, 136]},
            {"self": [1, 512, 14, 14], "other": [1, 512, 14, 14]},
            {"self": [1, 512, 23, 40], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 25, 34], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 28, 28], "other": [1, 512, 28, 28]},
            {"self": [1, 512, 32, 32], "other": [1, 512, 32, 32]},
            {"self": [1, 512, 45, 80], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 50, 68], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 7, 7], "other": [1, 512, 7, 7]},
            {"self": [1, 512, 90, 160], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 90, 160], "other": [1, 512, 90, 160]},
            {"self": [1, 528, 96, 96], "other": [1, 528, 96, 96]},
            {"self": [1, 56, 48, 48], "other": [1, 56, 48, 48]},
            {"self": [1, 56, 56, 128], "other": [1, 56, 56, 128]},
            {"self": [1, 56, 56, 96], "other": [1, 56, 56, 96]},
            {"self": [1, 576, 14, 14], "other": [1, 576, 14, 14]},
            {"self": [1, 59, 1024], "other": [1, 59, 1024]},
            {"self": [1, 6, 1, 15], "other": [1, 1, 1, 15]},
            {"self": [1, 6, 1, 15], "other": [1, 6, 1, 15]},
            {"self": [1, 6, 1, 17], "other": [1, 1, 1, 17]},
            {"self": [1, 6, 1, 17], "other": [1, 6, 1, 17]},
            {"self": [1, 6, 1, 1], "other": [1, 1, 1, 1]},
            {"self": [1, 6, 1, 1], "other": [1, 6, 1, 1]},
            {"self": [1, 6, 1, 2], "other": [1, 1, 1, 2]},
            {"self": [1, 6, 1, 2], "other": [1, 6, 1, 2]},
            {"self": [1, 6, 15, 15], "other": [1, 1, 1, 15]},
            {"self": [1, 6, 15, 15], "other": [1, 6, 15, 15]},
            {"self": [1, 64, 120, 160], "other": [1, 64, 120, 160]},
            {"self": [1, 64, 1280], "other": [1, 64, 1280]},
            {"self": [1, 64, 14, 14], "other": [1, 64, 14, 14]},
            {"self": [1, 64, 180, 320], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 200, 272], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 240, 320], "other": [1, 64, 240, 320]},
            {"self": [1, 64, 256, 256], "other": [1, 64, 256, 256]},
            {"self": [1, 64, 28, 28], "other": [1, 64, 28, 28]},
            {"self": [1, 64, 3, 49, 49], "other": [1, 64, 1, 49, 49]},
            {"self": [1, 64, 3, 64, 64], "other": [1, 64, 1, 64, 64]},
            {"self": [1, 64, 30, 40], "other": [1, 64, 30, 40]},
            {"self": [1, 64, 360, 640], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 4, 49, 49], "other": [1, 64, 1, 49, 49]},
            {"self": [1, 64, 4, 64, 64], "other": [1, 64, 1, 64, 64]},
            {"self": [1, 64, 400, 544], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 480, 640], "other": [1, 64, 480, 640]},
            {"self": [1, 64, 56, 56], "other": [1, 64, 56, 56]},
            {"self": [1, 64, 60, 80], "other": [1, 64, 60, 80]},
            {"self": [1, 64, 6144], "other": [1, 64, 6144]},
            {"self": [1, 64, 64, 128], "other": [1, 64, 64, 128]},
            {"self": [1, 64, 64, 96], "other": [1, 64, 64, 96]},
            {"self": [1, 64, 9, 9], "other": [1, 1, 1, 9]},
            {"self": [1, 640, 32, 32], "other": [1, 640, 1, 1]},
            {"self": [1, 640, 32, 32], "other": [1, 640, 32, 32]},
            {"self": [1, 672, 28, 28], "other": [1, 672, 28, 28]},
            {"self": [1, 672, 7, 7], "other": [1, 672, 7, 7]},
            {"self": [1, 696, 28, 28], "other": [1, 696, 28, 28]},
            {"self": [1, 7, 3072], "other": [1, 7, 3072]},
            {"self": [1, 7, 4544], "other": [1, 7, 4544]},
            {"self": [1, 7, 7, 1024], "other": [1, 7, 7, 1024]},
            {"self": [1, 7, 7, 768], "other": [1, 7, 7, 768]},
            {"self": [1, 7, 768], "other": [1, 7, 768]},
            {"self": [1, 71, 7, 64], "other": [1, 71, 7, 64]},
            {"self": [1, 71, 7, 7], "other": [7, 7]},
            {"self": [1, 72, 14, 14], "other": [1, 72, 14, 14]},
            {"self": [1, 72, 56, 56], "other": [1, 72, 56, 56]},
            {"self": [1, 720, 14, 14], "other": [1, 720, 14, 14]},
            {"self": [1, 728, 19, 19], "other": [1, 728, 19, 19]},
            {"self": [1, 728, 38, 38], "other": [1, 728, 38, 38]},
            {"self": [1, 7392, 12, 12], "other": [1, 7392, 12, 12]},
            {"self": [1, 768, 384], "other": [384]},
            {"self": [1, 784, 7, 7], "other": [1, 784, 7, 7]},
            {"self": [1, 8, 1, 10], "other": [1, 1, 1, 10]},
            {"self": [1, 8, 1, 10], "other": [1, 8, 1, 10]},
            {"self": [1, 8, 1, 1], "other": [1, 1, 1, 1]},
            {"self": [1, 8, 1, 1], "other": [1, 8, 1, 1]},
            {"self": [1, 8, 1, 2], "other": [1, 1, 1, 2]},
            {"self": [1, 8, 1, 2], "other": [1, 8, 1, 2]},
            {"self": [1, 8, 10, 10], "other": [1, 1, 1, 10]},
            {"self": [1, 8, 10, 10], "other": [1, 8, 10, 10]},
            {"self": [1, 8, 256, 2048], "other": [1, 1, 1, 2048]},
            {"self": [1, 8, 768], "other": [1, 8, 768]},
            {"self": [1, 8, 8, 1024], "other": [1, 8, 8, 1024]},
            {"self": [1, 8, 8, 768], "other": [1, 8, 8, 768]},
            {"self": [1, 80, 10, 10], "other": [1, 80, 10, 10]},
            {"self": [1, 80, 14, 14], "other": [1, 80, 14, 14]},
            {"self": [1, 80, 15, 15], "other": [1, 80, 15, 15]},
            {"self": [1, 80, 20, 20], "other": [1, 80, 20, 20]},
            {"self": [1, 80, 56, 56], "other": [1, 80, 56, 56]},
            {"self": [1, 88, 17, 17], "other": [1, 88, 17, 17]},
            {"self": [1, 888, 7, 7], "other": [1, 888, 7, 7]},
            {"self": [1, 896, 14, 14], "other": [1, 896, 14, 14]},
            {"self": [1, 9, 1024], "other": [1, 9, 1024]},
            {"self": [1, 9, 128], "other": [1, 9, 128]},
            {"self": [1, 9, 16384], "other": [1, 9, 16384]},
            {"self": [1, 9, 2048], "other": [1, 9, 2048]},
            {"self": [1, 9, 3072], "other": [1, 9, 3072]},
            {"self": [1, 9, 4096], "other": [1, 9, 4096]},
            {"self": [1, 9, 768], "other": [1, 9, 768]},
            {"self": [1, 9, 8192], "other": [1, 9, 8192]},
            {"self": [1, 912, 7, 7], "other": [1, 912, 7, 7]},
            {"self": [1, 96, 14, 14], "other": [1, 96, 14, 14]},
            {"self": [1, 96, 19, 19], "other": [1, 96, 19, 19]},
            {"self": [1, 96, 56, 56], "other": [1, 96, 56, 56]},
            {"self": [1, 96, 7, 7], "other": [1, 96, 7, 7]},
            {"self": [1, 96, 80], "other": [1, 96, 80]},
            {"self": [10, 10], "other": [10, 10]},
            {"self": [100, 1, 256], "other": [100, 1, 256]},
            {"self": [12, 24, 24], "other": [12, 24, 24]},
            {"self": [13600, 1, 4], "other": [1, 9, 4]},
            {"self": [15, 15], "other": [15, 15]},
            {"self": [16, 6, 49, 49], "other": [1, 6, 49, 49]},
            {"self": [16, 6, 64, 64], "other": [1, 6, 64, 64]},
            {"self": [16, 8, 49, 49], "other": [1, 8, 49, 49]},
            {"self": [16, 8, 64, 64], "other": [1, 8, 64, 64]},
            {"self": [2, 7, 512], "other": [1, 7, 512]},
            {"self": [2, 7, 512], "other": [2, 7, 512]},
            {"self": [2, 8, 7, 7], "other": [2, 1, 7, 7]},
            {"self": [2048, 262], "other": [262]},
            {"self": [221, 1, 4], "other": [1, 9, 4]},
            {"self": [25, 4], "other": [25, 1]},
            {"self": [3234, 1], "other": [3234, 1]},
            {"self": [3234, 2], "other": [3234, 2]},
            {"self": [3234], "other": [3234]},
            {"self": [3400, 1, 4], "other": [1, 9, 4]},
            {"self": [4, 12, 49, 49], "other": [1, 12, 49, 49]},
            {"self": [4, 12, 64, 64], "other": [1, 12, 64, 64]},
            {"self": [4, 16, 49, 49], "other": [1, 16, 49, 49]},
            {"self": [4, 16, 64, 64], "other": [1, 16, 64, 64]},
            {"self": [59, 1024], "other": [59, 1024]},
            {"self": [63, 1, 4], "other": [1, 9, 4]},
            {"self": [64, 3, 49, 49], "other": [1, 3, 49, 49]},
            {"self": [64, 3, 64, 64], "other": [1, 3, 64, 64]},
            {"self": [64, 4, 49, 49], "other": [1, 4, 49, 49]},
            {"self": [64, 4, 64, 64], "other": [1, 4, 64, 64]},
            {"self": [850, 1, 4], "other": [1, 9, 4]},
            {"self": [8732, 1], "other": [8732, 1]},
            {"self": [8732, 2], "other": [8732, 2]},
            {"self": [8732], "other": [8732]},
            # {"self": [], "other": []}, #without empty tensor
            {"self": [920, 1, 256], "other": [256]},
            {"self": [920, 1, 256], "other": [920, 1, 256]},
            {"self": [1, 1, 1, 42], "other": -6.0},
            {"self": [1, 1, 1, 42], "other": 0.5},
            {"self": [1, 1, 1, 42], "other": 1.0},
            {"self": [1, 1, 1, 42], "other": 1.0},
            {"self": [1, 1, 1, 42], "other": 2.0},
            {"self": [1, 1, 1024], "other": 1.0},
            {"self": [1, 1, 1], "other": 1e-06},
            {"self": [1, 1, 224, 224], "other": -0.030000000000000027},
            {"self": [1, 1, 224, 224], "other": -0.08799999999999997},
            {"self": [1, 1, 224, 224], "other": -0.18799999999999994},
            {"self": [1, 1, 3072], "other": 1.0},
            {"self": [1, 1, 32, 1], "other": -6.0},
            {"self": [1, 1, 32, 1], "other": 0.5},
            {"self": [1, 1, 32, 1], "other": 1.0},
            {"self": [1, 1, 32, 1], "other": 1.0},
            {"self": [1, 1, 32, 1], "other": 2.0},
            {"self": [1, 1, 4096], "other": 1.0},
            {"self": [1, 1, 40], "other": 1e-06},
            {"self": [1, 10, 1], "other": 1e-06},
            {"self": [1, 1024, 1, 1], "other": 0.0},
            {"self": [1, 1024, 1, 1], "other": 1e-05},
            {"self": [1, 10], "other": 0.0},
            {"self": [1, 10], "other": 1.0},
            {"self": [1, 12, 3072], "other": 1.0},
            {"self": [1, 128, 1, 1], "other": 0.0},
            {"self": [1, 128, 1, 1], "other": 1e-05},
            {"self": [1, 14, 3072], "other": 1.0},
            {"self": [1, 15, 1024], "other": 1.0},
            {"self": [1, 15, 1], "other": 1e-06},
            {"self": [1, 19], "other": 2.0},
            {"self": [1, 1], "other": 0.0},
            {"self": [1, 1], "other": 16.0},
            {"self": [1, 1], "other": 2.0},
            {"self": [1, 2048, 1, 1], "other": 0.0},
            {"self": [1, 2048, 1, 1], "other": 1e-05},
            {"self": [1, 23, 1], "other": 1e-06},
            {"self": [1, 256, 1, 1], "other": 0.0},
            {"self": [1, 256, 1, 1], "other": 1e-05},
            {"self": [1, 32, 6144], "other": 1.0},
            {"self": [1, 32, 6144], "other": 1.0},
            {"self": [1, 45, 3072], "other": 1.0},
            {"self": [1, 5, 4096], "other": 1.0},
            {"self": [1, 512, 1, 1], "other": 0.0},
            {"self": [1, 512, 1, 1], "other": 1e-05},
            {"self": [1, 59], "other": 2.0},
            {"self": [1, 64, 1, 1], "other": 0.0},
            {"self": [1, 64, 1, 1], "other": 1e-05},
            {"self": [1, 7, 3072], "other": 1.0},
            {"self": [1, 9, 128], "other": 1.0},
            {"self": [1, 9, 16384], "other": 1.0},
            {"self": [1, 9, 3072], "other": 1.0},
            {"self": [1, 9, 4096], "other": 1.0},
            {"self": [1, 9, 8192], "other": 1.0},
            {"self": [10, 10], "other": 0.0},
            {"self": [10, 10], "other": 8.0},
            {"self": [100], "other": 0.0},
            {"self": [1066], "other": 0.5},
            {"self": [10], "other": 0.5},
            {"self": [120], "other": 0.5},
            {"self": [128], "other": 0.5},
            {"self": [12], "other": 0.0},
            {"self": [136], "other": 0.0},
            {"self": [14], "other": 0.0},
            {"self": [15, 15], "other": 0.0},
            {"self": [15, 15], "other": 8.0},
            {"self": [160], "other": 0.5},
            {"self": [16], "other": 0.0},
            {"self": [17, 17], "other": 0.0},
            {"self": [17, 17], "other": 16.0},
            {"self": [19], "other": 0.5},
            {"self": [1], "other": 0.5},
            {"self": [2, 2], "other": 0.0},
            {"self": [2, 2], "other": 16.0},
            {"self": [20], "other": 0.5},
            {"self": [23], "other": 0.0},
            {"self": [24, 24], "other": 160.0},
            {"self": [240], "other": 0.5},
            {"self": [28], "other": 0.0},
            {"self": [2], "other": 0.5},
            {"self": [300], "other": 0.5},
            {"self": [30], "other": 0.5},
            {"self": [320], "other": 0.5},
            {"self": [32], "other": 0.0},
            {"self": [38], "other": 0.5},
            {"self": [3], "other": 0.5},
            {"self": [40], "other": 0.0},
            {"self": [40], "other": 0.5},
            {"self": [480], "other": 0.5},
            {"self": [50], "other": 0.0},
            {"self": [56], "other": 0.0},
            {"self": [5], "other": 0.5},
            {"self": [60], "other": 0.5},
            {"self": [640], "other": 0.5},
            {"self": [64], "other": 0.0},
            {"self": [68], "other": 0.0},
            {"self": [7], "other": 0.0},
            {"self": [800], "other": 0.5},
            {"self": [80], "other": 0.5},
            # {"self": [], "other": 1}, #without empty tensor
        ],
        # {"self": [s0 + 1, s0 + 1], "other": 16},
        # {"self": [s0 + 1, s0 + 1], "other": 0},
        # {"self": [1, 16, 1, "s0 + 1"], "other": [1, 1, 1, "s0 + 1"]},
        # {"self": [1, 16, 1, "s0 + 1"], "other": [1, 16, 1, "s0 + 1"]},
        # {"self": [1, 8, 1, "s0 + 1"], "other": [1, 1, 1, "s0 + 1"]},
        # {"self": [1, 8, 1, "s0 + 1"], "other": [1, 8, 1, "s0 + 1"]},
        # {"self": [1, 6, 1, "s0 + 1"], "other": [1, 1, 1, "s0 + 1"]},
        # {"self": [1, 6, 1, "s0 + 1"], "other": [1, 6, 1, "s0 + 1"]},
        # {"self": [1, 12, 1, "s0 + 1"], "other": [1, 1, 1, "s0 + 1"]},
        # {"self": [1, 12, 1, "s0 + 1"], "other": [1, 12, 1, "s0 + 1"]},
        # {"self": [1, 32, "s0", "s1"], "other": [1, 32, "s0", "s1"]},
        # {"self": [1, 12, 1, "s10 + 1"], "other": [1, 1, 1, "s10 + 1"]},
        # {"self": [1, 64, "s1", "s2"], "other": [1, 64, "s1", "s2"]},
        # {"self": [1, 128, "s1", "s2"], "other": [1, 128, "s1", "s2"]},
        # {"self": [1, 16, 1, "s10 + 1"], "other": [1, 1, 1, "s10 + 1"]},
        # {"self": [1, 256, "s1", "s2"], "other": [1, 256, "s1", "s2"]},
        # {"self": [1, "s0", 768], "other": [1, "s0", 768]}
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid. len(test_vector["input_shape"]["other"]) >= 4
# def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
#     if isinstance(test_vector["input_shape"]["other"], list) and len(test_vector["input_shape"]["self"]) >= 4:
#         is_less_than_3d = len(test_vector["input_shape"]["other"]) < 3
#         c_index = test_vector["input_shape"]["self"][-3]
#         if is_less_than_3d or (
#             len(test_vector["input_shape"]["other"]) >= 3 and test_vector["input_shape"]["other"][-3] < c_index
#         ):
#             print("checking channel bcast")
#             print("input ", test_vector["input_shape"]["self"])
#             print("other ", test_vector["input_shape"]["other"])
#             return True, "channel dim bcast not supported"

#     return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), input_a_dtype
    )(input_shape["self"])

    if isinstance(input_shape["other"], list):
        if len(input_shape["other"]):
            torch_input_tensor_b = gen_func_with_cast_tt(
                partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), input_b_dtype
            )(input_shape["other"])
        else:
            print("input shape", input_shape)
            torch_input_tensor_b = torch.tensor(0, dtype=torch.bfloat16)
            print("torch_input_tensor_b shape", torch_input_tensor_b)
    else:
        torch_input_tensor_b = torch.tensor(input_shape["other"], dtype=torch.bfloat16)
        # torch_input_tensor_b = input_shape["other"]

    golden_function = ttnn.get_golden_function(ttnn.add)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    # if isinstance(input_shape["other"], list):
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )
    # else:
    #     input_tensor_b = input_shape["other"]

    start_time = start_measuring_time()
    result = ttnn.add(input_tensor_a, input_tensor_b)

    # handles 1 D input_a and scalar or empty [] input_b
    if len(input_shape["self"]) == 1 and (not isinstance(input_shape["other"], list) or not input_shape["other"]):
        output_tensor = ttnn.to_torch(result, original_shape=input_shape["self"])
    else:
        output_tensor = ttnn.to_torch(result)

    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, pcc=0.999), e2e_perf]
