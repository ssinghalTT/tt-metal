# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import gc


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()
