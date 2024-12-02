# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_metal.tools.profiler.merge_meta_class import MergeMetaclass
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG


class default_setup(metaclass=MergeMetaclass):
    riscs = [
        "BRISC",
        "NCRISC",
        "TRISC_0",
        "TRISC_1",
        "TRISC_2",
        "ERISC",
    ]

    riscTypes = [
        "BRISC",
        "NCRISC",
        "TRISC",
        "ERISC",
    ]

    timerAnalysis = {
        "device_fw_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-FW" for risc in riscTypes]},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-FW" for risc in riscTypes]},
        },
        "device_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-KERNEL" for risc in riscTypes]},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-KERNEL" for risc in riscTypes]},
        },
        "device_brisc_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
        "device_ncrisc_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
        },
        "device_trisc0_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
        },
        "device_trisc1_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
        },
        "device_trisc2_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
        },
        "device_erisc_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ERISC", "zone_name": "ERISC-KERNEL"},
            "end": {"core": "ANY", "risc": "ERISC", "zone_name": "ERISC-KERNEL"},
        },
        "device_compute_cb_wait_front": {
            "across": "ops",
            "type": "sum",
            "marker": {"risc": "TRISC_0", "zone_name": "CB-COMPUTE-WAIT-FRONT"},
        },
        "device_compute_cb_reserve_back": {
            "across": "ops",
            "type": "sum",
            "marker": {"risc": "TRISC_2", "zone_name": "CB-COMPUTE-RESERVE-BACK"},
        },
    }

    displayStats = ["Count", "Average", "Max", "Median", "Min", "Sum", "Range"]

    detectOps = True

    outputFolder = f"output/device"
    deviceInputLog = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"
    deviceAnalysisData = "device_analysis_data.json"
    deviceStatsTXT = "device_stats.txt"
    deviceTarball = "device_perf_results.tgz"


class test_multi_op(default_setup):
    timerAnalysis = {
        "BRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
    }
    detectOps = False


class test_custom_cycle_count(default_setup):
    timerAnalysis = {
        "BRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
        "NCRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
        },
        "TRISC_0 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
        },
        "TRISC_1 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
        },
        "TRISC_2 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
        },
    }
    detectOps = False


class test_custom_cycle_count_slow_dispatch(default_setup):
    timerAnalysis = {
        "BRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
        "NCRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
        },
        "TRISC_0 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
        },
        "TRISC_1 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
        },
        "TRISC_2 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
        },
    }
    detectOps = False


class test_full_buffer(default_setup):
    timerAnalysis = {
        "Marker Repeat": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "zone_name": "TEST-FULL"},
            "end": {"risc": "ANY", "zone_name": "TEST-FULL"},
        },
        "Marker Repeat ETH": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ERISC", "zone_name": "TEST-FULL"},
            "end": {"risc": "ERISC", "zone_name": "TEST-FULL"},
        },
    }
    detectOps = False


class test_dispatch_cores(default_setup):
    timerAnalysis = {
        "Tensix CQ Dispatch": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "zone_name": "CQ-DISPATCH"},
            "end": {"risc": "NCRISC", "zone_name": "CQ-DISPATCH"},
        },
        "Tensix CQ Prefetch": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "zone_name": "KERNEL-MAIN-HD"},
            "end": {"risc": "NCRISC", "zone_name": "KERNEL-MAIN-HD"},
        },
        "Ethernet CQ Dispatch": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ERISC", "zone_name": "CQ-DISPATCH"},
            "end": {"risc": "ERISC", "zone_name": "CQ-DISPATCH"},
        },
        "Ethernet CQ Prefetch": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ERISC", "zone_name": "KERNEL-MAIN-HD"},
            "end": {"risc": "ERISC", "zone_name": "KERNEL-MAIN-HD"},
        },
    }
    detectOps = False


class test_noc(default_setup):
    timerAnalysis = {
        "NoC For Loop": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "zone_name": "NOC-FOR-LOOP"},
            "end": {"risc": "NCRISC", "zone_name": "NOC-FOR-LOOP"},
        }
    }
    detectOps = False
