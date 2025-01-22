# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
import csv

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_possetupt_proc_config

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


def append_to_csv(file_path, header, data, write_header=True):
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or write_header:
            writer.writerow(header)
        writer.writerows([data])


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def profile_results(sample_count, sample_size, num_iterations):
    freq = get_device_freq() / 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    setup.timerAnalysis = {
        "SENDER-LOOP-ITER": {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ERISC", "zone_name": "SENDER-LOOP-ITER"},
            "end": {"core": "ANY", "risc": "ERISC", "zone_name": "SENDER-LOOP-ITER"},
        },
        "WAIT-ACKS-PHASE": {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ERISC", "zone_name": "WAIT-ACKS-PHASE"},
            "end": {"core": "ANY", "risc": "ERISC", "zone_name": "WAIT-ACKS-PHASE"},
        },
        "SEND-PAYLOADS-PHASE": {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ERISC", "zone_name": "SEND-PAYLOADS-PHASE"},
            "end": {"core": "ANY", "risc": "ERISC", "zone_name": "SEND-PAYLOADS-PHASE"},
        },
        "PING-REPLIES": {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ERISC", "zone_name": "PING-REPLIES"},
            "end": {"core": "ANY", "risc": "ERISC", "zone_name": "PING-REPLIES"},
        },
    }
    devices_data = import_log_run_stats(setup)
    device_0 = list(devices_data["devices"].keys())[0]
    device_1 = list(devices_data["devices"].keys())[1]

    # SENDER-LOOP-ITER
    sender_loop_cycle = devices_data["devices"][device_0]["cores"]["DEVICE"]["analysis"]["SENDER-LOOP-ITER"]["stats"][
        "Average"
    ]
    sender_loop_latency = sender_loop_cycle / freq

    # WAIT-ACKS-PHASE
    wait_ack_cycle = devices_data["devices"][device_0]["cores"]["DEVICE"]["analysis"]["WAIT-ACKS-PHASE"]["stats"][
        "Average"
    ]
    wait_ack_latency = wait_ack_cycle / freq

    # SEND-PAYLOADS-PHASE
    send_payload_cycle = devices_data["devices"][device_0]["cores"]["DEVICE"]["analysis"]["SEND-PAYLOADS-PHASE"][
        "stats"
    ]["Average"]
    send_payload_latency = send_payload_cycle / freq

    # PING-REPLIES
    ping_reply_cycle = devices_data["devices"][device_1]["cores"]["DEVICE"]["analysis"]["PING-REPLIES"]["stats"][
        "Average"
    ]
    ping_reply_latency = ping_reply_cycle / freq

    file_name = PROFILER_LOGS_DIR / "test_ethernet_link_ping_latency.py.csv"
    header = [
        "SAMPLE COUNT",
        "SAMPLE SIZE",
        "SENDER-LOOP-ITER (ns)",
        "WAIT-ACKS-PHASE",
        "SEND-PAYLOADS-PHASE",
        "PING-REPLIES",
    ]
    write_header = not os.path.exists(file_name)
    append_to_csv(
        file_name,
        header,
        [sample_count, sample_size, sender_loop_latency, wait_ack_latency, send_payload_latency, ping_reply_latency],
        write_header,
    )

    return sender_loop_latency, wait_ack_latency, send_payload_latency, ping_reply_latency


@pytest.mark.parametrize(
    "sample_sizes",
    [(16,), (1024,), (2048,), (4096,), (8192,), (16384,)],
)  # , 1024, 2048, 4096],
@pytest.mark.parametrize("sample_counts", [(1,), (100,), (1000,)])  # , 8, 16, 64, 256],
@pytest.mark.parametrize(
    "channel_counts",
    [(1,)],
)
def test_bidirectional_erisc_bandwidth(sample_counts, sample_sizes, channel_counts):
    test_string_name = f"test_ethernet_send_data_microbenchmark - \
            sample_counts: {sample_counts}, \
                sample_sizes: {sample_sizes}, \
                    channel_counts: {channel_counts}"
    print(f"{test_string_name}")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    sample_counts_str = " ".join([str(s) for s in sample_counts])
    sample_sizes_str = " ".join([str(s) for s in sample_sizes])
    channel_counts_str = " ".join([str(s) for s in channel_counts])

    print(sample_counts_str)
    print(sample_sizes_str)
    print(channel_counts_str)

    ARCH_NAME = os.getenv("ARCH_NAME")
    rc = os.system(
        f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_link_ping_latency_no_edm_{ARCH_NAME} \
                {len(sample_counts)} {sample_counts_str} \
                    {len(sample_sizes)} {sample_sizes_str} \
                        {len(channel_counts)} {channel_counts_str} \
            "
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    sender_loop_latency, wait_ack_latency, send_payload_latency, ping_reply_latency = profile_results(
        sample_counts[0], sample_sizes[0], sample_counts[0]
    )

    print(f"sender_loop_latency {sender_loop_latency}")
    print(f"wait_ack_latency {wait_ack_latency}")
    print(f"send_payload_latency {send_payload_latency}")
    print(f"ping_reply_latency {ping_reply_latency}")

    return True
