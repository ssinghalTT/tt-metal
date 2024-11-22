# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import pathlib
import importlib
import datetime
import os
import enlighten
import numpy as np
import json
from tt_metal.tools.profiler.process_ops_logs import get_device_data_generate_report
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR
from multiprocessing import Process, Queue
from queue import Empty
import subprocess
from framework.statuses import TestStatus, VectorValidity, VectorStatus
import framework.tt_smi_util as tt_smi_util
from elasticsearch import Elasticsearch, NotFoundError
from framework.elastic_config import *
from framework.sweeps_logger import sweeps_logger as logger

ARCH = os.getenv("ARCH_NAME")


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        return "Couldn't get git hash!"


def get_hostname():
    return subprocess.check_output(["uname", "-n"]).decode("ascii").strip()


def get_username():
    return os.environ["USER"]


def get_devices(test_module):
    try:
        return test_module.mesh_device_fixture()
    except:
        return default_device()


def gather_single_test_perf(device, test_passed, chunk_size=1):
    if not isinstance(device, ttnn.Device):
        logger.error("Multi-device perf is not supported. Failing.")
        return None
    ttnn.DumpDeviceProfiler(device)
    opPerfData = get_device_data_generate_report(
        PROFILER_LOGS_DIR, None, None, None, export_csv=False, cleanup_device_log=True
    )
    if not test_passed:
        return None
    if chunk_size == 1:
        if opPerfData == []:
            logger.error("No profiling data available. Ensure you are running with the profiler build.")
            return None
        elif len(opPerfData) > 1:
            keys = [
                "DEVICE KERNEL DURATION [ns]",
                "DEVICE BRISC KERNEL DURATION [ns]",
                "DEVICE NCRISC KERNEL DURATION [ns]",
                "DEVICE FW DURATION [ns]",
            ]
            avg_data = {}
            for key in keys:
                data = np.asarray([int(op[key]) for op in opPerfData])
                avg_data[key] = np.mean(data)
                avg_data[f"{key}_std"] = np.std(data)
            avg_data["runs"] = len(opPerfData)
            return avg_data
        else:
            return opPerfData[0]
    else:
        assert len(opPerfData) % chunk_size == 0
        keys = [
            "DEVICE KERNEL DURATION [ns]",
            "DEVICE BRISC KERNEL DURATION [ns]",
            "DEVICE NCRISC KERNEL DURATION [ns]",
            "DEVICE FW DURATION [ns]",
        ]
        runs_per_item = len(opPerfData) // chunk_size
        results = []
        for i in range(0, len(opPerfData), runs_per_item):
            avg_data = {}
            for key in keys:
                data = np.asarray([int(op[key]) for op in opPerfData[i : i + runs_per_item]])
                avg_data[key] = np.mean(data)
                avg_data[f"{key}_std"] = np.std(data)
            avg_data["runs"] = runs_per_item
            results.append(avg_data)
        return results


def run(test_module, input_queue, output_queue):
    device_generator = get_devices(test_module)
    try:
        device, device_name = next(device_generator)
        logger.info(f"Opened device configuration, {device_name}.")
    except AssertionError as e:
        output_queue.put([[False, "DEVICE EXCEPTION: " + str(e), None, None]])
        return
    try:
        while True:
            test_vectors = input_queue.get(block=True, timeout=1)
            test_vectors = [deserialize_vector(vec) for vec in test_vectors]
            try:
                # try chunked first:
                results = test_module.run_chunk(test_vectors, device)
                perf_results = gather_single_test_perf(device, True, len(test_vectors))
                e2e_perf = None
                results = [result + [e2e_perf, perf_result] for result, perf_result in zip(results, perf_results)]
                print("Batched run passed")
            except Exception as e:
                print("Batched run failed. Trying one by one")
                # retry one by one:
                _ = gather_single_test_perf(device, False, len(test_vectors))  # clean up the previous profiles
                results = []
                for test_vector in test_vectors:
                    try:
                        status, message = test_module.run(**test_vector, device=device)
                    except Exception as e:
                        status, message = False, str(e)
                    e2e_perf = None
                    perf_result = gather_single_test_perf(device, status, chunk_size=1)
                    results.append([status, message, e2e_perf, perf_result])
            output_queue.put(results)
    except Empty as e:
        try:
            # Run teardown in mesh_device_fixture
            next(device_generator)
        except StopIteration:
            logger.info(f"Closed device configuration, {device_name}.")


def get_timeout(test_module):
    try:
        timeout = test_module.TIMEOUT
    except:
        timeout = 30
    return timeout


def execute_suite(test_module, test_vectors, pbar_manager, suite_name):
    results = []
    input_queue = Queue()
    output_queue = Queue()
    p = None
    timeout = get_timeout(test_module)
    # result tabulation code doesn't work if chunk is set to 1
    chunk_size = 5
    suite_pbar = pbar_manager.counter(total=len(test_vectors), desc=f"Suite: {suite_name}", leave=False)
    for i in range(0, len(test_vectors), chunk_size):
        if i + chunk_size < len(test_vectors):
            actual_chunk_size = chunk_size
        else:
            actual_chunk_size = len(test_vectors) - i
        vec_chunk = test_vectors[i : i + actual_chunk_size]
        if DRY_RUN:
            print(f"Would have executed test for vectors {json.dumps(vec_chunk, indent=2)}")
            continue
        result_chunk = [{} for v in vec_chunk]
        for test_vector, result in zip(vec_chunk, result_chunk):
            if deserialize(test_vector["validity"]) == VectorValidity.INVALID:
                result["status"] = TestStatus.NOT_RUN
                result["exception"] = "INVALID VECTOR: " + test_vector["invalid_reason"]
                result["e2e_perf"] = None
            else:
                test_vector.pop("invalid_reason")
                test_vector.pop("status")
                test_vector.pop("validity")

        if p is None and len(test_vectors) > 1:
            p = Process(target=run, args=(test_module, input_queue, output_queue))
            p.start()
        try:
            if MEASURE_PERF:
                # E2E performance is not meaningful when we have batched execution
                raise NotImplementedError
            input_queue.put(vec_chunk)
            responses = output_queue.get(block=True, timeout=timeout * len(vec_chunk))
            for response, result in zip(responses, result_chunk):
                status, message, e2e_perf, device_perf = response[0], response[1], response[2], response[3]
                if status and MEASURE_DEVICE_PERF and device_perf is None:
                    result["status"] = TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF
                    result["message"] = message
                elif status and MEASURE_DEVICE_PERF:
                    result["status"] = TestStatus.PASS
                    result["message"] = message
                    result["device_perf"] = device_perf
                elif status:
                    result["status"] = TestStatus.PASS
                    result["message"] = message
                else:
                    if "DEVICE EXCEPTION" in message:
                        logger.error(
                            "DEVICE EXCEPTION: Device could not be initialized. The following assertion was thrown: "
                            + message,
                        )
                        logger.info("Skipping test suite because of device error, proceeding...")
                        return []
                    if "Out of Memory: Not enough space to allocate" in message:
                        result["status"] = TestStatus.FAIL_L1_OUT_OF_MEM
                    elif "Watcher" in message:
                        result["status"] = TestStatus.FAIL_WATCHER
                    else:
                        result["status"] = TestStatus.FAIL_ASSERT_EXCEPTION
                    result["exception"] = message
                result["e2e_perf"] = None
                result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                result["host"] = get_hostname()
                result["user"] = get_username()
        except Empty as e:
            logger.warning(f"TEST TIMED OUT, Killing child process {p.pid} and running tt-smi...")
            p.terminate()
            p = None
            result["status"], result["exception"] = TestStatus.FAIL_CRASH_HANG, "TEST TIMED OUT (CRASH / HANG)"
            result["e2e_perf"] = None
            try:
                tt_smi_util.run_tt_smi(ARCH)
            except:
                logger.warning("tt-smi failed... exiting")
                break

        for i in range(actual_chunk_size):
            suite_pbar.update()
        results.extend(result_chunk)

    if p is not None:
        p.join()

    suite_pbar.close()
    return results


def sanitize_inputs(test_vectors):
    info_field_names = ["sweep_name", "suite_name", "vector_id", "input_hash"]
    header_info = []
    for vector in test_vectors:
        header = dict()
        for field in info_field_names:
            header[field] = vector.pop(field)
        vector.pop("timestamp")
        vector.pop("tag")
        header_info.append(header)
    return header_info, test_vectors


def get_suite_vectors(client, vector_index, suite):
    response = client.search(
        index=vector_index,
        query={
            "bool": {
                "must": [
                    {"match": {"status": str(VectorStatus.CURRENT)}},
                    {"match": {"suite_name.keyword": suite}},
                    {"match": {"tag.keyword": SWEEPS_TAG}},
                ]
            }
        },
        size=10000,
    )
    test_ids = [hit["_id"] for hit in response["hits"]["hits"]]
    test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]
    for i in range(len(test_ids)):
        test_vectors[i]["vector_id"] = test_ids[i]
    header_info, test_vectors = sanitize_inputs(test_vectors)
    return header_info, test_vectors


def export_test_results_json(header_info, results, result_file=None):
    if len(results) == 0:
        return
    if not result_file:
        module_name = header_info[0]["sweep_name"]
        EXPORT_DIR_PATH = pathlib.Path(__file__).parent / "results_export"
        EXPORT_PATH = EXPORT_DIR_PATH / str(module_name + ".json")
        if not EXPORT_DIR_PATH.exists():
            EXPORT_DIR_PATH.mkdir()
    else:
        EXPORT_PATH = pathlib.Path(result_file)

    curr_git_hash = git_hash()
    for result in results:
        result["git_hash"] = curr_git_hash

    new_data = []

    for i in range(len(results)):
        result = header_info[i]
        for elem in results[i].keys():
            if elem == "device_perf":
                result[elem] = results[i][elem]
                continue
            result[elem] = serialize(results[i][elem])
        new_data.append(result)

    if EXPORT_PATH.exists():
        with open(EXPORT_PATH, "r") as file:
            old_data = json.load(file)
        new_data = old_data + new_data
        with open(EXPORT_PATH, "w") as file:
            json.dump(new_data, file, indent=4)
    else:
        with open(EXPORT_PATH, "w") as file:
            json.dump(new_data, file, indent=4)


def run_sweeps(module_name, suite_name, vector_id):
    pbar_manager = enlighten.get_manager()

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if READ_FILE:
        with open(READ_FILE, "r") as file:
            data = json.load(file)
            for suite in data:
                if suite_name and suite_name != suite:
                    continue

                for input_hash in data[suite]:
                    data[suite][input_hash]["vector_id"] = input_hash
                vectors = [data[suite][input_hash] for input_hash in data[suite]]
                module_name = vectors[0]["sweep_name"]
                test_module = importlib.import_module("sweeps." + module_name)
                header_info, test_vectors = sanitize_inputs(vectors)
                logger.info(f"Executing tests for module {module_name}, suite {suite}")
                results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                logger.info(f"Completed tests for module {module_name}, suite {suite}.")
                logger.info(f"Tests Executed - {len(results)}")
                logger.info("Dumping results to JSON file.")
                export_test_results_json(header_info, results, RESULT_FILE)
        return

    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))

    if not module_name:
        for file in sorted(sweeps_path.glob("**/*.py")):
            sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
            test_module = importlib.import_module("sweeps." + sweep_name)
            vector_index = VECTOR_INDEX_PREFIX + sweep_name
            logger.info(f"Executing tests for module {sweep_name}...")
            try:
                if not suite_name:
                    response = client.search(
                        index=vector_index,
                        query={"match": {"tag.keyword": SWEEPS_TAG}},
                        aggregations={"suites": {"terms": {"field": "suite_name.keyword", "size": 10000}}},
                    )
                    suites = [suite["key"] for suite in response["aggregations"]["suites"]["buckets"]]
                else:
                    response = client.search(
                        index=vector_index,
                        query={
                            "bool": {
                                "must": [
                                    {"match": {"tag.keyword": SWEEPS_TAG}},
                                    {"match": {"suite_name.keyword": suite_name}},
                                ]
                            }
                        },
                        aggregations={"suites": {"terms": {"field": "suite_name.keyword", "size": 10000}}},
                    )
                    suites = [suite["key"] for suite in response["aggregations"]["suites"]["buckets"]]
                if len(suites) == 0:
                    if not suite_name:
                        logger.info(
                            f"No suites found for module {sweep_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suites of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again. Continuing..."
                        )
                    else:
                        logger.info(
                            f"No suite named {suite_name} found for module {sweep_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suite of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again. Continuing..."
                        )
                    continue

                module_pbar = pbar_manager.counter(total=len(suites), desc=f"Module: {sweep_name}", leave=False)
                for suite in suites:
                    logger.info(f"Executing tests for module {sweep_name}, suite {suite}.")
                    header_info, test_vectors = get_suite_vectors(client, vector_index, suite)
                    results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                    logger.info(f"Completed tests for module {sweep_name}, suite {suite}.")
                    logger.info(f"Tests Executed - {len(results)}")
                    export_test_results(header_info, results)
                    module_pbar.update()
                module_pbar.close()
            except NotFoundError as e:
                logger.info(f"No test vectors found for module {sweep_name}. Skipping...")
                continue
            except Exception as e:
                logger.error(e)
                continue

    else:
        try:
            test_module = importlib.import_module("sweeps." + module_name)
        except ModuleNotFoundError as e:
            logger.error(f"No module found with name {module_name}")
            exit(1)
        vector_index = VECTOR_INDEX_PREFIX + module_name

        if vector_id:
            test_vector = client.get(index=vector_index, id=vector_id)["_source"]
            test_vector["vector_id"] = vector_id
            header_info, test_vectors = sanitize_inputs([test_vector])
            results = execute_suite(test_module, test_vectors, pbar_manager, "Single Vector")
            export_test_results(header_info, results)
        else:
            try:
                if not suite_name:
                    response = client.search(
                        index=vector_index,
                        query={"match": {"tag.keyword": SWEEPS_TAG}},
                        aggregations={"suites": {"terms": {"field": "suite_name.keyword", "size": 10000}}},
                        size=10000,
                    )
                    suites = [suite["key"] for suite in response["aggregations"]["suites"]["buckets"]]
                    if len(suites) == 0:
                        logger.info(
                            f"No suites found for module {module_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suites of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again."
                        )
                        return

                    for suite in suites:
                        logger.info(f"Executing tests for module {module_name}, suite {suite}.")
                        header_info, test_vectors = get_suite_vectors(client, vector_index, suite)
                        results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                        logger.info(f"Completed tests for module {module_name}, suite {suite}.")
                        logger.info(f"Tests Executed - {len(results)}")
                        export_test_results(header_info, results)
                else:
                    logger.info(f"Executing tests for module {module_name}, suite {suite_name}.")
                    header_info, test_vectors = get_suite_vectors(client, vector_index, suite_name)
                    results = execute_suite(test_module, test_vectors, pbar_manager, suite_name)
                    logger.info(f"Completed tests for module {module_name}, suite {suite_name}.")
                    logger.info(f"Tests Executed - {len(results)}")
                    export_test_results(header_info, results)
            except Exception as e:
                logger.info(e)

    client.close()


# Export test output (msg), status, exception (if applicable), git hash, timestamp, test vector, test UUID?,
def export_test_results(header_info, results):
    if len(results) == 0:
        return
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    sweep_name = header_info[0]["sweep_name"]
    results_index = RESULT_INDEX_PREFIX + sweep_name

    curr_git_hash = git_hash()
    for result in results:
        result["git_hash"] = curr_git_hash

    for i in range(len(results)):
        result = header_info[i]
        for elem in results[i].keys():
            if elem == "device_perf":
                result[elem] = results[i][elem]
                continue
            result[elem] = serialize(results[i][elem])
        client.index(index=results_index, body=result)

    client.close()


def enable_watcher():
    logger.info("Enabling Watcher")
    os.environ["TT_METAL_WATCHER"] = "120"
    os.environ["TT_METAL_WATCHER_APPEND"] = "1"


def disable_watcher():
    logger.info("Disabling Watcher")
    os.environ.pop("TT_METAL_WATCHER")
    os.environ.pop("TT_METAL_WATCHER_APPEND")


def enable_profiler():
    logger.info("Enabling Device Profiler")
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
    os.environ["ENABLE_TRACY"] = "1"


def disable_profiler():
    logger.info("Disabling Device Profiler")
    os.environ.pop("TT_METAL_DEVICE_PROFILER")
    os.environ.pop("ENABLE_TRACY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Runner",
        description="Run test vector suites from generated vector database.",
    )

    parser.add_argument(
        "--elastic",
        required=False,
        default="corp",
        help="Elastic Connection String for the vector and results database. Available presets are ['corp', 'cloud']",
    )
    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted.")
    parser.add_argument("--suite-name", required=False, help="Suite of Test Vectors to run, or all tests if omitted.")
    parser.add_argument(
        "--vector-id", required=False, help="Specify vector id with a module name to run an individual test vector."
    )
    parser.add_argument(
        "--watcher", action="store_true", required=False, help="Add this flag to run sweeps with watcher enabled."
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        required=False,
        help="Add this flag to measure e2e perf, for op tests with performance markers.",
    )

    parser.add_argument(
        "--device-perf",
        required=False,
        action="store_true",
        help="Measure device perf using device profiler. REQUIRES PROFILER BUILD!",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=False,
        help="Add this flag to perform a dry run.",
    )
    parser.add_argument(
        "--tag",
        required=False,
        default=os.getenv("USER"),
        help="Custom tag for the vectors you are running. This is to keep copies seperate from other people's test vectors. By default, this will be your username. You are able to specify a tag when generating tests using the generator.",
    )
    parser.add_argument("--read-file", required=False, help="Read and execute test vectors from a specified file path.")
    parser.add_argument("--result-file")

    args = parser.parse_args(sys.argv[1:])
    if not args.module_name and args.vector_id:
        parser.print_help()
        logger.error("Module name is required if vector id is specified.")
        exit(1)

    if not args.read_file:
        from elasticsearch import Elasticsearch, NotFoundError
        from framework.elastic_config import *

        global ELASTIC_CONNECTION_STRING
        ELASTIC_CONNECTION_STRING = get_elastic_url(args.elastic)
    else:
        if not args.module_name:
            logger.error("You must specify a module with a local file.")
            exit(1)
        global READ_FILE
        READ_FILE = args.read_file

    global RESULT_FILE
    RESULT_FILE = args.result_file

    global MEASURE_PERF
    MEASURE_PERF = args.perf

    global MEASURE_DEVICE_PERF
    MEASURE_DEVICE_PERF = args.device_perf

    global DRY_RUN
    DRY_RUN = args.dry_run

    global SWEEPS_TAG
    SWEEPS_TAG = args.tag

    logger.info(f"Running current sweeps with tag: {SWEEPS_TAG}.")

    if args.watcher:
        enable_watcher()

    if MEASURE_DEVICE_PERF:
        enable_profiler()

    from ttnn import *
    from framework.serialize import *
    from framework.device_fixtures import default_device
    from framework.sweeps_logger import sweeps_logger as logger

    run_sweeps(args.module_name, args.suite_name, args.vector_id)

    if args.watcher:
        disable_watcher()

    if MEASURE_DEVICE_PERF:
        disable_profiler()
