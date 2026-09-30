#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Utility
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_command(command, check=False, capture=True):
    return subprocess.run(
        command,
        check=check,
        capture_output=capture,
        text=True
    )


def command_exists(command):
    result = subprocess.run(
        ["which", command],
        capture_output=True,
        text=True
    )

    return result.returncode == 0


# ============================================================
# Checks
# ============================================================

def check_required_commands():
    required = [
        "iw",
        "ss",
        "iperf3",
        "sysctl"
    ]

    print("[CHECK] Required commands:")

    for command in required:

        if not command_exists(command):
            print(f"  [ERROR] {command} not found")
            return False

        print(f"  [OK] {command}")

    return True


def check_wifi(interface):
    result = run_command([
        "iw",
        "dev",
        interface,
        "link"
    ])

    if result.returncode != 0:
        print("[ERROR] Cannot read Wi-Fi link state")
        print(result.stderr)
        return False

    output = result.stdout

    if "Not connected" in output:
        print(
            f"[ERROR] Wi-Fi interface "
            f"{interface} is not connected"
        )
        return False

    print(
        f"[OK] Wi-Fi interface "
        f"{interface} is connected"
    )

    print()
    print("----- Wi-Fi link -----")
    print(output.strip())
    print("----------------------")
    print()

    return True


def check_route(server):
    print(f"[CHECK] Checking route to {server}...")

    result = run_command([
        "ip",
        "route",
        "get",
        server
    ])

    if result.returncode != 0:
        print("[ERROR] Cannot determine route")
        print(result.stderr)
        return False

    output = result.stdout.strip()

    print(output)

    # We expect traffic to go through the Wi-Fi interface.
    if "dev wlx" not in output:
        print(
            "[WARN] Route does not appear to use "
            "the expected Wi-Fi interface."
        )

    return True


def get_available_cc():
    result = run_command([
        "sysctl",
        "-n",
        "net.ipv4.tcp_available_congestion_control"
    ])

    if result.returncode != 0:
        return []

    return result.stdout.strip().split()


def get_current_cc():
    result = run_command([
        "sysctl",
        "-n",
        "net.ipv4.tcp_congestion_control"
    ])

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def check_congestion_control(cc, direction):
    available = get_available_cc()

    print(
        f"[TCP] Available congestion controls: "
        f"{' '.join(available)}"
    )

    if direction.lower() == "downlink":
        print()
        print("[WARN] DOWNLINK selected.")
        print("[WARN] TCP sender is the Windows iperf3 server.")
        print("[WARN] Linux congestion control does not control the downlink TCP sender.")
        print()
        return True

    if cc not in available:
        print(
            f"[ERROR] Congestion control '{cc}' "
            f"is not available."
        )
        print("[INFO] Available: " + ", ".join(available))
        return False

    return True


def set_congestion_control(cc):
    print(
        f"[TCP] Setting congestion control: {cc}"
    )

    result = subprocess.run(
        [
            "sudo",
            "sysctl",
            "-w",
            f"net.ipv4.tcp_congestion_control={cc}"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("[ERROR] Failed to set congestion control")
        print(result.stderr)
        return False

    current = get_current_cc()

    if current != cc:
        print(
            f"[ERROR] Verification failed. "
            f"Expected {cc}, got {current}"
        )
        return False

    print(
        f"[OK] Congestion control = {current}"
    )

    return True


# ============================================================
# Experiment directory
# ============================================================

def create_run_directory(results_dir):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        Path(results_dir)
        / f"run_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    return run_dir


# ============================================================
# Metadata
# ============================================================

def save_metadata(
    run_dir,
    config,
    interface,
    server,
    cc,
    direction,
    experiment_start
):

    metadata = {
        "experiment_start": experiment_start,

        "interface": interface,
        "server": server,

        "congestion_control": cc,
        "direction": direction,

        "distance_m": config.get(
            "distance_m",
            None
        ),

        "background_load_percent": config.get(
            "background_load_percent",
            0
        ),

        "repeat": config.get(
            "repeat",
            1
        ),

        "duration_sec": config["duration_sec"],

        "wifi_interval_sec":
            config["wifi_interval_sec"],

        "tcp_interval_sec":
            config["tcp_interval_sec"],

        "iperf_interval_sec":
            config["iperf_interval_sec"],

        "stabilization_sec":
            config["stabilization_sec"]
    }

    path = run_dir / "metadata.json"

    with open(path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    return path


def update_metadata(run_dir, **updates):

    path = run_dir / "metadata.json"

    with open(path, "r") as f:
        metadata = json.load(f)

    metadata.update(updates)

    with open(path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

# ============================================================
# Background UDP load
# ============================================================

def start_background_load(
    server,
    port,
    capacity_mbps,
    load_percent,
    duration,
    output_log,
    experiment_log
):
    if load_percent <= 0:
        print("[LOAD] Background load disabled")
        return None

    bitrate_mbps = (
        capacity_mbps * load_percent / 100.0
    )

    bitrate = f"{bitrate_mbps:.3f}M"

    command = [
        "iperf3",
        "-c",
        server,

        "-p",
        str(port),

        "-u",

        "-b",
        bitrate,

        "-t",
        str(duration),

        "-i",
        "1"
    ]

    print()
    print("[LOAD] Starting background UDP:")
    print(" ".join(command))
    print()

    log = open(output_log, "w")

    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True
    )

    process._experiment_log = log

    return process

# ============================================================
# Collectors
# ============================================================

def start_wifi_collector(
    interface,
    output,
    interval,
    duration,
    log_file
):

    command = [
        sys.executable,
        "wifi_collector.py",

        "--interface",
        interface,

        "--output",
        str(output),

        "--interval",
        str(interval),

        "--duration",
        str(duration)
    ]

    print("[START] Wi-Fi collector")

    log = open(log_file, "a")

    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True
    )

    process._experiment_log = log

    return process


def start_tcp_reader(
    server,
    output,
    interval,
    duration,
    log_file
):

    command = [
        sys.executable,
        "tcpinfo_reader.py",

        "--server",
        server,

        "--output",
        str(output),

        "--interval",
        str(interval),

        "--duration",
        str(duration)
    ]

    print("[START] TCP reader")

    log = open(log_file, "a")

    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True
    )

    process._experiment_log = log

    return process


# ============================================================
# Stop process
# ============================================================

def stop_process(process, name):

    if process is None:
        return

    if process.poll() is None:

        print(f"[STOP] {name}")

        try:

            process.terminate()
            process.wait(timeout=5)

        except subprocess.TimeoutExpired:

            print(
                f"[WARN] {name} did not terminate. "
                f"Killing..."
            )

            process.kill()
            process.wait()

    return_code = process.returncode

    print(
        f"[STOP] {name} exit code={return_code}"
    )

    log = getattr(
        process,
        "_experiment_log",
        None
    )

    if log is not None:
        log.close()

# ============================================================
# iperf3
# ============================================================

def run_iperf(
    server,
    duration,
    interval,
    direction,
    output_json,
    log_file
):

    command = [
        "iperf3",
        "-c",
        server,

        "-t",
        str(duration),

        "-i",
        str(interval),

        "--json"
    ]

    if direction.lower() == "downlink":

        command.append("-R")

    elif direction.lower() == "uplink":

        pass

    else:

        raise ValueError(
            "direction must be "
            "'downlink' or 'uplink'"
        )

    print()
    print("[IPERF] Starting:")
    print(" ".join(command))
    print()

    start_time = time.monotonic()

    with open(output_json, "w") as json_file, \
         open(log_file, "a") as log:

        process = subprocess.Popen(
            command,
            stdout=json_file,
            stderr=log
        )

        return_code = process.wait()

    elapsed = (
        time.monotonic()
        - start_time
    )

    print(
        f"[IPERF] Finished "
        f"(exit code={return_code}, "
        f"time={elapsed:.1f}s)"
    )

    return return_code


# ============================================================
# Main experiment
# ============================================================

def run_experiment(config):

    interface = config["interface"]
    server = config["server"]

    duration = config["duration_sec"]

    wifi_interval = config["wifi_interval_sec"]
    tcp_interval = config["tcp_interval_sec"]

    iperf_interval = config["iperf_interval_sec"]

    cc = config["congestion_control"]
    direction = config["direction"]

    results_dir = config["results_dir"]

    stabilization = config["stabilization_sec"]

    distance = config.get(
        "distance_m",
        None
    )

    background_load = config.get(
        "background_load_percent",
        0
    )
    
    channel_capacity = config.get(
        "channel_capacity_mbps",
        100
    )

    background_port = config.get(
        "background_port",
        5201
    )

    repeat = config.get(
        "repeat",
        1
    )

    experiment_start = utc_now()

    print()
    print("=" * 60)
    print("Wi-Fi / TCP EXPERIMENT")
    print("=" * 60)

    print(f"Interface       : {interface}")
    print(f"Server          : {server}")
    print(f"CC              : {cc}")
    print(f"Direction       : {direction}")
    print(f"Distance        : {distance} m")
    print(f"Background load : {background_load}%")
    print(
    f"Channel capacity: {channel_capacity} Mbps")

    if background_load > 0:
        print(
            f"UDP load        : "
            f"{channel_capacity * background_load / 100:.3f} Mbps"
        )

    print(
        f"Background port : {background_port}"
    )
    print(f"Repeat          : {repeat}")
    print(f"Duration        : {duration} sec")

    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    if not check_required_commands():
        return False

    if not check_wifi(interface):
        return False

    if not check_route(server):
        return False

    if not check_congestion_control(
        cc,
        direction
    ):
        return False

    # --------------------------------------------------------
    # Set CC
    # --------------------------------------------------------

    if direction.lower() == "uplink":
        if not set_congestion_control(cc):
            return False
    else:
        print()
        print("[TCP] DOWNLINK: Linux TCP congestion control will not be changed.")
        print(f"[TCP] Requested CC: {cc}")
        print("[TCP] Actual TCP sender: Windows")
        print()

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    run_dir = create_run_directory(
        results_dir
    )

    print(
        "[RUN] Results directory:"
    )

    print(
        f"      {run_dir}"
    )

    wifi_csv = (
        run_dir / "wifi.csv"
    )

    tcp_csv = (
        run_dir / "tcp.csv"
    )

    iperf_json = (
        run_dir / "iperf.json"
    )

    experiment_log = (
        run_dir / "experiment.log"
    )
    
    background_log = (
    run_dir / "background_iperf.log"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    save_metadata(
        run_dir,
        config,
        interface,
        server,
        cc,
        direction,
        experiment_start
    )

    # --------------------------------------------------------
    # Collectors
    # --------------------------------------------------------

    wifi_process = None
    tcp_process = None
    background_process = None

    success = False
    iperf_exit_code = None

    try:

        wifi_process = start_wifi_collector(
            interface,
            wifi_csv,
            wifi_interval,
            duration + 10,
            experiment_log
        )

        tcp_process = start_tcp_reader(
            server,
            tcp_csv,
            tcp_interval,
            duration + 10,
            experiment_log
        )

        # ----------------------------------------------------
        # Background UDP load
        # ----------------------------------------------------

        background_process = start_background_load(
            server=server,
            port=background_port,
            capacity_mbps=channel_capacity,
            load_percent=background_load,
            duration=duration + 10,
            output_log=background_log,
            experiment_log=experiment_log
        )

        print(
            "[WAIT] Starting collectors and "
            "background traffic..."
        )

        # Give collectors and UDP traffic time to start.
        time.sleep(2)

        # ----------------------------------------------------
        # Main TCP iperf3
        # ----------------------------------------------------

        iperf_exit_code = run_iperf(
            server,
            duration,
            iperf_interval,
            direction,
            iperf_json,
            experiment_log
        )

        if iperf_exit_code != 0:

            print(
                "[ERROR] iperf3 finished "
                f"with exit code "
                f"{iperf_exit_code}"
            )

        else:

            success = True
            
        # ----------------------------------------------------
        # Stop background UDP load
        # ----------------------------------------------------

        stop_process(
            background_process,
            "Background UDP load"
        )

        background_process = None

        # ----------------------------------------------------
        # Stop collectors
        # ----------------------------------------------------

        stop_process(
            wifi_process,
            "Wi-Fi collector"
        )

        stop_process(
            tcp_process,
            "TCP reader"
        )

        wifi_process = None
        tcp_process = None

    except KeyboardInterrupt:

        print()
        print(
            "[WARN] Experiment interrupted "
            "by user"
        )

        success = False

    finally:

        stop_process(
            background_process,
            "Background UDP load"
        )

        stop_process(
            wifi_process,
            "Wi-Fi collector"
        )

        stop_process(
            tcp_process,
            "TCP reader"
        )

    # --------------------------------------------------------
    # Metadata finalization
    # --------------------------------------------------------

    experiment_end = utc_now()

    update_metadata(
        run_dir,
        experiment_end=experiment_end,
        iperf3_exit_code=iperf_exit_code,
        success=success
    )

    # --------------------------------------------------------
    # Stabilization
    # --------------------------------------------------------

    print()

    print(
        f"[WAIT] Stabilization: "
        f"{stabilization} sec"
    )

    time.sleep(stabilization)

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 60)

    if success:
        print("EXPERIMENT FINISHED SUCCESSFULLY")
    else:
        print("EXPERIMENT FAILED")

    print("=" * 60)

    print()
    print("Files:")

    for file in sorted(run_dir.iterdir()):

        print(
            f"  {file.name:20} "
            f"{file.stat().st_size} bytes"
        )

    print()
    print(
        f"Results: {run_dir}"
    )
    print()

    return success


# ============================================================
# Entry point
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run Wi-Fi/TCP experiment"
    )

    parser.add_argument(
        "--config",
        default="config.json"
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    if not config_path.exists():

        print(
            "[ERROR] Config file not found: "
            f"{config_path}"
        )

        sys.exit(1)

    try:

        with open(config_path, "r") as f:
            config = json.load(f)

    except json.JSONDecodeError as e:

        print(
            "[ERROR] Invalid JSON config"
        )

        print(e)

        sys.exit(1)

    required_fields = [
        "interface",
        "server",
        "duration_sec",
        "wifi_interval_sec",
        "tcp_interval_sec",
        "iperf_interval_sec",
        "congestion_control",
        "direction",
        "results_dir",
        "stabilization_sec"
    ]

    for field in required_fields:

        if field not in config:

            print(
                "[ERROR] Missing config field: "
                f"{field}"
            )

            sys.exit(1)

    success = run_experiment(
        config
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
