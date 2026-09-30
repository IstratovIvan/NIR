from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# НАСТРОЙКИ
# ============================================================

RESULTS_DIR = {
    "uplink": Path("results"),
    "downlink": Path("results_downlink"),
}
OUTPUT_DIR = Path("analysis")

PAIRS = [
    ("rssi_dbm", "rtt_ms"),
    ("rssi_dbm", "throughput_mbps"),
    ("mcs", "throughput_mbps"),
]

RSSI_BINS = [-100, -60, -55, -50, -45, 0]

RSSI_LABELS = [
    "< -60 dBm",
    "-60 ... -55 dBm",
    "-55 ... -50 dBm",
    "-50 ... -45 dBm",
    "> -45 dBm",
]

def make_rssi_boxplots(df):
    configs = [
        ("uplink", "cubic", "linux"),
        ("uplink", "bbr", "linux"),
        ("downlink", "cubic", "windows"),
        ("downlink", "bbr", "windows"),
    ]

    for direction, algorithm, cc_host in configs:

        subset = df[
            (df["direction"] == direction)
            & (df["algorithm"] == algorithm)
            & (df["cc_host"] == cc_host)
        ].copy()

        if subset.empty:
            print(
                f"[BOXPLOT] Нет данных: "
                f"{direction} / {cc_host} / {algorithm}"
            )
            continue

        subset["rssi_bin"] = pd.cut(
            subset["rssi_dbm"],
            bins=RSSI_BINS,
            labels=RSSI_LABELS,
            include_lowest=True
        )

        # -------------------------
        # RTT
        # -------------------------

        rtt_data = []
        rtt_labels = []

        for label in RSSI_LABELS:
            values = subset.loc[
                subset["rssi_bin"] == label,
                "rtt_ms"
            ].dropna()

            if len(values) > 0:
                rtt_data.append(values)
                rtt_labels.append(label)

        if rtt_data:
            plt.figure(figsize=(10, 6))

            plt.boxplot(
                rtt_data,
                tick_labels=rtt_labels
            )

            plt.xlabel("RSSI")
            plt.ylabel("RTT, ms")

            plt.title(
                f"RTT distribution by RSSI\n"
                f"{direction} / {cc_host} / {algorithm}"
            )

            plt.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            filename = (
                f"boxplot_rtt_"
                f"{direction}_"
                f"{cc_host}_"
                f"{algorithm}.png"
            )

            plt.savefig(
                OUTPUT_DIR / filename,
                dpi=200
            )

            plt.close()

            print(
                f"[BOXPLOT] Saved: "
                f"{OUTPUT_DIR / filename}"
            )

        # -------------------------
        # Throughput
        # -------------------------

        throughput_data = []
        throughput_labels = []

        for label in RSSI_LABELS:
            values = subset.loc[
                subset["rssi_bin"] == label,
                "throughput_mbps"
            ].dropna()

            if len(values) > 0:
                throughput_data.append(values)
                throughput_labels.append(label)

        if throughput_data:
            plt.figure(figsize=(10, 6))

            plt.boxplot(
                throughput_data,
                tick_labels=throughput_labels
            )

            plt.xlabel("RSSI")
            plt.ylabel("Throughput, Mbps")

            plt.title(
                f"Throughput distribution by RSSI\n"
                f"{direction} / {cc_host} / {algorithm}"
            )

            plt.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            filename = (
                f"boxplot_throughput_"
                f"{direction}_"
                f"{cc_host}_"
                f"{algorithm}.png"
            )

            plt.savefig(
                OUTPUT_DIR / filename,
                dpi=200
            )

            plt.close()

            print(
                f"[BOXPLOT] Saved: "
                f"{OUTPUT_DIR / filename}"
            )

def analyze_cwnd_stability(df):
    configs = [
        ("uplink", "cubic", "linux"),
        ("uplink", "bbr", "linux"),
    ]

    if "cwnd" not in df.columns:
        print(
            "[CWND] Ошибка: "
            "столбец tcpi_snd_cwnd отсутствует."
        )
        return pd.DataFrame()

    results = []

    for direction, algorithm, cc_host in configs:

        subset = df[
            (df["direction"] == direction)
            & (df["algorithm"] == algorithm)
            & (df["cc_host"] == cc_host)
        ].copy()

        if subset.empty:
            print(
                f"[CWND] Нет данных: "
                f"{direction} / {cc_host} / {algorithm}"
            )
            continue

        subset["rssi_bin"] = pd.cut(
            subset["rssi_dbm"],
            bins=RSSI_BINS,
            labels=RSSI_LABELS,
            include_lowest=True
        )

        for rssi_label in RSSI_LABELS:

            values = subset.loc[
                subset["rssi_bin"] == rssi_label,
                "cwnd"
            ].dropna()

            if len(values) < 2:
                continue

            mean_cwnd = values.mean()
            std_cwnd = values.std()

            if mean_cwnd == 0:
                cv = np.nan
            else:
                cv = std_cwnd / mean_cwnd

            results.append({
                "direction": direction,
                "algorithm": algorithm,
                "cc_host": cc_host,
                "rssi_range": rssi_label,
                "n": len(values),
                "cwnd_mean": mean_cwnd,
                "cwnd_std": std_cwnd,
                "cwnd_cv": cv,
            })

    result_df = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR /
        "cwnd_stability_by_rssi.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"[CWND] Saved: {output_file}"
    )

    return result_df

def diagnose_runs():
    print("\n" + "=" * 90)
    print("ДИАГНОСТИКА ЗАПУСКОВ")
    print("=" * 90)

    runs = find_runs()

    print(f"\nВсего найдено запусков: {len(runs)}")

    diagnostics = []

    for run in runs:
        run_dir = run["run_dir"]
        status = "OK"
        reason = ""

        # Проверяем metadata
        try:
            metadata = run["metadata"]

            if not metadata.get("experiment_start"):
                status = "ERROR"
                reason = "нет experiment_start"

        except Exception as e:
            status = "ERROR"
            reason = f"metadata: {e}"

        # Проверяем wifi.csv
        if status == "OK":
            try:
                wifi = load_wifi(run["wifi_file"])

                if wifi.empty:
                    status = "ERROR"
                    reason = "wifi.csv пустой"

            except Exception as e:
                status = "ERROR"
                reason = f"wifi.csv: {e}"

        # Проверяем tcp.csv
        if status == "OK":
            try:
                tcp = load_tcp(run["tcp_file"])

                if tcp.empty:
                    status = "ERROR"
                    reason = "tcp.csv пустой"

            except Exception as e:
                status = "ERROR"
                reason = f"tcp.csv: {e}"

        # Проверяем iperf.json
        if status == "OK":
            try:
                throughput = load_throughput(
                    run["iperf_file"],
                    run["metadata"]["experiment_start"]
                )

                if throughput.empty:
                    status = "ERROR"
                    reason = "iperf.json не дал данных"

            except json.JSONDecodeError:
                status = "ERROR"
                reason = "iperf.json содержит некорректный JSON"

            except Exception as e:
                status = "ERROR"
                reason = f"iperf.json: {e}"

        diagnostics.append({
            "run": run_dir.name,
            "algorithm": run["algorithm"],
            "distance": run["distance_m"],
            "udp": run["background_load_percent"],
            "status": status,
            "reason": reason
        })

    # ---------------------------------------------------------
    # Вывод всех запусков
    # ---------------------------------------------------------

    print("\n" + "-" * 90)
    print("ВСЕ ЗАПУСКИ")
    print("-" * 90)

    for d in diagnostics:
        print(
            f"{d['algorithm']:>5} | "
            f"{d['distance']:>1} м | "
            f"{d['udp']:>2}% UDP | "
            f"{d['status']:<5} | "
            f"{d['run']}"
            + (f" | {d['reason']}" if d["reason"] else "")
        )

    # ---------------------------------------------------------
    # Сводка по условиям
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("СВОДКА ПО УСЛОВИЯМ")
    print("=" * 90)

    condition_stats = {}

    for d in diagnostics:
        key = (
            d["algorithm"],
            d["distance"],
            d["udp"]
        )

        if key not in condition_stats:
            condition_stats[key] = {
                "total": 0,
                "valid": 0,
                "invalid": 0
            }

        condition_stats[key]["total"] += 1

        if d["status"] == "OK":
            condition_stats[key]["valid"] += 1
        else:
            condition_stats[key]["invalid"] += 1

    print(
        f"{'Algorithm':<10}"
        f"{'Distance':<10}"
        f"{'UDP':<8}"
        f"{'Всего':<8}"
        f"{'OK':<8}"
        f"{'Ошибок':<8}"
    )

    print("-" * 60)

    for key in sorted(condition_stats):
        algorithm, distance, udp = key
        stats = condition_stats[key]

        print(
            f"{algorithm:<10}"
            f"{distance:<10}"
            f"{udp:<8}"
            f"{stats['total']:<8}"
            f"{stats['valid']:<8}"
            f"{stats['invalid']:<8}"
        )

    print("\n" + "=" * 90)

# ============================================================
# ЗАГРУЗКА WIFI
# ============================================================

def load_wifi(path):
    wifi = pd.read_csv(path)

    wifi["timestamp"] = pd.to_datetime(
        wifi["timestamp"],
        utc=True
    )

    wifi["time"] = wifi["timestamp"].dt.floor("s")

    return wifi

# ============================================================
# ЗАГРУЗКА TCP
# ============================================================

def load_tcp(path):
    tcp = pd.read_csv(path)

    tcp["timestamp"] = pd.to_datetime(
        tcp["timestamp"],
        utc=True
    )

    tcp["time"] = tcp["timestamp"].dt.floor("s")

    return tcp

# ============================================================
# ЗАГРУЗКА IPERF
# ============================================================

def load_throughput(path, experiment_start):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    start_time = pd.to_datetime(
        experiment_start,
        utc=True
    )

    for interval in data.get("intervals", []):
        interval_data = interval.get("sum", {})

        if "bits_per_second" not in interval_data:
            continue

        relative_time = interval_data.get("start", 0)

        timestamp = (
            start_time +
            pd.to_timedelta(relative_time, unit="s")
        )

        throughput = (
            interval_data["bits_per_second"] / 1_000_000
        )

        rows.append({
            "time": timestamp.floor("s"),
            "throughput_mbps": throughput
        })

    return pd.DataFrame(rows)


# ============================================================
# СИНХРОНИЗАЦИЯ
# ============================================================

def synchronize(wifi, tcp, throughput):
    wifi_sec = (
        wifi
        .groupby("time", as_index=False)
        .mean(numeric_only=True)
    )

    tcp_sec = (
        tcp
        .groupby("time", as_index=False)
        .mean(numeric_only=True)
    )

    merged = pd.merge(
        wifi_sec,
        tcp_sec,
        on="time",
        how="inner",
        suffixes=("_wifi", "_tcp")
    )

    if not throughput.empty:
        throughput_sec = (
            throughput
            .groupby("time", as_index=False)
            .mean()
        )

        merged = pd.merge(
            merged,
            throughput_sec,
            on="time",
            how="left"
        )

    return merged

def preprocess_data(df):
    df = df.copy()

    # Непрерывные метрики:
    # для них применяем правило 3σ от медианы
    continuous_metrics = [
        "rssi_dbm",
        "snr_db",
        "tx_bitrate_mbps",
        "rx_bitrate_mbps",
        "signal_avg",
        "expected_throughput",
        "rtt_ms",
        "rttvar_ms",
        "cwnd",
        "throughput_mbps",
    ]

    # Дискретные / категориальные метрики:
    # к ним 3σ и линейную интерполяцию не применяем
    discrete_metrics = [
        "frequency_mhz",
        "channel",
        "mcs",
    ]

    # Счётчики:
    # не интерполируем линейно и не удаляем как статистические выбросы
    counter_metrics = [
        "tx_retries",
        "tx_failed",
        "retrans",
        "lost",
        "total_retrans",
    ]

    report = []

    # Обрабатываем каждый эксперимент отдельно,
    # чтобы значения одного run никогда не влияли на другой.
    for run_id, run_df in df.groupby("run", sort=False):

        run_df = run_df.copy()
        run_df = run_df.sort_values("time")

        # ---------------------------------------------------------
        # 1. Непрерывные метрики
        # ---------------------------------------------------------

        for metric in continuous_metrics:

            if metric not in run_df.columns:
                continue

            initial_points = len(run_df)

            series = run_df[metric].copy()

            missing_before = series.isna().sum()

            valid = series.dropna()

            outliers_removed = 0

            if len(valid) >= 3:
                median = valid.median()
                std = valid.std()

                if std > 0:
                    outlier_mask = (
                        (series - median).abs() > 3 * std
                    )

                    outlier_mask &= series.notna()

                    outliers_removed = int(outlier_mask.sum())

                    series.loc[outlier_mask] = np.nan

            missing_after_outliers = series.isna().sum()

            # Интерполируем только короткие внутренние пропуски.
            #
            # limit_area="inside" означает:
            # - не трогаем начало/конец ряда;
            # - используем только реальные соседние значения.
            series = (
                series
                .interpolate(
                    method="linear",
                    limit=3,
                    limit_area="inside"
                )
            )

            interpolated = (
                missing_after_outliers
                - series.isna().sum()
            )

            remaining_missing = series.isna().sum()

            df.loc[run_df.index, metric] = series

            report.append({
                "run": run_id,
                "metric": metric,
                "type": "continuous",
                "initial_points": initial_points,
                "missing_before": int(missing_before),
                "outliers_removed": int(outliers_removed),
                "missing_after_outliers": int(missing_after_outliers),
                "interpolated": int(interpolated),
                "remaining_missing": int(remaining_missing),
                "removed_percent": (
                    outliers_removed / initial_points * 100
                ),
                "interpolated_percent": (
                    interpolated / initial_points * 100
                ),
            })

        # ---------------------------------------------------------
        # 2. Дискретные метрики
        # ---------------------------------------------------------

        for metric in discrete_metrics:

            if metric not in run_df.columns:
                continue

            initial_points = len(run_df)

            series = run_df[metric].copy()

            missing_before = series.isna().sum()

            # Для параметров Wi-Fi конфигурации используем
            # ближайшее известное значение внутри конкретного run.
            #
            # Например:
            # 5220, 5220, NaN, 5220
            #
            # превращается в:
            # 5220, 5220, 5220, 5220
            series = series.ffill().bfill()

            interpolated = (
                missing_before
                - series.isna().sum()
            )

            remaining_missing = series.isna().sum()

            df.loc[run_df.index, metric] = series

            report.append({
                "run": run_id,
                "metric": metric,
                "type": "discrete",
                "initial_points": initial_points,
                "missing_before": int(missing_before),
                "outliers_removed": 0,
                "missing_after_outliers": int(missing_before),
                "interpolated": int(interpolated),
                "remaining_missing": int(remaining_missing),
                "removed_percent": 0.0,
                "interpolated_percent": (
                    interpolated / initial_points * 100
                ),
            })

        # ---------------------------------------------------------
        # 3. Счётчики
        # ---------------------------------------------------------

        for metric in counter_metrics:

            if metric not in run_df.columns:
                continue

            initial_points = len(run_df)

            series = run_df[metric].copy()

            missing_before = series.isna().sum()

            # Счётчики не интерполируем.
            #
            # Например, если TCP_INFO не дал значение:
            #
            # 0, 0, NaN, NaN, 2
            #
            # мы НЕ превращаем это в:
            #
            # 0, 0, 0.67, 1.33, 2
            #
            # потому что это создаёт искусственные события.
            remaining_missing = missing_before

            df.loc[run_df.index, metric] = series

            report.append({
                "run": run_id,
                "metric": metric,
                "type": "counter",
                "initial_points": initial_points,
                "missing_before": int(missing_before),
                "outliers_removed": 0,
                "missing_after_outliers": int(missing_before),
                "interpolated": 0,
                "remaining_missing": int(remaining_missing),
                "removed_percent": 0.0,
                "interpolated_percent": 0.0,
            })

    # -------------------------------------------------------------
    # 4. Min-Max normalization
    # -------------------------------------------------------------

    normalization_metrics = (
        continuous_metrics +
        discrete_metrics +
        counter_metrics
    )

    normalization_report = []

    for metric in normalization_metrics:

        if metric not in df.columns:
            continue

        valid = df[metric].dropna()

        if valid.empty:
            continue

        min_value = valid.min()
        max_value = valid.max()

        if max_value == min_value:
            print(
                f"[NORMALIZATION] {metric}: "
                f"константный признак, нормализация пропущена"
            )
            continue

        df[f"{metric}_norm"] = (
                (df[metric] - min_value)
                / (max_value - min_value)
        )

        normalization_report.append({
            "metric": metric,
            "min_value": min_value,
            "max_value": max_value,
        })

    # -------------------------------------------------------------
    # 5. Сохраняем отчёты
    # -------------------------------------------------------------

    report_df = pd.DataFrame(report)

    normalization_df = pd.DataFrame(normalization_report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(
        OUTPUT_DIR / "preprocessing_report.csv",
        index=False
    )

    normalization_df.to_csv(
        OUTPUT_DIR / "normalization_report.csv",
        index=False
    )

    print(
        f"[PREPROCESSING] "
        f"Отчёт сохранён: "
        f"{OUTPUT_DIR / 'preprocessing_report.csv'}"
    )

    print(
        f"[NORMALIZATION] "
        f"Отчёт сохранён: "
        f"{OUTPUT_DIR / 'normalization_report.csv'}"
    )

    # -------------------------------------------------------------
    # Сводный отчёт по предобработке
    # -------------------------------------------------------------

    preprocessing_summary = (
        report_df
        .groupby(["metric", "type"], as_index=False)
        .agg(
            total_points=("initial_points", "sum"),
            missing_before=("missing_before", "sum"),
            outliers_removed=("outliers_removed", "sum"),
            interpolated=("interpolated", "sum"),
            remaining_missing=("remaining_missing", "sum"),
        )
    )

    total_points = preprocessing_summary["total_points"].sum()
    total_removed = preprocessing_summary["outliers_removed"].sum()
    total_interpolated = preprocessing_summary["interpolated"].sum()
    total_remaining_missing = preprocessing_summary["remaining_missing"].sum()

    print("\n[PREPROCESSING SUMMARY]")
    print(f"Всего значений метрик:    {total_points}")
    print(
        f"Удалено выбросов:         "
        f"{total_removed} "
        f"({total_removed / total_points * 100:.2f}%)"
    )
    print(
        f"Интерполировано:          "
        f"{total_interpolated} "
        f"({total_interpolated / total_points * 100:.2f}%)"
    )
    print(
        f"Осталось пропусков:       "
        f"{total_remaining_missing} "
        f"({total_remaining_missing / total_points * 100:.2f}%)"
    )

    preprocessing_summary["removed_percent"] = (
            preprocessing_summary["outliers_removed"]
            / preprocessing_summary["total_points"]
            * 100
    )

    preprocessing_summary["interpolated_percent"] = (
            preprocessing_summary["interpolated"]
            / preprocessing_summary["total_points"]
            * 100
    )

    preprocessing_summary["remaining_missing_percent"] = (
            preprocessing_summary["remaining_missing"]
            / preprocessing_summary["total_points"]
            * 100
    )

    preprocessing_summary.to_csv(
        OUTPUT_DIR / "preprocessing_summary.csv",
        index=False
    )

    print(
        f"[PREPROCESSING] "
        f"Сводный отчёт сохранён: "
        f"{OUTPUT_DIR / 'preprocessing_summary.csv'}"
    )

    return df, report_df


# ============================================================
# SPEARMAN
# ============================================================

def calculate_spearman(df, x, y):
    if x not in df.columns or y not in df.columns:
        return np.nan, np.nan, 0

    data = df[[x, y]].dropna()

    n = len(data)

    if n < 3:
        return np.nan, np.nan, n

    # Если одна из переменных постоянная,
    # корреляция не определена.
    if data[x].nunique() < 2 or data[y].nunique() < 2:
        return np.nan, np.nan, n

    rho, p_value = spearmanr(
        data[x],
        data[y]
    )

    return rho, p_value, n


# ============================================================
# СИЛА КОРРЕЛЯЦИИ
# ============================================================

def correlation_strength(rho):
    if pd.isna(rho):
        return "undefined"

    abs_rho = abs(rho)

    if abs_rho >= 0.7:
        return "strong"

    if abs_rho >= 0.5:
        return "moderate"

    return "weak"


# ============================================================
# АНАЛИЗ ОДНОГО ЗАПУСКА
# ============================================================

def analyze_run(run):
    run_dir = run["run_dir"]
    metadata = run["metadata"]

    wifi_path = run["wifi_file"]
    tcp_path = run["tcp_file"]
    iperf_path = run["iperf_file"]

    wifi = load_wifi(wifi_path)
    tcp = load_tcp(tcp_path)

    throughput = load_throughput(
        iperf_path,
        metadata["experiment_start"]
    )

    df = synchronize(
        wifi,
        tcp,
        throughput
    )

    results = []

    for wifi_metric, transport_metric in PAIRS:

        rho, p_value, n = calculate_spearman(
            df,
            wifi_metric,
            transport_metric
        )

        results.append({
            "run": str(run_dir),

            "direction": run["direction"],
            "algorithm": run["algorithm"],
            "cc_host": run["cc_host"],

            "distance_m": run["distance_m"],

            "background_load_percent": run[
                "background_load_percent"
            ],

            "band": run["band"],

            "repeat": run["repeat"],

            "wifi_metric": wifi_metric,
            "transport_metric": transport_metric,
            "rho": rho,
            "p_value": p_value,
            "n": n,
            "strength": correlation_strength(rho)
        })

    return results

# ============================================================
# ПОИСК ЗАПУСКОВ
# ============================================================

import re


def parse_condition_from_path(run_dir):
    """
    Из пути вида:

    results/run_1m/run_1m_80udp_bbr_5GHz/run_20260924_185004

    извлекает:
        distance = 1
        udp = 80
        algorithm = bbr
        band = 5GHz
    """

    condition_dir = run_dir.parent.name

    pattern = r"run_(\d+)m_(\d+)udp_(cubic|bbr)_(\d+)GHz"
    match = re.match(pattern, condition_dir, re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Не удалось определить условие из имени папки: {condition_dir}"
        )

    distance = int(match.group(1))
    udp = int(match.group(2))
    algorithm = match.group(3).lower()
    band = f"{match.group(4)}GHz"

    return {
        "distance_m": distance,
        "background_load_percent": udp,
        "algorithm": algorithm,
        "band": band,
    }


def find_runs():
    runs = []

    for direction, results_dir in RESULTS_DIR.items():

        if not results_dir.exists():
            print(
                f"\nПРЕДУПРЕЖДЕНИЕ: "
                f"директория не найдена: {results_dir}"
            )
            continue

        for run_dir in results_dir.rglob("*"):

            if not run_dir.is_dir():
                continue

            metadata_file = run_dir / "metadata.json"
            wifi_file = run_dir / "wifi.csv"
            tcp_file = run_dir / "tcp.csv"
            iperf_file = run_dir / "iperf.json"

            if not (
                metadata_file.exists()
                and wifi_file.exists()
                and tcp_file.exists()
                and iperf_file.exists()
            ):
                continue

            try:
                with open(
                    metadata_file,
                    "r",
                    encoding="utf-8"
                ) as f:
                    metadata = json.load(f)

                condition = parse_condition_from_path(
                    run_dir
                )

                metadata_algorithm = metadata.get(
                    "congestion_control"
                )
                metadata_distance = metadata.get(
                    "distance_m"
                )
                metadata_udp = metadata.get(
                    "background_load_percent"
                )

                if (
                    metadata_algorithm != condition["algorithm"]
                    or metadata_distance != condition["distance_m"]
                    or metadata_udp != condition[
                        "background_load_percent"
                    ]
                ):
                    print(
                        "\nПРЕДУПРЕЖДЕНИЕ: "
                        "metadata не совпадает "
                        "с именем папки:"
                    )

                    print(f"  Папка: {run_dir}")

                    print(
                        f"  Из папки: "
                        f"{condition['algorithm']} | "
                        f"{condition['distance_m']} м | "
                        f"{condition['background_load_percent']}% UDP"
                    )

                    print(
                        f"  Metadata: "
                        f"{metadata_algorithm} | "
                        f"{metadata_distance} м | "
                        f"{metadata_udp}% UDP"
                    )

                # -------------------------------------------------
                # Реальный алгоритм
                # -------------------------------------------------

                if direction == "downlink":
                    actual_algorithm = condition["algorithm"]
                    cc_host = "windows"
                else:
                    actual_algorithm = condition["algorithm"]
                    cc_host = "linux"

                runs.append({
                    "run_dir": run_dir,
                    "metadata": metadata,

                    "direction": direction,

                    "algorithm": actual_algorithm,

                    "cc_host": cc_host,

                    "distance_m": condition[
                        "distance_m"
                    ],

                    "background_load_percent": condition[
                        "background_load_percent"
                    ],

                    "band": condition["band"],

                    "repeat": metadata.get(
                        "repeat",
                        1
                    ),

                    "wifi_file": wifi_file,
                    "tcp_file": tcp_file,
                    "iperf_file": iperf_file,
                })

            except Exception as e:

                print(
                    f"\nОШИБКА: {run_dir}"
                )

                print(e)

    # ---------------------------------------------------------
    # Автоматически назначаем повторы
    # ---------------------------------------------------------

    groups = {}

    for run in runs:

        key = (
            run["direction"],
            run["algorithm"],
            run["distance_m"],
            run["background_load_percent"],
            run["band"],
        )

        groups.setdefault(
            key,
            []
        ).append(run)

    for group in groups.values():

        group.sort(
            key=lambda x: x["run_dir"].name
        )

        for repeat, run in enumerate(
            group,
            start=1
        ):
            run["repeat"] = repeat

    return runs

# ============================================================
# ОБЩАЯ СТАТИСТИКА
# ============================================================

def make_summary(df):
    summary = (
        df
        .groupby(
            [
                "direction",
                "algorithm",
                "cc_host",
                "wifi_metric",
                "transport_metric"
            ],
            as_index=False
        )
        .agg(
            rho_mean=("rho", "mean"),
            rho_std=("rho", "std"),
            p_value_mean=("p_value", "mean"),
            n_mean=("n", "mean")
        )
    )

    return summary


# ============================================================
# СТАТИСТИКА ПО УСЛОВИЯМ
# ============================================================

def make_condition_summary(results):
    df = pd.DataFrame(results)

    if df.empty:
        return df

    grouped = (
        df.groupby([
            "direction",
            "algorithm",
            "cc_host",
            "distance_m",
            "background_load_percent",
            "band",
            "wifi_metric",
            "transport_metric"
        ])
        .agg(
            rho_mean=("rho", "mean"),
            rho_std=("rho", "std"),
            p_value_mean=("p_value", "mean"),
            n_mean=("n", "mean"),
            valid_correlations=("rho", lambda x: x.notna().sum())
        )
        .reset_index()
    )

    physical_repeats = (
        df.groupby([
            "direction",
            "algorithm",
            "cc_host",
            "distance_m",
            "background_load_percent",
            "band"
        ])["repeat"]
        .nunique()
        .reset_index(name="physical_repeats")
    )

    grouped = grouped.merge(
        physical_repeats,
        on=[
            "direction",
            "algorithm",
            "cc_host",
            "distance_m",
            "background_load_percent",
            "band"
        ],
        how="left"
    )

    grouped["strength"] = grouped["rho_mean"].apply(
        correlation_strength
    )

    return grouped

# ============================================================
# HEATMAP
# ============================================================

def make_heatmap(df, direction, algorithm, cc_host):
    subset = df[
        (df["direction"] == direction)
        & (df["algorithm"] == algorithm)
        & (df["cc_host"] == cc_host)
    ]

    if subset.empty:
        print(
            f"[HEATMAP] Нет данных: "
            f"{direction} / {cc_host} / {algorithm}"
        )
        return

    pivot = subset.pivot_table(
        index="wifi_metric",
        columns="transport_metric",
        values="rho",
        aggfunc="mean"
    )

    plt.figure(figsize=(8, 5))

    image = plt.imshow(
        pivot.values,
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(image, label="Spearman ρ")

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index
    )

    plt.title(
        f"Spearman correlation — "
        f"{direction} / {cc_host} / {algorithm}"
    )

    plt.tight_layout()

    filename = (
        f"heatmap_{direction}_"
        f"{cc_host}_{algorithm}.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200
    )

    plt.close()

    print(
        f"[HEATMAP] Saved: "
        f"{OUTPUT_DIR / filename}"
    )


# ============================================================
# HEATMAP ПО КАЖДОМУ УСЛОВИЮ
# ============================================================

def make_condition_heatmaps(df):
    configs = [
        ("uplink", "cubic", "linux"),
        ("uplink", "bbr", "linux"),
        ("downlink", "cubic", "windows"),
        ("downlink", "bbr", "windows"),
    ]

    for direction, algorithm, cc_host in configs:

        subset = df[
            (df["direction"] == direction)
            & (df["algorithm"] == algorithm)
            & (df["cc_host"] == cc_host)
        ]

        if subset.empty:
            print(
                f"[CONDITION HEATMAP] "
                f"Нет данных: "
                f"{direction} / {cc_host} / {algorithm}"
            )
            continue

        distances = sorted(
            subset["distance_m"].dropna().unique()
        )

        udp_loads = sorted(
            subset["background_load_percent"]
            .dropna()
            .unique()
        )

        for distance in distances:
            for udp in udp_loads:

                condition = subset[
                    (subset["distance_m"] == distance)
                    & (
                        subset["background_load_percent"]
                        == udp
                    )
                ]

                if condition.empty:
                    continue

                pivot = condition.pivot_table(
                    index="wifi_metric",
                    columns="transport_metric",
                    values="rho",
                    aggfunc="mean"
                )

                if pivot.empty:
                    continue

                plt.figure(figsize=(8, 5))

                image = plt.imshow(
                    pivot.values,
                    aspect="auto",
                    vmin=-1,
                    vmax=1
                )

                plt.colorbar(
                    image,
                    label="Spearman ρ"
                )

                plt.xticks(
                    range(len(pivot.columns)),
                    pivot.columns,
                    rotation=45,
                    ha="right"
                )

                plt.yticks(
                    range(len(pivot.index)),
                    pivot.index
                )

                plt.title(
                    f"Spearman correlation\n"
                    f"{direction} / {cc_host} / {algorithm}\n"
                    f"{distance} m / {udp}% UDP"
                )

                plt.tight_layout()

                filename = (
                    f"heatmap_condition_"
                    f"{direction}_"
                    f"{cc_host}_"
                    f"{algorithm}_"
                    f"{distance}m_"
                    f"{udp}udp.png"
                )

                plt.savefig(
                    OUTPUT_DIR / filename,
                    dpi=200
                )

                plt.close()

                print(
                    f"[CONDITION HEATMAP] Saved: "
                    f"{OUTPUT_DIR / filename}"
                )

def collect_observations(runs):
    observations = []

    for run in runs:
        try:
            wifi = load_wifi(run["wifi_file"])
            tcp = load_tcp(run["tcp_file"])
            throughput = load_throughput(
                run["iperf_file"],
                run["metadata"]["experiment_start"]
            )

            synced = synchronize(
                wifi,
                tcp,
                throughput
            )

            if synced.empty:
                continue

            synced["direction"] = run["direction"]
            synced["algorithm"] = run["algorithm"]
            synced["cc_host"] = run["cc_host"]
            synced["distance_m"] = run["distance_m"]
            synced["background_load_percent"] = (
                run["background_load_percent"]
            )
            synced["band"] = run["band"]
            synced["repeat"] = run["repeat"]
            synced["run"] = str(run["run_dir"])

            observations.append(synced)

        except Exception as e:
            print(
                f"[OBSERVATIONS] Ошибка: "
                f"{run['run_dir']}"
            )
            print(e)

    if not observations:
        return pd.DataFrame()

    return pd.concat(
        observations,
        ignore_index=True
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print_runs_structure()

    OUTPUT_DIR.mkdir(exist_ok=True)

    runs = find_runs()

    print("\n" + "=" * 70)
    print("РАСПОЗНАННЫЕ ЗАПУСКИ")
    print("=" * 70)
    for run in runs: print(
        f"{run['algorithm']:>6} | " 
        f"{run['distance_m']} м | " 
        f"{run['background_load_percent']:>2}% UDP | " 
        f"{run['band']:>4} | " 
        f"{run['run_dir'].name}"
    )

    print(f"\nВсего запусков: {len(runs)}")

    print(f"Найдено запусков: {len(runs)}")

    all_results = []

    for run in runs:

        print(f"Анализ: {run}")

        try:
            results = analyze_run(run)
            all_results.extend(results)

        except Exception as e:
            print(
                f"ОШИБКА в {run}: {e}"
            )

    if not all_results:
        print("Нет результатов для анализа.")
        return

    df = pd.DataFrame(all_results)

    # --------------------------------------------------------
    # Результаты каждого запуска
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_DIR / "correlations.csv",
        index=False
    )

    # --------------------------------------------------------
    # Общая сводка
    # --------------------------------------------------------

    summary = make_summary(df)

    summary.to_csv(
        OUTPUT_DIR / "correlations_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Сводка по условиям
    # --------------------------------------------------------

    condition_summary = make_condition_summary(df)

    condition_summary.to_csv(
        OUTPUT_DIR / "correlations_by_condition.csv",
        index=False
    )

    # --------------------------------------------------------
    # Общие heatmap
    # --------------------------------------------------------

    heatmap_configs = [
        ("uplink", "cubic", "linux"),
        ("uplink", "bbr", "linux"),
        ("downlink", "cubic", "windows"),
        ("downlink", "bbr", "windows"),
    ]

    for direction, algorithm, cc_host in heatmap_configs:
        make_heatmap(
            df,
            direction,
            algorithm,
            cc_host
        )

    # --------------------------------------------------------
    # Heatmap по каждому условию
    # --------------------------------------------------------

    make_condition_heatmaps(df)

    observations_df = collect_observations(runs)

    if observations_df.empty:
        print(
            "[OBSERVATIONS] "
            "Нет данных для дополнительного анализа."
        )
    else:
        print(
            "\nНаблюдений для дополнительного анализа: "
            f"{len(observations_df)}"
        )

        observations_df, preprocessing_report = (
            preprocess_data(observations_df)
        )

        preprocessing_report.to_csv(
            OUTPUT_DIR / "preprocessing_report.csv",
            index=False
        )

        print(
            "[PREPROCESSING] "
            "Отчёт сохранён: "
            "analysis/preprocessing_report.csv"
        )

        make_rssi_boxplots(observations_df)
        analyze_cwnd_stability(observations_df)

    # --------------------------------------------------------
    # Информация
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)

    print(f"Обработано запусков: {len(runs)}")
    print(f"Получено строк анализа: {len(df)}")

    print()
    print("Файлы:")
    print("  analysis/correlations.csv")
    print("  analysis/correlations_summary.csv")
    print("  analysis/correlations_by_condition.csv")
    print("  analysis/heatmap_cubic.png")
    print("  analysis/heatmap_bbr.png")
    print("  analysis/heatmap_*_1m_*.png")
    print("  analysis/heatmap_*_5m_*.png")

def print_runs_structure():
    print("\n" + "=" * 80)
    print("СТРУКТУРА ЭКСПЕРИМЕНТОВ")
    print("=" * 80)

    runs = find_runs()

    # Группируем запуски по реальному условию,
    # определённому из имени папки
    groups = {}

    for run in runs:
        key = (
            run["direction"],
            run["algorithm"],
            run["distance_m"],
            run["background_load_percent"],
            run["band"],
        )

        groups.setdefault(key, []).append(run)

    for key, group in sorted(groups.items()):
        direction, algorithm, distance, udp, band = key

        print(
            f"\n{direction.upper()} | "
            f"{algorithm.upper()} | "
            f"{distance} м | "
            f"{udp}% UDP | "
            f"{band}"
        )

        for i, run in enumerate(
            sorted(group, key=lambda x: x["run_dir"].name),
            start=1
        ):
            print(
                f"    repeat {i}: "
                f"{run['run_dir'].name}"
            )

    print("\n" + "-" * 80)
    print(f"Всего запусков: {len(runs)}")
    print(f"Всего условий:  {len(groups)}")
    print("-" * 80)

if __name__ == "__main__":
    main()