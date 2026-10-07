"""Create Section 2.3 basic statistics, tables, figures, and report."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/manufacturing_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SENSORS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SHORT_NAMES = {"AI0_Vibration": "AI0", "AI1_Vibration": "AI1", "AI2_Current": "Current"}
FILES = {
    "normal": PROJECT / "press_data_normal.csv",
    "anomaly": PROJECT / "outlier_data.csv",
}
COLORS = {"normal": "#377eb8", "anomaly": "#e41a1c"}
EXPECTED_INTERVAL = 0.1


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_data() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for label, path in FILES.items():
        frame = pd.read_csv(path)
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("Unnamed:")])
        frame["TimeStamp"] = pd.to_datetime(frame["TimeStamp"], errors="coerce")
        frame = (
            frame.dropna(subset=["TimeStamp", *SENSORS])
            .drop_duplicates()
            .sort_values("TimeStamp", kind="stable")
            .reset_index(drop=True)
        )
        dt = frame["TimeStamp"].diff().dt.total_seconds()
        continuous = np.isclose(dt.to_numpy(), EXPECTED_INTERVAL, atol=1e-9, rtol=0)
        continuous[0] = False
        frame["interval_seconds"] = dt
        frame["segment_id"] = (~continuous).cumsum()
        frames[label] = frame
    return frames


def sensor_statistics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, frame in frames.items():
        for sensor in SENSORS:
            x = frame[sensor].to_numpy(dtype=float)
            q05, q25, median, q75, q95 = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
            rms = float(np.sqrt(np.mean(x**2)))
            rows.append(
                {
                    "class": label,
                    "sensor": sensor,
                    "count": len(x),
                    "mean": np.mean(x),
                    "std": np.std(x, ddof=1),
                    "median": median,
                    "iqr": q75 - q25,
                    "min": np.min(x),
                    "max": np.max(x),
                    "q05": q05,
                    "q25": q25,
                    "q75": q75,
                    "q95": q95,
                    "rms": rms,
                    "skewness": pd.Series(x).skew(),
                    "excess_kurtosis": pd.Series(x).kurt(),
                    "crest_factor": np.max(np.abs(x)) / rms if rms > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def interval_statistics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, frame in frames.items():
        x = frame["interval_seconds"].dropna().to_numpy(dtype=float)
        x = x[x > 0]
        rows.append(
            {
                "class": label,
                "count": len(x),
                "mean_seconds": np.mean(x),
                "std_seconds": np.std(x, ddof=1),
                "min_seconds": np.min(x),
                "q05_seconds": np.quantile(x, 0.05),
                "median_seconds": np.median(x),
                "q95_seconds": np.quantile(x, 0.95),
                "max_seconds": np.max(x),
                "exact_0.1s_count": np.isclose(x, EXPECTED_INTERVAL, atol=1e-9, rtol=0).sum(),
                "gap_over_0.1s_count": (x > EXPECTED_INTERVAL + 1e-9).sum(),
            }
        )
    return pd.DataFrame(rows)


def segment_tables(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    for label, frame in frames.items():
        lengths = frame.groupby("segment_id", sort=False).size().astype(int)
        for segment_id, samples in lengths.items():
            detail_rows.append(
                {
                    "class": label,
                    "segment_id": int(segment_id),
                    "samples": int(samples),
                    "span_seconds": (int(samples) - 1) * EXPECTED_INTERVAL,
                    "nominal_window_seconds": int(samples) * EXPECTED_INTERVAL,
                }
            )
        q05, q25, median, q75, q95 = np.quantile(lengths, [0.05, 0.25, 0.5, 0.75, 0.95])
        summary_rows.append(
            {
                "class": label,
                "segment_count": len(lengths),
                "mean_samples": lengths.mean(),
                "std_samples": lengths.std(ddof=1),
                "min_samples": lengths.min(),
                "q05_samples": q05,
                "q25_samples": q25,
                "median_samples": median,
                "q75_samples": q75,
                "q95_samples": q95,
                "max_samples": lengths.max(),
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def correlation_tables(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson_rows = []
    spearman_rows = []
    for label, frame in frames.items():
        for method, rows in (("pearson", pearson_rows), ("spearman", spearman_rows)):
            matrix = frame[SENSORS].corr(method=method)
            for left in SENSORS:
                for right in SENSORS:
                    rows.append(
                        {
                            "class": label,
                            "sensor_1": left,
                            "sensor_2": right,
                            "correlation": matrix.loc[left, right],
                        }
                    )
    return pd.DataFrame(pearson_rows), pd.DataFrame(spearman_rows)


def hedges_g(normal: np.ndarray, anomaly: np.ndarray) -> float:
    n1, n2 = len(normal), len(anomaly)
    pooled_var = ((n1 - 1) * np.var(normal, ddof=1) + (n2 - 1) * np.var(anomaly, ddof=1)) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return np.nan
    d = (np.mean(anomaly) - np.mean(normal)) / np.sqrt(pooled_var)
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return correction * d


def cliffs_delta(normal: np.ndarray, anomaly: np.ndarray) -> float:
    """P(anomaly > normal) - P(anomaly < normal), computed without a large pair matrix."""
    ordered = np.sort(normal)
    less = np.searchsorted(ordered, anomaly, side="left")
    greater = len(ordered) - np.searchsorted(ordered, anomaly, side="right")
    return float(np.sum(less - greater) / (len(normal) * len(anomaly)))


def effect_sizes(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sensor in SENSORS:
        normal_raw = frames["normal"][sensor].to_numpy(dtype=float)
        anomaly_raw = frames["anomaly"][sensor].to_numpy(dtype=float)
        for representation, normal, anomaly in (
            ("raw_signed", normal_raw, anomaly_raw),
            ("absolute_magnitude", np.abs(normal_raw), np.abs(anomaly_raw)),
        ):
            rows.append(
                {
                    "sensor": sensor,
                    "representation": representation,
                    "direction": "anomaly_minus_normal",
                    "hedges_g": hedges_g(normal, anomaly),
                    "cliffs_delta": cliffs_delta(normal, anomaly),
                    "normal_mean": np.mean(normal),
                    "anomaly_mean": np.mean(anomaly),
                    "normal_median": np.median(normal),
                    "anomaly_median": np.median(anomaly),
                }
            )
    return pd.DataFrame(rows)


def heatmap(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [SHORT_NAMES[s] for s in SENSORS]
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_title(title)
    for row in range(3):
        for col in range(3):
            color = "white" if abs(matrix[row, col]) > 0.55 else "black"
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=color)
    return image


def create_figures(
    frames: dict[str, pd.DataFrame],
    sensor_stats: pd.DataFrame,
    segment_detail: pd.DataFrame,
    effect: pd.DataFrame,
) -> None:
    # 1. Full raw-value distributions; no class balancing or sampling.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sensor in zip(axes, SENSORS):
        display_low = min(np.quantile(frames[label][sensor], 0.005) for label in ("normal", "anomaly"))
        display_high = max(np.quantile(frames[label][sensor], 0.995) for label in ("normal", "anomaly"))
        for label in ("normal", "anomaly"):
            x = frames[label][sensor].to_numpy(dtype=float)
            ax.hist(x, bins=80, range=(display_low, display_high), density=True, alpha=0.42,
                    color=COLORS[label], label=f"{label} (n={len(x):,})")
        ax.set_title(SHORT_NAMES[sensor])
        ax.set_xlabel("Raw sensor value (0.5%–99.5% display range)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Raw sensor distributions (all cleaned rows; density-normalized)")
    fig.tight_layout()
    save_figure(fig, "01_sensor_distributions.png")

    # 2. Median/IQR and 5–95% range.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, sensor in zip(axes, SENSORS):
        subset = sensor_stats[sensor_stats.sensor == sensor].set_index("class")
        for pos, label in enumerate(("normal", "anomaly")):
            row = subset.loc[label]
            ax.vlines(pos, row.q05, row.q95, color=COLORS[label], linewidth=3, alpha=0.55)
            ax.vlines(pos, row.q25, row.q75, color=COLORS[label], linewidth=10, alpha=0.9)
            ax.scatter(pos, row["median"], s=55, color="white", edgecolor="black", zorder=3)
        ax.set_xticks([0, 1], ["Normal", "Anomaly"])
        ax.set_title(SHORT_NAMES[sensor])
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Median (dot), IQR (thick), and 5–95% range (thin)")
    fig.tight_layout()
    save_figure(fig, "02_quantile_ranges.png")

    # 3. Shape and magnitude metrics on separate axes because their scales differ.
    metrics = [
        ("rms", "RMS"),
        ("skewness", "Skewness"),
        ("excess_kurtosis", "Excess kurtosis"),
        ("crest_factor", "Crest factor"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(SENSORS))
    width = 0.36
    for ax, (metric, title) in zip(axes.flat, metrics):
        for offset, label in ((-width / 2, "normal"), (width / 2, "anomaly")):
            values = sensor_stats[sensor_stats["class"] == label].set_index("sensor").loc[SENSORS, metric]
            ax.bar(x + offset, values, width, label=label.title(), color=COLORS[label], alpha=0.82)
        ax.set_xticks(x, [SHORT_NAMES[s] for s in SENSORS])
        ax.set_title(title)
        if metric == "rms":
            ax.set_yscale("log")
        else:
            ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Signal magnitude and distribution-shape statistics")
    fig.tight_layout()
    save_figure(fig, "03_rms_shape_crest.png")

    # 4. Actual positive timestamp intervals, including gaps.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label in zip(axes, ("normal", "anomaly")):
        x = frames[label]["interval_seconds"].dropna().to_numpy(dtype=float)
        x = np.sort(x[x > 0])
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=COLORS[label])
        ax.set_xscale("log")
        ax.axvline(0.1, color="black", linestyle="--", linewidth=1, label="Expected 0.1 s")
        ax.set_title(label.title())
        ax.set_xlabel("Actual timestamp interval (seconds, log scale)")
        ax.set_ylabel("Cumulative proportion")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Timestamp-interval ECDF including discontinuity gaps")
    fig.tight_layout()
    save_figure(fig, "04_timestamp_intervals.png")

    # 5. Segment length distribution; every segment is shown through count bars.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, label in zip(axes, ("normal", "anomaly")):
        lengths = segment_detail.loc[segment_detail["class"] == label, "samples"]
        counts = lengths.value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=COLORS[label], alpha=0.82)
        ax.set_title(f"{label.title()} (segments={len(lengths):,})")
        ax.set_xlabel("Segment length (samples; 10 samples ≈ 1 s window)")
        ax.set_ylabel("Number of segments")
        ax.set_xticks(np.arange(0, max(50, int(lengths.max())) + 1, 5))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Distribution of continuous segment lengths")
    fig.tight_layout()
    save_figure(fig, "05_segment_length_distribution.png")

    # 6-7. Full-row correlations by class.
    for method, filename in (("pearson", "06_pearson_correlations.png"), ("spearman", "07_spearman_correlations.png")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        last_image = None
        for ax, label in zip(axes, ("normal", "anomaly")):
            matrix = frames[label][SENSORS].corr(method=method).to_numpy()
            last_image = heatmap(ax, matrix, label.title())
        fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.85, pad=0.04, label="Correlation")
        fig.suptitle(f"{method.title()} correlation of simultaneous raw sensor values")
        save_figure(fig, filename)

    # 8. Effect sizes; positive means anomaly is larger than normal.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = []
    for sensor in SENSORS:
        names.extend([f"{SHORT_NAMES[sensor]} raw", f"{SHORT_NAMES[sensor]} |x|"])
    ordered = effect.copy()
    ordered["sensor_order"] = ordered.sensor.map({s: i for i, s in enumerate(SENSORS)})
    ordered["repr_order"] = ordered.representation.map({"raw_signed": 0, "absolute_magnitude": 1})
    ordered = ordered.sort_values(["sensor_order", "repr_order"])
    y = np.arange(len(ordered))
    for ax, metric, title in (
        (axes[0], "hedges_g", "Hedges' g"),
        (axes[1], "cliffs_delta", "Cliff's delta"),
    ):
        values = ordered[metric].to_numpy()
        ax.barh(y, values, color=np.where(values >= 0, "#e41a1c", "#377eb8"), alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        if metric == "cliffs_delta":
            for boundary in (-0.474, -0.33, -0.147, 0.147, 0.33, 0.474):
                ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.7)
            ax.set_xlim(-1, 1)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Positive = anomaly larger")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Normal–anomaly effect sizes (all cleaned rows)")
    fig.tight_layout()
    save_figure(fig, "08_effect_sizes.png")


def fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def write_report(
    sensor_stats: pd.DataFrame,
    intervals: pd.DataFrame,
    segment_summary: pd.DataFrame,
    effect: pd.DataFrame,
) -> None:
    lines = [
        "# 2.3 기초 통계",
        "",
        "정상·이상 데이터의 규모, 분포, 시간 품질, 센서 관계와 효과크기를 한 번에 확인한다. 통계는 결측·정확히 중복된 행을 제거하고 타임스탬프를 정렬한 데이터로 계산했다. 정상은 19,999행, 이상은 600행이며 클래스 균형을 맞추기 위한 샘플링은 하지 않았다.",
        "",
        "## 계산 기준",
        "",
        "- 센서 기술통계와 상관계수는 정제된 모든 원시 행을 사용했다.",
        "- RMS는 `sqrt(mean(x²))`, IQR은 `75% 분위수−25% 분위수`다.",
        "- kurtosis는 정규분포가 0인 excess kurtosis다.",
        "- crest factor는 `max(|x|)/RMS`다. 여기서는 클래스 전체 파형에 대한 탐색 통계이며, 모델 입력에서는 연속 segment 또는 window 안에서 다시 계산해야 한다.",
        "- 실제 타임스탬프 간격 통계에는 0.1초 구간뿐 아니라 불연속 공백도 포함했다.",
        "- 연속 segment는 인접 타임스탬프 간격이 정확히 0.1초인 행만 이어서 구성했다.",
        "- 효과크기의 방향은 `이상−정상`이다. 양수이면 이상 값이 더 크고 음수이면 더 작다.",
        "",
        "## 1. 데이터 개수와 센서 분포",
        "",
        "![센서 분포](figures/01_sensor_distributions.png)",
        "",
        "히스토그램은 정상 19,999행과 이상 600행을 모두 사용하고, 클래스별 면적이 1이 되도록 density로 정규화했다. 따라서 막대 높이는 데이터 개수가 아니라 분포 모양을 비교한다. 극단값 때문에 중심부가 눌리지 않도록 그림만 각 클래스의 0.5~99.5% 범위를 표시했으며 CSV 통계에는 전체 최솟값과 최댓값이 포함된다.",
        "",
        "## 2. 평균·표준편차·중앙값·IQR·분위수·범위",
        "",
        "![분위수 범위](figures/02_quantile_ranges.png)",
        "",
        "| 구분 | 센서 | 개수 | 평균 ± 표준편차 | 중앙값 [IQR] | 최솟값–최댓값 | 5% / 25% / 75% / 95% |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in sensor_stats.iterrows():
        lines.append(
            f"| {row['class']} | {SHORT_NAMES[row['sensor']]} | {int(row['count']):,} | "
            f"{fmt(row['mean'])} ± {fmt(row['std'])} | {fmt(row['median'])} [{fmt(row['iqr'])}] | "
            f"{fmt(row['min'])}–{fmt(row['max'])} | {fmt(row['q05'])} / {fmt(row['q25'])} / {fmt(row['q75'])} / {fmt(row['q95'])} |"
        )
    lines.extend([
        "",
        "전체 정밀값은 `tables/sensor_descriptive_statistics.csv`에 저장했다.",
        "",
        "## 3. RMS·왜도·kurtosis·crest factor",
        "",
        "![RMS와 분포 형태](figures/03_rms_shape_crest.png)",
        "",
        "| 구분 | 센서 | RMS | 왜도 | Excess kurtosis | Crest factor |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in sensor_stats.iterrows():
        lines.append(
            f"| {row['class']} | {SHORT_NAMES[row['sensor']]} | {fmt(row['rms'])} | "
            f"{fmt(row['skewness'])} | {fmt(row['excess_kurtosis'])} | {fmt(row['crest_factor'])} |"
        )
    lines.extend([
        "",
        "RMS는 전체적인 신호 크기, 왜도는 좌우 비대칭, kurtosis는 꼬리와 극단값의 강도, crest factor는 RMS 대비 최대 peak의 크기를 나타낸다. 전체 클래스 단위 crest factor는 데이터 길이에 영향을 받으므로 정상·이상 비교를 확정하는 지표로 단독 사용하지 않는다.",
        "",
        "## 4. 실제 타임스탬프 간격",
        "",
        "![타임스탬프 간격](figures/04_timestamp_intervals.png)",
        "",
        "| 구분 | 간격 수 | 평균 ± 표준편차(초) | 최솟값 | 중앙값 | 최댓값 | 0.1초 | 0.1초 초과 공백 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in intervals.iterrows():
        lines.append(
            f"| {row['class']} | {int(row['count']):,} | {fmt(row['mean_seconds'])} ± {fmt(row['std_seconds'])} | "
            f"{fmt(row['min_seconds'])} | {fmt(row['median_seconds'])} | {fmt(row['max_seconds'])} | "
            f"{int(row['exact_0.1s_count']):,} | {int(row['gap_over_0.1s_count']):,} |"
        )
    lines.extend([
        "",
        "평균과 표준편차에는 긴 수집 공백이 포함되므로 0.1초 샘플링 주기의 안정성을 나타내는 값이 아니다. 실제 연속 여부는 0.1초 간격 수와 공백 수로 판단해야 한다.",
        "",
        "## 5. 연속 segment 개수와 길이 분포",
        "",
        "![세그먼트 길이](figures/05_segment_length_distribution.png)",
        "",
        "| 구분 | segment 수 | 평균 ± 표준편차(시점) | 최소 | 5% | 25% | 중앙값 | 75% | 95% | 최대 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in segment_summary.iterrows():
        lines.append(
            f"| {row['class']} | {int(row['segment_count']):,} | {fmt(row['mean_samples'])} ± {fmt(row['std_samples'])} | "
            f"{fmt(row['min_samples'])} | {fmt(row['q05_samples'])} | {fmt(row['q25_samples'])} | "
            f"{fmt(row['median_samples'])} | {fmt(row['q75_samples'])} | {fmt(row['q95_samples'])} | {fmt(row['max_samples'])} |"
        )
    lines.extend([
        "",
        "한 시점은 0.1초 간격이다. 예를 들어 20시점은 모델 입력 길이로 약 2초이며, 첫 타임스탬프와 마지막 타임스탬프 사이의 span은 1.9초다. 모든 segment의 길이는 `tables/segment_lengths.csv`에 저장했다.",
        "",
        "## 6. 센서 간 Pearson·Spearman 상관계수",
        "",
        "![Pearson 상관](figures/06_pearson_correlations.png)",
        "",
        "![Spearman 상관](figures/07_spearman_correlations.png)",
        "",
        "Pearson은 동시 센서값의 선형 관계, Spearman은 값의 순위가 함께 증가하거나 감소하는 단조 관계를 본다. 이 그림은 모든 원시 행을 한꺼번에 계산한 탐색 결과다. 프레스 사이클이나 시간 지연 관계를 의미하지 않으며, 불연속 segment를 이어 붙여 시차 상관을 계산한 것도 아니다.",
        "",
        "## 7. 정상·이상 간 효과크기",
        "",
        "![효과크기](figures/08_effect_sizes.png)",
        "",
        "| 센서 | 표현 | Hedges' g | Cliff's delta | 정상 중앙값 | 이상 중앙값 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in effect.iterrows():
        representation = "원신호" if row["representation"] == "raw_signed" else "절댓값(abs)"
        lines.append(
            f"| {SHORT_NAMES[row['sensor']]} | {representation} | {fmt(row['hedges_g'])} | "
            f"{fmt(row['cliffs_delta'])} | {fmt(row['normal_median'])} | {fmt(row['anomaly_median'])} |"
        )
    lines.extend([
        "",
        "Hedges' g는 평균 차이를 합동 표준편차 단위로 나타내며, 일반적으로 절댓값 0.2/0.5/0.8을 작은/중간/큰 차이의 참고선으로 본다. Cliff's delta는 임의의 이상 값이 정상 값보다 클 확률에서 작을 확률을 뺀 비모수 효과크기이며, -1~1 범위다. 진동 원신호는 양·음이 상쇄될 수 있으므로 `|x|` 효과도 함께 제시했다.",
        "",
        "## 해석 시 제한",
        "",
        "- 정상과 이상은 서로 다른 날짜에 측정되어 날짜·운전조건 효과와 고장 효과가 섞일 수 있다.",
        "- 행 단위 표본은 시간적으로 독립이 아니므로 행 수가 많다고 독립 실험 수가 많은 것은 아니다.",
        "- 정상 19,999행과 이상 600행으로 불균형하며, 특히 효과크기의 불확실성은 segment 단위 bootstrap 등으로 추가 확인해야 한다.",
        "- 이 단계는 탐색 통계다. 모델 검증에서는 반드시 segment 단위 분리를 사용한다.",
    ])
    (OUT / "2.3_기초_통계.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = load_data()
    stats = sensor_statistics(frames)
    intervals = interval_statistics(frames)
    segment_detail, segment_summary = segment_tables(frames)
    pearson, spearman = correlation_tables(frames)
    effect = effect_sizes(frames)

    tables = {
        "sensor_descriptive_statistics.csv": stats,
        "timestamp_interval_statistics.csv": intervals,
        "segment_lengths.csv": segment_detail,
        "segment_length_summary.csv": segment_summary,
        "pearson_correlations.csv": pearson,
        "spearman_correlations.csv": spearman,
        "normal_anomaly_effect_sizes.csv": effect,
    }
    for name, table in tables.items():
        table.to_csv(TABLES / name, index=False)

    create_figures(frames, stats, segment_detail, effect)
    write_report(stats, intervals, segment_summary, effect)
    print(f"Wrote {len(tables)} tables, 8 figures, and {OUT / '2.3_기초_통계.md'}")


if __name__ == "__main__":
    main()
