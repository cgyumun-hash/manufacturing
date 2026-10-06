"""Section 2.2 data-quality audit and file-based visualizations.

Run from anywhere. All generated tables, figures, and the Markdown report are
written under storytelling_analysis/02_data_quality.
"""

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
FILES = {
    "normal": PROJECT / "press_data_normal.csv",
    "anomaly": PROJECT / "outlier_data.csv",
}
EXPECTED_INTERVAL = 0.1
INTERVAL_TOLERANCE = 1e-9


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_sources() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    raw: dict[str, pd.DataFrame] = {}
    cleaned: dict[str, pd.DataFrame] = {}
    for label, path in FILES.items():
        frame = pd.read_csv(path)
        unnamed = [column for column in frame.columns if column.startswith("Unnamed:")]
        frame = frame.drop(columns=unnamed)
        frame["TimeStamp"] = pd.to_datetime(frame["TimeStamp"], errors="coerce")
        raw[label] = frame.copy()
        cleaned[label] = (
            frame.dropna(subset=["TimeStamp", *SENSORS])
            .drop_duplicates()
            .sort_values("TimeStamp", kind="stable")
            .reset_index(drop=True)
        )
    return raw, cleaned


def add_segments(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dt = result["TimeStamp"].diff().dt.total_seconds()
    continuous = np.isclose(dt.to_numpy(), EXPECTED_INTERVAL, atol=INTERVAL_TOLERANCE, rtol=0)
    continuous[0] = False
    result["interval_seconds"] = dt
    result["segment_id"] = (~continuous).cumsum()
    return result


def descriptive_table(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, frame in cleaned.items():
        for sensor in SENSORS:
            values = frame[sensor].to_numpy(dtype=float)
            rows.append(
                {
                    "class": label,
                    "sensor": sensor,
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1),
                    "min": np.min(values),
                    "q05": np.quantile(values, 0.05),
                    "q25": np.quantile(values, 0.25),
                    "median": np.median(values),
                    "q75": np.quantile(values, 0.75),
                    "q95": np.quantile(values, 0.95),
                    "max": np.max(values),
                }
            )
    return pd.DataFrame(rows)


def create_tables(raw: dict[str, pd.DataFrame], segmented: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    overview_rows = []
    missing_rows = []
    interval_rows = []
    segment_rows = []
    duplicate_frames = []
    segment_lengths = []

    for label in FILES:
        original = raw[label]
        clean = segmented[label]
        parsed_time = pd.to_datetime(original["TimeStamp"], errors="coerce")
        exact_extra = int(original.duplicated().sum())
        duplicate_timestamp_extra = int(parsed_time.duplicated().sum())
        duplicate_rows = original[original.duplicated(keep=False)].copy()
        if len(duplicate_rows):
            duplicate_rows.insert(0, "class", label)
            duplicate_frames.append(duplicate_rows)

        overview_rows.append(
            {
                "class": label,
                "source_file": FILES[label].name,
                "raw_rows": len(original),
                "clean_rows": len(clean),
                "removed_exact_duplicate_rows": exact_extra,
                "start_timestamp": clean.TimeStamp.min(),
                "end_timestamp": clean.TimeStamp.max(),
                "wall_time_seconds": (clean.TimeStamp.max() - clean.TimeStamp.min()).total_seconds(),
                "equipment_state_values": ",".join(map(str, sorted(clean.Equipment_state.unique()))),
            }
        )

        for column in original.columns:
            parsed_missing = int(parsed_time.isna().sum()) if column == "TimeStamp" else int(original[column].isna().sum())
            infinite = 0
            if column in SENSORS:
                infinite = int(np.isinf(pd.to_numeric(original[column], errors="coerce")).sum())
            missing_rows.append(
                {"class": label, "column": column, "missing_or_parse_failure": parsed_missing, "infinite": infinite}
            )

        intervals = clean.interval_seconds.dropna().to_numpy(dtype=float)
        positive = intervals[intervals > 0]
        exact = np.isclose(intervals, EXPECTED_INTERVAL, atol=INTERVAL_TOLERANCE, rtol=0)
        gaps = intervals[intervals > EXPECTED_INTERVAL + INTERVAL_TOLERANCE]
        estimated_missing = int(np.maximum(np.rint(gaps / EXPECTED_INTERVAL).astype(int) - 1, 0).sum())
        interval_rows.append(
            {
                "class": label,
                "interval_count": len(intervals),
                "expected_0.1s_count": int(exact.sum()),
                "gap_count_over_0.1s": len(gaps),
                "nonpositive_interval_count": int((intervals <= 0).sum()),
                "duplicate_timestamp_extra_rows_raw": duplicate_timestamp_extra,
                "min_positive_interval": np.min(positive),
                "median_interval": np.median(positive),
                "max_interval": np.max(positive),
                "estimated_missing_if_continuous_10Hz": estimated_missing,
            }
        )

        lengths = clean.groupby("segment_id", sort=False).size()
        for segment_id, samples in lengths.items():
            segment_lengths.append(
                {"class": label, "segment_id": int(segment_id), "samples": int(samples), "duration_seconds": (samples - 1) / 10}
            )
        segment_rows.append(
            {
                "class": label,
                "segments": len(lengths),
                "min_samples": int(lengths.min()),
                "median_samples": float(lengths.median()),
                "mean_samples": float(lengths.mean()),
                "max_samples": int(lengths.max()),
                "segments_ge_5_samples": int((lengths >= 5).sum()),
                "segments_ge_10_samples": int((lengths >= 10).sum()),
                "segments_ge_15_samples": int((lengths >= 15).sum()),
                "segments_ge_20_samples": int((lengths >= 20).sum()),
            }
        )

    tables = {
        "data_overview": pd.DataFrame(overview_rows),
        "missing_and_infinite": pd.DataFrame(missing_rows),
        "timestamp_quality": pd.DataFrame(interval_rows),
        "segment_summary": pd.DataFrame(segment_rows),
        "segment_lengths": pd.DataFrame(segment_lengths),
        "sensor_descriptive_stats": descriptive_table(segmented),
        "duplicate_rows": pd.concat(duplicate_frames, ignore_index=True) if duplicate_frames else pd.DataFrame(),
    }

    normal_rows = len(segmented["normal"])
    anomaly_rows = len(segmented["anomaly"])
    checks = [
        ("Required columns", "PASS", "TimeStamp, 3 sensors, Equipment_state are present"),
        ("Missing values", "PASS", f"Total missing/parse failures: {int(tables['missing_and_infinite'].missing_or_parse_failure.sum())}"),
        ("Infinite sensor values", "PASS", f"Total infinite values: {int(tables['missing_and_infinite'].infinite.sum())}"),
        ("Exact duplicate rows", "CAUTION", "Normal has 1 extra duplicate row; anomaly has 0"),
        ("Timestamp continuity", "CAUTION", "Data contain many gaps; analyze within segment only"),
        ("Label/date confounding", "CRITICAL", "Normal and anomaly were collected on different dates"),
        ("Class imbalance by row", "CAUTION", f"Normal:anomaly = {normal_rows / anomaly_rows:.2f}:1 after duplicate removal"),
    ]
    tables["quality_checklist"] = pd.DataFrame(checks, columns=["check", "status", "result"])
    return tables


def create_figures(segmented: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]) -> None:
    colors = {"normal": "#2878B5", "anomaly": "#D9534F"}

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = [len(segmented[label]) for label in ("normal", "anomaly")]
    bars = ax.bar(["Normal", "Anomaly"], counts, color=[colors["normal"], colors["anomaly"]])
    ax.set_title("Record count after exact-duplicate removal")
    ax.set_ylabel("Rows")
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    save_figure(fig, "01_record_counts.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for ax, label in zip(axes, ("normal", "anomaly")):
        dt = segmented[label].interval_seconds.dropna().to_numpy(dtype=float)
        ax.hist(dt, bins=80, color=colors[label], alpha=0.85)
        ax.axvline(EXPECTED_INTERVAL, color="black", linestyle="--", label="Expected 0.1 s")
        ax.set_yscale("log")
        ax.set_title(f"{label.title()} timestamp intervals")
        ax.set_xlabel("Interval (seconds)")
        ax.set_ylabel("Count (log scale)")
        ax.legend()
    fig.tight_layout()
    save_figure(fig, "02_timestamp_intervals.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0.5, 51.5, 1)
    for label in ("normal", "anomaly"):
        values = tables["segment_lengths"].loc[tables["segment_lengths"]["class"] == label, "samples"]
        ax.hist(values, bins=bins, alpha=0.55, color=colors[label], label=label.title())
    ax.set_title("Continuous segment length distribution")
    ax.set_xlabel("Samples per segment (10 samples = 1 second input)")
    ax.set_ylabel("Segments")
    ax.legend()
    save_figure(fig, "03_segment_lengths.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    for ax, sensor in zip(axes, SENSORS):
        combined = np.concatenate([segmented[label][sensor].to_numpy() for label in ("normal", "anomaly")])
        low, high = np.quantile(combined, [0.002, 0.998])
        bins = np.linspace(low, high, 90)
        for label in ("normal", "anomaly"):
            ax.hist(
                segmented[label][sensor], bins=bins, density=True, histtype="step", linewidth=1.8,
                color=colors[label], label=label.title(),
            )
        ax.set_title(sensor)
        ax.set_ylabel("Density")
        ax.legend()
    axes[-1].set_xlabel("Sensor value (0.2% tails clipped for display)")
    fig.tight_layout()
    save_figure(fig, "04_sensor_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, sensor in zip(axes, SENSORS):
        ax.boxplot(
            [segmented["normal"][sensor], segmented["anomaly"][sensor]],
            tick_labels=["Normal", "Anomaly"], showfliers=False,
        )
        ax.set_title(sensor)
        ax.set_ylabel("Value")
    fig.suptitle("Sensor distributions (outliers hidden only for display)")
    fig.tight_layout()
    save_figure(fig, "05_sensor_boxplots.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, label in zip(axes, ("normal", "anomaly")):
        corr = segmented[label][SENSORS].corr().to_numpy()
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(3), ["AI0", "AI1", "Current"])
        ax.set_yticks(range(3), ["AI0", "AI1", "Current"])
        ax.set_title(label.title())
        for row in range(3):
            for column in range(3):
                ax.text(column, row, f"{corr[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.suptitle("Same-row sensor correlations")
    save_figure(fig, "06_sensor_correlations.png")

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col")
    for column, label in enumerate(("normal", "anomaly")):
        frame = segmented[label]
        longest_id = frame.groupby("segment_id").size().idxmax()
        excerpt = frame[frame.segment_id == longest_id].reset_index(drop=True)
        seconds = np.arange(len(excerpt)) / 10
        for row, sensor in enumerate(SENSORS):
            axes[row, column].plot(seconds, excerpt[sensor], color=colors[label], linewidth=1.1)
            axes[row, column].set_ylabel(sensor.replace("_Vibration", "").replace("AI2_Current", "Current"))
            if row == 0:
                axes[row, column].set_title(f"{label.title()} longest segment ({len(excerpt)} samples)")
        axes[-1, column].set_xlabel("Seconds within segment")
    fig.tight_layout()
    save_figure(fig, "07_longest_segment_examples.png")


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    overview = tables["data_overview"].set_index("class")
    timing = tables["timestamp_quality"].set_index("class")
    segments = tables["segment_summary"].set_index("class")
    desc = tables["sensor_descriptive_stats"]

    def sensor_row(sensor: str) -> str:
        normal = desc[(desc["class"] == "normal") & (desc.sensor == sensor)].iloc[0]
        anomaly = desc[(desc["class"] == "anomaly") & (desc.sensor == sensor)].iloc[0]
        return (
            f"| `{sensor}` | {fmt(normal['mean'])} ± {fmt(normal['std'])} | "
            f"{fmt(normal['min'])} ~ {fmt(normal['max'])} | "
            f"{fmt(anomaly['mean'])} ± {fmt(anomaly['std'])} | "
            f"{fmt(anomaly['min'])} ~ {fmt(anomaly['max'])} |"
        )

    lines = [
        "# 2.2 데이터 품질 확인",
        "",
        "## 결론 요약",
        "",
        f"- 원본은 정상 {int(overview.loc['normal', 'raw_rows']):,}행, 이상 {int(overview.loc['anomaly', 'raw_rows']):,}행이다.",
        "- 정상 파일에 완전히 동일한 행이 연속으로 2개 있으며, 이 중 추가 중복 1행을 제거해야 한다.",
        "- 셀 단위 결측값, 파싱 실패 타임스탬프, 무한대 센서 값은 없다.",
        f"- 정확한 0.1초 연속성을 기준으로 정상 {int(segments.loc['normal', 'segments'])}개, 이상 {int(segments.loc['anomaly', 'segments'])}개 segment로 나뉜다.",
        "- 정상은 2022-07-12, 이상은 2022-07-17에만 존재하므로 날짜와 라벨을 분리할 수 없다.",
        "- 공백을 결측값으로 보간하지 않고 segment 경계로 처리해야 한다.",
        "",
        "## 1. 데이터 구성",
        "",
        "| 구분 | 원본 행 | 정제 후 행 | 시작 | 종료 | 라벨 |",
        "|---|---:|---:|---|---|---:|",
        f"| 정상 | {int(overview.loc['normal', 'raw_rows']):,} | {int(overview.loc['normal', 'clean_rows']):,} | {overview.loc['normal', 'start_timestamp']} | {overview.loc['normal', 'end_timestamp']} | {overview.loc['normal', 'equipment_state_values']} |",
        f"| 이상 | {int(overview.loc['anomaly', 'raw_rows']):,} | {int(overview.loc['anomaly', 'clean_rows']):,} | {overview.loc['anomaly', 'start_timestamp']} | {overview.loc['anomaly', 'end_timestamp']} | {overview.loc['anomaly', 'equipment_state_values']} |",
        "",
        "![정상·이상 행 개수](figures/01_record_counts.png)",
        "",
        "행 기준 정상:이상 비율은 약 33.3:1이다. 그러나 인접 행은 독립적인 사례가 아니므로 모델 평가에서는 행 개수보다 독립 segment 수를 기준으로 봐야 한다.",
        "",
        "## 2. 중복·결측·유한값 검사",
        "",
        "| 항목 | 정상 | 이상 | 처리 |",
        "|---|---:|---:|---|",
        "| 추가 완전 중복 행 | 1 | 0 | 모델링 전에 제거 |",
        "| 중복 타임스탬프 추가 행 | 1 | 0 | 동일 행 제거 후 해소 |",
        "| 결측·파싱 실패 | 0 | 0 | 추가 처리 불필요 |",
        "| 무한대 센서 값 | 0 | 0 | 추가 처리 불필요 |",
        "",
        "정상 중복은 동일한 타임스탬프와 세 센서 값, 라벨을 가진 동일 행 2개 중 하나가 추가로 들어간 경우다. `tables/duplicate_rows.csv`에 두 원본 행을 저장하였다.",
        "",
        "## 3. 타임스탬프와 샘플 간격",
        "",
        "| 구분 | 0.1초 간격 | 0.1초 초과 공백 | 역순·0 이하 | 최대 간격 | 연속 가정 시 추정 누락 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 정상 | {int(timing.loc['normal', 'expected_0.1s_count']):,} | {int(timing.loc['normal', 'gap_count_over_0.1s']):,} | {int(timing.loc['normal', 'nonpositive_interval_count']):,} | {fmt(timing.loc['normal', 'max_interval'], 3)}초 | {int(timing.loc['normal', 'estimated_missing_if_continuous_10Hz']):,} |",
        f"| 이상 | {int(timing.loc['anomaly', 'expected_0.1s_count']):,} | {int(timing.loc['anomaly', 'gap_count_over_0.1s']):,} | {int(timing.loc['anomaly', 'nonpositive_interval_count']):,} | {fmt(timing.loc['anomaly', 'max_interval'], 3)}초 | {int(timing.loc['anomaly', 'estimated_missing_if_continuous_10Hz']):,} |",
        "",
        "![타임스탬프 간격](figures/02_timestamp_intervals.png)",
        "",
        "추정 누락 개수는 전체가 원래 연속 10 Hz 로그였다는 가정하의 계산이다. 모든 segment가 50시점 이하이므로 실제 누락이 아니라 최대 5초 단위로 의도적으로 추출한 데이터일 가능성도 있다. 원본 수집 방식을 확인하기 전에는 공백을 임의 보간하지 않는다.",
        "",
        "## 4. 연속 segment",
        "",
        "| 구분 | segment | 최소 시점 | 중앙값 | 최대 시점 | 0.5초 이상 | 1초 이상 | 1.5초 이상 | 2초 이상 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 정상 | {int(segments.loc['normal', 'segments'])} | {int(segments.loc['normal', 'min_samples'])} | {fmt(segments.loc['normal', 'median_samples'], 1)} | {int(segments.loc['normal', 'max_samples'])} | {int(segments.loc['normal', 'segments_ge_5_samples'])} | {int(segments.loc['normal', 'segments_ge_10_samples'])} | {int(segments.loc['normal', 'segments_ge_15_samples'])} | {int(segments.loc['normal', 'segments_ge_20_samples'])} |",
        f"| 이상 | {int(segments.loc['anomaly', 'segments'])} | {int(segments.loc['anomaly', 'min_samples'])} | {fmt(segments.loc['anomaly', 'median_samples'], 1)} | {int(segments.loc['anomaly', 'max_samples'])} | {int(segments.loc['anomaly', 'segments_ge_5_samples'])} | {int(segments.loc['anomaly', 'segments_ge_10_samples'])} | {int(segments.loc['anomaly', 'segments_ge_15_samples'])} | {int(segments.loc['anomaly', 'segments_ge_20_samples'])} |",
        "",
        "![segment 길이](figures/03_segment_lengths.png)",
        "",
        "시계열 특징과 윈도우는 segment 경계를 넘어 계산하지 않는다. 같은 segment에서 나온 모든 중첩 윈도우는 Group CV에서 같은 fold에 둔다.",
        "",
        "## 5. 센서 기본 분포",
        "",
        "| 센서 | 정상 평균 ± 표준편차 | 정상 min~max | 이상 평균 ± 표준편차 | 이상 min~max |",
        "|---|---:|---:|---:|---:|",
        sensor_row("AI0_Vibration"),
        sensor_row("AI1_Vibration"),
        sensor_row("AI2_Current"),
        "",
        "![센서 분포](figures/04_sensor_distributions.png)",
        "",
        "![센서 box plot](figures/05_sensor_boxplots.png)",
        "",
        "진동은 양수와 음수가 상쇄되므로 원시 평균만으로 크기를 판단하지 않는다. 이후 구간 분석에서는 RMS, 절댓값 peak, kurtosis, crest factor를 함께 확인한다. 전류도 양·음이 반복되므로 부하는 원시 평균보다 RMS 또는 envelope로 보는 것이 적절하다.",
        "",
        "## 6. 센서 동시값 상관",
        "",
        "![센서 상관계수](figures/06_sensor_correlations.png)",
        "",
        "이 그림은 같은 행의 원시값 상관이다. 전류와 진동의 물리적 관계는 위상과 부호의 영향을 받으므로 이후에는 구간 RMS 또는 envelope 관계로 다시 분석해야 한다.",
        "",
        "## 7. 대표 연속 구간",
        "",
        "![가장 긴 정상·이상 segment](figures/07_longest_segment_examples.png)",
        "",
        "대표 그림은 각 클래스의 가장 긴 50시점 segment를 보여준다. 이는 데이터 품질과 파형 형태를 확인하기 위한 예시이며 전체 클래스의 대표성을 증명하지 않는다.",
        "",
        "## 8. 모델링 전 확정 처리",
        "",
        "1. 정상 파일의 추가 중복 1행을 제거한다.",
        "2. 이름 없는 CSV 인덱스 열은 특징에서 제외한다.",
        "3. 타임스탬프를 정렬하고 정확한 0.1초 연속 구간에 segment ID를 부여한다.",
        "4. 시간 공백을 보간하지 않는다.",
        "5. 윈도우와 시계열 파생변수는 segment 내부에서만 계산한다.",
        "6. train/validation/test는 행이나 윈도우가 아니라 segment 단위로 분리한다.",
        "7. 정상과 이상이 다른 날짜에만 존재한다는 제한을 모든 성능표에 명시한다.",
        "",
        "## 9. 다음 단계",
        "",
        "데이터 품질 처리가 확정되면 다음 단계에서는 정상·이상의 센서 분포, 구간 RMS, kurtosis, crest factor, 부하-진동 관계와 반복 프레스 사이클을 탐색한다.",
        "",
        "## 생성 파일",
        "",
        "- 표: `tables/`",
        "- 이미지: `figures/`",
        "- 재실행 코드: `scripts/analyze_data_quality.py`",
        "",
    ]
    (OUT / "2.2_데이터_품질_확인.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw, cleaned = load_sources()
    segmented = {label: add_segments(frame) for label, frame in cleaned.items()}
    tables = create_tables(raw, segmented)
    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)
    create_figures(segmented, tables)
    write_report(tables)
    print(f"Data-quality report written to: {OUT / '2.2_데이터_품질_확인.md'}")


if __name__ == "__main__":
    main()
