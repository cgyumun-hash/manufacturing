"""2.3 기초 통계 — 센서 간 상관계수 (전체 행을 한꺼번에 계산, Pearson · Spearman).

출력: 표_센서_상관.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import pandas as pd  # noqa: E402

from common.data import load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402

fr = load_frames()
pairs = [("AI0 ↔ AI1", "AI0_Vibration", "AI1_Vibration"), ("AI0 ↔ 전류", "AI0_Vibration", "AI2_Current"),
         ("AI1 ↔ 전류", "AI1_Vibration", "AI2_Current")]
rows = []
for name, a, b in pairs:
    row = {"센서 쌍": name}
    for method, ko in (("pearson", "Pearson"), ("spearman", "Spearman")):
        for k, lab in (("normal", "정상"), ("anomaly", "이상")):
            row[f"{lab} {ko}"] = round(fr[k][a].corr(fr[k][b], method=method), 3)
    rows.append(row)
table = pd.DataFrame(rows)[["센서 쌍", "정상 Pearson", "이상 Pearson", "정상 Spearman", "이상 Spearman"]]
save_table(table, HERE / "표_센서_상관.csv")
print(table.to_string(index=False))
