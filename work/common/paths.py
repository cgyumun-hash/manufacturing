"""work 폴더 안의 경로."""
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data"
RAW = DATA / "raw"
NORMAL_CSV = RAW / "press_data_normal.csv"
ANOMALY_CSV = RAW / "outlier_data.csv"
MODELS = WORK / "models"
SUMMARY = MODELS / "summary"
FIGURES_CH3 = WORK / "figures_ch3"
FIGURES_CH4 = WORK / "figures_ch4"
SHAP_ABLATION = WORK / "shap_ablation"


def save_table(frame, path, **kwargs):
    """엑셀에서 한글이 깨지지 않도록 utf-8-sig로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", **kwargs)
