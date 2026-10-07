"""그림 공통 스타일: matplotlib 기본 스타일 + 나눔고딕, 정상 = 파랑, 이상 = 빨강."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcdefaults()
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "savefig.dpi": 200})

C = {"normal": "#2878B5", "anomaly": "#D9534F"}
LABEL = {"normal": "정상", "anomaly": "이상"}
RAW_C, DER_C, SINGLE_C = "#2878B5", "#2CA02C", "#777777"
AQUA = "#2CA02C"
INK, INK2, MUTED, GRID = "#000000", "#333333", "#777777", "#dddddd"


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
