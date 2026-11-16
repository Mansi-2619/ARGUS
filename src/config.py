from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"

LABEL_COL = "Label"
RANDOM_SEED = 42
TEST_SIZE = 0.2
SKEW_THRESHOLD = 1.0