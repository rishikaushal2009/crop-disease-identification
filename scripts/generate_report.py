import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
HISTORY = ROOT / "outputs" / "training_history.csv"
CM_PATH = ROOT / "outputs" / "confusion_matrix.png"
SAMPLES_DIR = ROOT / "outputs" / "samples"

PERF_START = "<!-- PERF_SUMMARY_START -->"
PERF_END = "<!-- PERF_SUMMARY_END -->"
PRED_START = "<!-- PRED_SAMPLES_START -->"
PRED_END = "<!-- PRED_SAMPLES_END -->"

def build_perf_block() -> str:
    if not HISTORY.exists():
        return ("\nPerformance table pending: outputs/training_history.csv not found. Run training first.\n")
    df = pd.read_csv(HISTORY)
    if df.empty:
        return ("\nPerformance table pending: training history is empty.\n")
    best = df.loc[df['val_acc'].idxmax()]
    epoch = int(best['epoch'])
    val_acc = float(best['val_acc'])
    tr_acc = float(best.get('train_acc', float('nan')))
    header = (
        "| Model | Classes | Epochs | Best Epoch | Val Acc | Train Acc |\n"
        "|-------|---------|--------|-----------|---------|-----------|\n"
    )
    # We don't know which backbone was used if user changed it; default to MobileNetV3-S
    row = f"| MobileNetV3-S | 12 | {int(df['epoch'].max())} | {epoch} | {val_acc:.3f} | {tr_acc:.3f} |\n"
    return "\n" + header + row + "\n"

def build_pred_block() -> str:
    if not SAMPLES_DIR.exists():
        return ("\nSamples pending: outputs/samples not found. Run the notebook to export samples.\n")
    imgs = []
    for prefix in ("correct", "incorrect"):
        for i in range(1,6):
            p = SAMPLES_DIR / f"{prefix}_{i}.jpg"
            if p.exists():
                imgs.append((prefix, p))
    if not imgs:
        return ("\nSamples pending: no images found under outputs/samples.\n")
    lines = ["\n**Correct Predictions (5)**"]
    for i in range(1,6):
        p = SAMPLES_DIR / f"correct_{i}.jpg"
        if p.exists():
            rel = p.relative_to(ROOT).as_posix()
            lines.append(f"![]({rel})")
    lines.append("\n**Incorrect Predictions (5)**")
    for i in range(1,6):
        p = SAMPLES_DIR / f"incorrect_{i}.jpg"
        if p.exists():
            rel = p.relative_to(ROOT).as_posix()
            lines.append(f"![]({rel})")
    lines.append("")
    return "\n" + "\n".join(lines) + "\n"


def replace_block(text: str, start: str, end: str, new_block: str) -> str:
    if start in text and end in text:
        pattern = re.compile(re.escape(start) + r"[\s\S]*?" + re.escape(end), re.MULTILINE)
        return pattern.sub(start + new_block + end, text)
    else:
        # Append block if markers missing
        return text.rstrip() + "\n\n" + start + new_block + end + "\n"


def main():
    if not README.exists():
        print("README.md not found")
        sys.exit(1)
    content = README.read_text(encoding="utf-8")

    perf_block = build_perf_block()
    pred_block = build_pred_block()

    content = replace_block(content, PERF_START, PERF_END, perf_block)
    content = replace_block(content, PRED_START, PRED_END, pred_block)

    README.write_text(content, encoding="utf-8")
    print("README updated with performance summary and prediction samples.")

if __name__ == "__main__":
    main()
