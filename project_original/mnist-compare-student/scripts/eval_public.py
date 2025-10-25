import os, argparse, pandas as pd
from .utils.metrics import accuracy, f1_macro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--pred", type=str, required=True, help="pred_public.csv path")
    ap.add_argument("--labels", type=str, required=True, help="test_public_labels.csv path (TA only)")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    gt   = pd.read_csv(args.labels)
    df = gt.merge(pred, on="id", how="left", validate="one_to_one", suffixes=("_gt","_pred"))
    if df["label_pred"].isna().any():
        missing = df[df["label_pred"].isna()]["id"].head(5).tolist()
        raise SystemExit(f"Missing predictions for {sum(df['label_pred'].isna())} ids, e.g. {missing}")
    y_true = df["label_gt"].astype(int).values
    y_pred = df["label_pred"].astype(int).values

    acc = accuracy(y_true, y_pred)
    f1  = f1_macro(y_true, y_pred)
    print(f"Public Test — Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")

if __name__ == "__main__":
    main()
