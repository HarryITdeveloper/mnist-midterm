import os, argparse, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--pred', type=str, required=True)
    ap.add_argument('--test_file', type=str, default='test_public.npz', help='npz file to validate ids against')
    args = ap.parse_args()

    import numpy as np
    arr = np.load(os.path.join(args.data_dir, args.test_file), allow_pickle=False)
    ids = set(map(str, arr['id']))

    df = pd.read_csv(args.pred)
    assert set(df.columns) == {'id','label'}, "CSV must have exactly two columns: id,label"
    if df['id'].duplicated().any():
        dups = df[df['id'].duplicated()]['id'].unique()[:5]
        raise SystemExit(f"Duplicate ids in predictions, e.g. {dups}")

    pred_ids = set(map(str, df['id']))
    missing = ids - pred_ids
    extra = pred_ids - ids
    if missing:
        print(f"ERROR: {len(missing)} ids missing (e.g. {list(sorted(missing))[:5]})")
    if extra:
        print(f"ERROR: {len(extra)} extra ids not in test set (e.g. {list(sorted(extra))[:5]})")
    if not missing and not extra:
        print("OK: id coverage matches the test set.")

    if df['label'].isna().any():
        raise SystemExit("Found NaN labels.")
    if not set(df['label'].unique()).issubset({0,1}):
        raise SystemExit("Labels must be 0 or 1.")
    print("CSV format looks good.")

if __name__ == "__main__":
    main()
