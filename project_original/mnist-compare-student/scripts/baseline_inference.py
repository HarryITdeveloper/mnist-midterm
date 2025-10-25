import os, argparse, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from .utils.data import PairNPZDataset
from .models.siamese_compare import Model as CompareNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="directory containing test_public.npz")
    ap.add_argument("--ckpt", type=str, required=True, help="path to model.pt")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out", type=str, default="./pred_public.csv")
    ap.add_argument("--private", type=str, default="False")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    if args.private == "True":
        test_path = os.path.join(args.data_dir, "test_private.npz")
    else:
        test_path = os.path.join(args.data_dir, "test_public.npz")
    ds = PairNPZDataset(test_path, is_train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompareNet().to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    ids_all, preds_all = [], []
    with torch.no_grad():
        for xa, xb, ids in loader:
            xa = xa.to(device); xb = xb.to(device)
            logit = model(xa, xb)
            prob = torch.sigmoid(logit)
            pred = (prob >= args.thresh).long().cpu().numpy()
            ids_all.extend(list(ids))
            preds_all.extend(pred.tolist())

    df = pd.DataFrame({"id": ids_all, "label": preds_all})
    df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
