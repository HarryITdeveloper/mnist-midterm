import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from .utils.seed import set_seed
from .utils.data import PairNPZDataset
from .models.simple_compare_cnn import CompareNet, count_params

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xa, xb, y in loader:
            xa = xa.to(device); xb = xb.to(device); y = y.to(device).float()
            logit = model(xa, xb)
            prob = torch.sigmoid(logit)
            pred = (prob >= 0.5).long()
            ys.append(y.long().cpu().numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = (y_true == y_pred).mean().item()
    # macro-F1
    f1s = []
    for cls in [0,1]:
        tp = np.sum((y_true==cls) & (y_pred==cls))
        fp = np.sum((y_true!=cls) & (y_pred==cls))
        fn = np.sum((y_true==cls) & (y_pred!=cls))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    f1_macro = float(np.mean(f1s))
    return acc, f1_macro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./outputs/baseline")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompareNet(feat_dim=128).to(device)
    n_params = int(count_params(model))

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path   = os.path.join(args.data_dir, "val.npz")
    train_ds = PairNPZDataset(train_path, is_train=True)
    val_ds   = PairNPZDataset(val_path, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best = {"acc":0.0, "f1":0.0, "epoch":-1}
    patience, bad = 3, 0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for xa, xb, y in pbar:
            xa = xa.to(device); xb = xb.to(device); y = y.to(device).float()
            logit = model(xa, xb)
            loss = criterion(logit, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=float(loss.item()))

        acc, f1 = evaluate(model, val_loader, device)
        print(f"[Val] epoch={epoch} acc={acc:.4f} f1_macro={f1:.4f}  (params={n_params})")

        if acc > best["acc"]:
            best = {"acc":acc, "f1":f1, "epoch":epoch}
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
            with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
                json.dump({"best_val_acc":acc, "best_val_f1":f1, "best_epoch":epoch, "params":n_params}, f, indent=2)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"Best @ epoch {best['epoch']}: acc={best['acc']:.4f}, f1_macro={best['f1']:.4f}, params={n_params}")

if __name__ == "__main__":
    main()
