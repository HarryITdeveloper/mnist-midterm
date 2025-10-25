import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import amp
import torch.backends.cudnn as cudnn


from .utils.seed import set_seed
from .utils.data import PairNPZDataset
from .models.siamese_compare import Model as CompareNet, count_params

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
    ap.add_argument("--label_smoothing", type=float, default=0.05)   # 二分类平滑标签
    ap.add_argument("--early_stop_patience", type=int, default=10)    # 提高耐心
    args = ap.parse_args()

    cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompareNet().to(device)
    n_params = int(count_params(model))

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path   = os.path.join(args.data_dir, "val.npz")
    train_ds = PairNPZDataset(train_path, is_train=True)
    val_ds   = PairNPZDataset(val_path, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best = {"acc":0.0, "f1":0.0, "epoch":-1}
    patience, bad = args.early_stop_patience, 0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for xa, xb, y in pbar:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            y  = y.to(device).float()

            # --- label smoothing for binary ---
            # 将 {0,1} 目标平滑到 {ε/2, 1-ε/2}
            eps = args.label_smoothing
            y_s = y * (1 - eps) + 0.5 * eps

            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with amp.autocast('cuda'):
                    logit = model(xa, xb)
                    loss  = criterion(logit, y_s)
                scaler.scale(loss).backward()
                # 先反缩放，再裁剪，再 step
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                logit = model(xa, xb)
                loss  = criterion(logit, y_s)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()


            pbar.set_postfix(loss=float(loss.item()))

        # 一轮结束：验证 + 调度器 step
        acc, f1 = evaluate(model, val_loader, device)
        print(f"[Val] epoch={epoch} acc={acc:.4f} f1_macro={f1:.4f}")

        scheduler.step()

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
