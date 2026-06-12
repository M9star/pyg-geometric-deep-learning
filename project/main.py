"""
Capstone CLI — Graph Classifier
================================
Train / evaluate a GNN graph classifier on any TUDataset.

    python project/main.py train --dataset PROTEINS --model gin
    python project/main.py train --dataset MUTAG --model gat --epochs 150
    python project/main.py evaluate --run runs/PROTEINS_gin

Outputs (model checkpoint, metrics, plots) go to runs/<dataset>_<model>/.
"""

import argparse
import json
import os

import torch

from src.data import load_dataset, make_loaders
from src.engine import (collect_embeddings, collect_predictions, evaluate,
                        train_one_epoch)
from src.models import GraphClassifier
from src.viz import plot_confusion_matrix, plot_curves, plot_tsne

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = load_dataset(args.dataset)
    print(f"{args.dataset}: {len(dataset)} graphs | "
          f"{dataset.num_features} features | {dataset.num_classes} classes | device={device}")
    train_loader, val_loader, test_loader = make_loaders(
        dataset, batch_size=args.batch_size, seed=args.seed)

    model = GraphClassifier(dataset.num_features, args.hidden,
                            dataset.num_classes, kind=args.model,
                            num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    run_dir = os.path.join(RUNS, f"{args.dataset}_{args.model}")
    os.makedirs(run_dir, exist_ok=True)

    history = {"loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
    best_val, best_state, best_test = 0.0, None, 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        tr = evaluate(model, train_loader, device)
        va = evaluate(model, val_loader, device)
        te = evaluate(model, test_loader, device)
        history["loss"].append(loss)
        history["train_acc"].append(tr)
        history["val_acc"].append(va)
        history["test_acc"].append(te)

        if va >= best_val:
            best_val, best_test = va, te
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {loss:.4f} | "
                  f"train {tr:.3f} | val {va:.3f} | test {te:.3f}")

    print(f"\nBest val {best_val:.3f} -> test {best_test:.3f}")

    # restore best model for saving + plots
    model.load_state_dict(best_state)

    # --- save checkpoint + config ---
    ckpt = {
        "state_dict": model.state_dict(),
        "config": vars(args),
        "in_dim": dataset.num_features,
        "num_classes": dataset.num_classes,
    }
    torch.save(ckpt, os.path.join(run_dir, "model.pt"))

    metrics = {"best_val_acc": best_val, "test_acc_at_best_val": best_test,
               "final_test_acc": history["test_acc"][-1]}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- plots ---
    title = f"{args.dataset} · {args.model.upper()}"
    plot_curves(history, os.path.join(run_dir, "curves.png"), title)
    y_true, y_pred = collect_predictions(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, dataset.num_classes,
                          os.path.join(run_dir, "confusion_matrix.png"),
                          f"Confusion matrix — {title}")
    embs, labels = collect_embeddings(model, test_loader, device)
    plot_tsne(embs, labels, os.path.join(run_dir, "embeddings_tsne.png"),
              f"Graph embeddings — {title}")

    print(f"\nAll outputs saved to: {os.path.relpath(run_dir, HERE)}/")


def cmd_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(args.run, "model.pt"), map_location=device,
                      weights_only=False)
    cfg = ckpt["config"]

    dataset = load_dataset(cfg["dataset"])
    _, _, test_loader = make_loaders(dataset, batch_size=cfg["batch_size"],
                                     seed=cfg["seed"])
    model = GraphClassifier(ckpt["in_dim"], cfg["hidden"], ckpt["num_classes"],
                            kind=cfg["model"], num_layers=cfg["layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    acc = evaluate(model, test_loader, device)
    print(f"Loaded {args.run}\nTest accuracy: {acc:.3f}")


def build_parser():
    p = argparse.ArgumentParser(description="GNN graph classifier")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("--dataset", default="PROTEINS")
    t.add_argument("--model", choices=["gin", "gcn", "gat"], default="gin")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--lr", type=float, default=0.005)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a saved run")
    e.add_argument("--run", required=True, help="path to runs/<name> directory")
    e.set_defaults(func=cmd_evaluate)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
