# imports necessary libraries

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""this function is equivalent of the 'oneHotCatVars' function of the original code which modified to run on the latest
versions of pandas. This function converts categorical (text) columns into numeric 0/1 columns via one hot encoding"""

def one_hot_cat(df, cols):
    return pd.concat(
        [df.drop(columns=cols), pd.get_dummies(df[cols], dtype=np.int64)],
        axis=1,
    )


"""This function is loading the Adult data, drop rows with missing values, doing one hot encoding, normalize 
the continuous features, filter out samples from two dominant races: White or Black, splitting the training and 
test set and Separate features, labels, and sensitive attribute. It mimic the whole data loading and 
processing pipeline of the original codebase."""


def load_adult(path):
    data = pd.read_csv(path)
    for c in ["native-country", "workclass", "occupation"]:
        if c in data.columns:
            data[c] = data[c].replace(" ?", np.nan)

    if "education-num" in data.columns:
        data = data.drop(columns=["education-num"])  # redundant with 'education'
    data = data.dropna(how="any").reset_index(drop=True)

    cat_cols = list(set(data.columns) - set(data.describe().columns))
    for c in cat_cols:
        data[c] = data[c].astype("category")
    data = one_hot_cat(data, data.select_dtypes("category").columns)

    norm_cols = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    scaler = preprocessing.StandardScaler()
    data[norm_cols] = scaler.fit_transform(data[norm_cols])

    # restrict to the two largest race groups (binary sensitive attribute)
    data = data[(data["race_White"] == 1) | (data["race_Black"] == 1)].copy()

    train_df, test_df = train_test_split(data, test_size=0.25, random_state=0)

    def split_xya(df):
        x = df.drop(columns=["income", "race_White", "race_Black"])
        y = (df["income"].to_numpy() > 50000).astype(np.int64)
        a = df["race_Black"].to_numpy().astype(np.int64)  # 1 = Black
        return x, y, a

    return split_xya(train_df), split_xya(test_df)


""""This class is the equivalent of the ShelterOutcomeDataset class of the original code. Its wrapping
the dataset into dataloader."""

class FairnessDataset(Dataset):
    def __init__(self, X, Y, A):
        self.x = X.to_numpy().astype(np.float32)
        self.y = Y.astype(np.int64)
        self.a = A.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.a[i]


""" MLP model used for training. Same architecure like the original one"""

class MLP(nn.Module):
    def __init__(self, input_size, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)

""""Surrogate MLP for the transfer attack. Different MLP than the target MLP"""

class SurrogateMLP(nn.Module):
    def __init__(self, input_size, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)

"""The L_EOd function resposible for fairness loss as stated in the original paper. For each true label we get the total prediction
difference. Here |TPR_b - TPR_w| + |TNR_b - TNR_w| as we have two true labels."""

def eod_loss(log_probs, y, a):
    p = torch.exp(log_probs)

    masks = {
        "bp": (a == 1) & (y == 1),
        "wp": (a == 0) & (y == 1),
        "bn": (a == 1) & (y == 0),
        "wn": (a == 0) & (y == 0),
    }

    def gmean(values, mask):
        n = mask.sum()
        if n == 0:
            return torch.zeros((), device=values.device)
        return values[mask].sum() / n

    tpr_b = gmean(p[:, 1], masks["bp"])
    tpr_w = gmean(p[:, 1], masks["wp"])
    tnr_b = gmean(p[:, 0], masks["bn"])
    tnr_w = gmean(p[:, 0], masks["wn"])
    return torch.abs(tpr_b - tpr_w) + torch.abs(tnr_b - tnr_w)


"""Conventional L-inf PGD attack code that maximizes NLL."""

def pgd_attack_acc(model, x, y, eps, alpha=2 / 255, iters=40):
    ori = x.detach().clone()
    x = x.detach().clone()
    for _ in range(iters):
        x.requires_grad_(True)
        loss = F.nll_loss(model(x), y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x = x + alpha * x.grad.sign()
            x = ori + torch.clamp(x - ori, -eps, eps)
        x = x.detach()
    return x

"""Conventional NES black-box attack against NLL (per-example loss)."""

def nes_acc_attack(model, x, y, eps, iters=15, n_samples=20, sigma=1e-3, alpha=None):
    if alpha is None:
        alpha = max(eps / 5.0, 1e-3)
    x_orig = x.detach().clone()
    x_adv = x.detach().clone()

    for _ in range(iters):
        with torch.no_grad():
            grad_est = torch.zeros_like(x_adv)
            for _ in range(n_samples // 2):
                noise = torch.randn_like(x_adv)
                lp = F.nll_loss(model(x_adv + sigma * noise), y, reduction="none")
                lm = F.nll_loss(model(x_adv - sigma * noise), y, reduction="none")
                grad_est = grad_est + (lp - lm).view(-1, 1) * noise
            grad_est = grad_est / (n_samples * sigma)
            x_adv = x_adv + alpha * grad_est.sign()
            delta = torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = (x_orig + delta).detach()
    return x_adv

"""Conventional transfer black-box attack. PGD attack with surrogate MLP."""

def transfer_acc_attack(target_model, x, y, eps, surrogate):
    return pgd_attack_acc(surrogate, x, y, eps)


"""Fair attack version of PGD, NES, and transfer attacks. Here the EOd is maximized.""" 

def pgd_attack_fair(model, x, y, a, eps, alpha=2 / 255, iters=40):
    """L-inf PGD that maximizes the EOd gap."""
    ori = x.detach().clone()
    x = x.detach().clone()
    for _ in range(iters):
        x.requires_grad_(True)
        loss = eod_loss(model(x), y, a)
        model.zero_grad()
        if loss.requires_grad:
            loss.backward()
            with torch.no_grad():
                x = x + alpha * x.grad.sign()
                x = ori + torch.clamp(x - ori, -eps, eps)
        x = x.detach()
    return x


def nes_fair_attack(model, x, y, a, eps, iters=15, n_samples=20, sigma=1e-3, alpha=None):
    if alpha is None:
        alpha = max(eps / 5.0, 1e-3)
    x_orig = x.detach().clone()
    x_adv = x.detach().clone()

    for _ in range(iters):
        with torch.no_grad():
            grad_est = torch.zeros_like(x_adv)
            for _ in range(n_samples // 2):
                noise = torch.randn_like(x_adv)
                lp = eod_loss(model(x_adv + sigma * noise), y, a)
                lm = eod_loss(model(x_adv - sigma * noise), y, a)
                grad_est = grad_est + (lp - lm) * noise
            grad_est = grad_est / (n_samples * sigma)
            x_adv = x_adv + alpha * grad_est.sign()
            delta = torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = (x_orig + delta).detach()
    return x_adv


def transfer_fair_attack(target_model, x, y, a, eps, surrogate):
    return pgd_attack_fair(surrogate, x, y, a, eps)


"""Evaluating DP, TPR, TNR, FPR, FNR, ACR"""

def model_eval(actual, pred):
    if hasattr(actual, "detach"):
        actual = actual.detach().cpu().numpy()
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    actual = np.asarray(actual).ravel().astype(int)
    pred = np.asarray(pred).ravel().astype(int)

    TP = int(((actual == 1) & (pred == 1)).sum()) + 1
    TN = int(((actual == 0) & (pred == 0)).sum()) + 1
    FP = int(((actual == 0) & (pred == 1)).sum()) + 1
    FN = int(((actual == 1) & (pred == 0)).sum()) + 1
    return {
        "DP":  (TP + FP - 2) / (TP + TN + FP + FN - 4),
        "TPR": (TP - 1) / (TP + FN - 2),
        "TNR": (TN - 1) / (FP + TN - 2),
        "FPR": (FP - 1) / (FP + TN - 2),
        "FNR": (FN - 1) / (TP + FN - 2),
        "ACR": (TP + TN - 2) / (TP + TN + FP + FN - 4),
    }

"""Attack Selection"""

def apply_attack(model, x, y, a, eps, attack, surrogate=None):
    if eps == 0 or attack == "clean":
        return x
    if attack == "pgd_acc":      return pgd_attack_acc(model, x, y, eps)
    if attack == "nes_acc":      return nes_acc_attack(model, x, y, eps)
    if attack == "transfer_acc": return transfer_acc_attack(model, x, y, eps, surrogate)

    if attack == "pgd_fair":      return pgd_attack_fair(model, x, y, a, eps)
    if attack == "nes_fair":      return nes_fair_attack(model, x, y, a, eps)
    if attack == "transfer_fair": return transfer_fair_attack(model, x, y, a, eps, surrogate)

    raise ValueError(f"unknown attack: {attack}")

"""Evaluating for different eps"""

def evaluate_at_eps(model, loader, eps, attack, surrogate=None):
    model.eval()
    n_correct, n_total = 0, 0
    w_metrics, b_metrics = None, None

    for x, y, a in loader:
        x = x.to(device); y = y.to(device).long(); a = a.to(device)
        x_adv = apply_attack(model, x, y, a, eps, attack, surrogate)

        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)

        n_correct += (pred == y).sum().item()
        n_total += y.size(0)

        idx_b = (a == 1)
        idx_w = (a == 0)
        if idx_b.any() and idx_w.any():
            b_metrics = model_eval(y[idx_b], pred[idx_b])
            w_metrics = model_eval(y[idx_w], pred[idx_w])

    if w_metrics is None or b_metrics is None:
        return {"acc": n_correct / n_total, "DI": 0, "EOd": 0,
                "w_TPR": 0, "w_TNR": 0, "b_TPR": 0, "b_TNR": 0}

    DI = 100 * abs(w_metrics["DP"] - b_metrics["DP"])
    EOd = 100 * (abs(w_metrics["TNR"] - b_metrics["TNR"])
                 + abs(w_metrics["TPR"] - b_metrics["TPR"]))
    return {
        "acc":   n_correct / n_total,
        "DI":    DI,
        "EOd":   EOd,
        "w_TPR": 100 * w_metrics["TPR"],
        "w_TNR": 100 * w_metrics["TNR"],
        "b_TPR": 100 * b_metrics["TPR"],
        "b_TNR": 100 * b_metrics["TNR"],
    }

"""For swipping eps"""

def sweep(model, loader, epsilons, attack, surrogate=None):
    return [evaluate_at_eps(model, loader, e, attack, surrogate)
            for e in epsilons]


"""Training function. If use_adv = True for adversarial training only. If use_fairness = True for 
fairness adversarial training. Note that for eod_loss clean samples are used and for normal adversarial training loss
adversarial samples form PGD attacks had been used."""

def train_model(train_loader, input_size,
                use_fairness=False, use_adv=False,
                lambda_fair=0.4, lambda_adv_fair=0.4,
                eps_train=0.1, max_epoch=100, lr=0.01, momentum=0.8,
                verbose=True, seed=0):
    torch.manual_seed(seed)
    model = MLP(input_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, max_epoch + 1):
        model.train()
        running, n = 0.0, 0
        for x, y, a in train_loader:
            x = x.to(device); y = y.to(device).long(); a = a.to(device)
            optimizer.zero_grad()

            if use_adv:
                x_in = pgd_attack_acc(model, x, y, eps_train)
                fair_lambda = lambda_adv_fair
            else:
                x_in = x
                fair_lambda = lambda_fair

            out_normal = model(x)
            out = model(x_in)
            loss = F.nll_loss(out, y)
            if use_fairness:
                loss = loss + fair_lambda * eod_loss(out_normal, y, a)

            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            n += y.size(0)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}/{max_epoch}  loss = {running / n:.4f}")
    return model

"""Training surrogate model for transfer attack. """

def train_surrogate(train_loader, input_size, epochs=25, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = SurrogateMLP(input_size).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y, _a in train_loader:
            x = x.to(device); y = y.to(device).long()
            opt.zero_grad()
            loss = F.nll_loss(model(x), y)
            loss.backward()
            opt.step()
    return model


"""Main function"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to adult_reconstruction.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="adult_results.png")
    parser.add_argument("--out-attacks", default="adult_attacks.png")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading Adult...")
    (train_x, train_y, train_a), (test_x, test_y, test_a) = load_adult(args.csv)
    input_size = train_x.shape[1]
    print(f"  input_size = {input_size}, "
          f"|train| = {len(train_y)}, |test| = {len(test_y)}, device = {device}")

    train_ds = FairnessDataset(train_x, train_y, train_a)
    test_ds = FairnessDataset(test_x, test_y, test_a)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    epsilons = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]

    print("\nTraining surrogate (for transfer attacks)...")
    surrogate = train_surrogate(train_loader, input_size, seed=args.seed + 1)
    print("  surrogate trained.")

    configs = {
        "baseline": dict(use_fairness=False, use_adv=False),
        #"fair":     dict(use_fairness=True,  use_adv=False),
        "adv":      dict(use_fairness=False, use_adv=True),
        "adv+fair": dict(use_fairness=True,  use_adv=True),
    }

    # Each "attack" is a (acc_variant, fair_variant) pair.
    attack_pairs = {
        "PGD":      ("pgd_acc",      "pgd_fair"),
        "NES":      ("nes_acc",      "nes_fair"),
        "Transfer": ("transfer_acc", "transfer_fair"),
    }

    results = {}
    for cname, cfg in configs.items():
        print(f"\nTraining: {cname}")
        model = train_model(train_loader, input_size,
                            max_epoch=args.epochs, seed=args.seed, **cfg)
        results[cname] = {}
        for label, (acc_atk, fair_atk) in attack_pairs.items():
            print(f"  {label:9s}  -- {acc_atk}")
            acc_sweep = sweep(model, test_loader, epsilons, acc_atk,
                              surrogate=surrogate)
            print(f"  {label:9s}  -- {fair_atk}")
            fair_sweep = sweep(model, test_loader, epsilons, fair_atk,
                               surrogate=surrogate)
            results[cname][label] = {"acc": acc_sweep, "fair": fair_sweep}

    #summary tables 
    print("\nCLEAN (eps = 0) summary")
    print(f"{'config':<10}{'acc':>8}{'EOd':>8}{'DI':>8}")
    for cname, r in results.items():
        r0 = r["PGD"]["acc"][0]
        print(f"{cname:<10}{r0['acc']:>8.4f}{r0['EOd']:>8.2f}{r0['DI']:>8.2f}")

    print("\nFinal accuracy at MAX eps (under each attack's acc-variant)")
    header = f"{'attack':<10}" + "".join(f"{c:>12}" for c in configs)
    print(header)
    for label in attack_pairs:
        row = f"{label:<10}"
        for cname in configs:
            row += f"{results[cname][label]['acc'][-1]['acc']:>12.4f}"
        print(row)

    print("\nFinal EOd at MAX eps (under each attack's fair-variant)")
    print(header)
    for label in attack_pairs:
        row = f"{label:<10}"
        for cname in configs:
            row += f"{results[cname][label]['fair'][-1]['EOd']:>12.2f}"
        print(row)

    # headline plot: PGD-only 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"baseline": "tab:blue", "fair": "tab:green",
              "adv": "tab:orange", "adv+fair": "tab:red"}
    for cname, r in results.items():
        axes[0].plot(epsilons, [s["acc"] for s in r["PGD"]["acc"]],
                     marker="*", label=cname, color=colors[cname])
        axes[1].plot(epsilons, [s["EOd"] / 100 for s in r["PGD"]["fair"]],
                     marker="*", label=cname, color=colors[cname])
    axes[0].set(title="Accuracy under PGD-acc",
                xlabel=r"$\epsilon$", ylabel="Accuracy", ylim=(0, 1))
    axes[1].set(title="EOd gap under PGD-fair",
                xlabel=r"$\epsilon$", ylabel="EOd")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved headline plot to {args.out}")
    plt.close(fig)

    # attacks-comparison plot: 3 rows (one per attack) x 2 cols (acc, EOd)
    attack_labels = list(attack_pairs.keys())  # PGD, NES, Transfer
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for r_idx, label in enumerate(attack_labels):
        ax_acc = axes[r_idx, 0]
        ax_eod = axes[r_idx, 1]
        for cname, r in results.items():
            accs = [s["acc"] for s in r[label]["acc"]]
            eods = [s["EOd"] / 100 for s in r[label]["fair"]]
            ax_acc.plot(epsilons, accs, marker="*",
                        label=cname, color=colors[cname])
            ax_eod.plot(epsilons, eods, marker="*",
                        label=cname, color=colors[cname])
        ax_acc.set(title=f"{label} (acc-attack)  --  Accuracy",
                   xlabel=r"$\epsilon$", ylabel="Accuracy", ylim=(0, 1))
        ax_eod.set(title=f"{label} (fair-attack)  --  EOd",
                   xlabel=r"$\epsilon$", ylabel="EOd")
        for ax in (ax_acc, ax_eod):
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_attacks, dpi=200)
    print(f"Saved attacks-comparison plot to {args.out_attacks}")
    plt.close(fig)


if __name__ == "__main__":
    main()