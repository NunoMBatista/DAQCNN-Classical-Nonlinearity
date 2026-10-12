"""TissueMNIST linear probes: standardised multinomial logistic regression (C=1),
macro one-vs-rest AUC. A topology's 45 channels are its 9 <Z_i> followed by the
36 <Z_i Z_j>. Three modes:

  topologies       digital cache, Z and Z+ZZ of each of the 8 topologies and two
                   four-kernel sets, on a subsample (15k train / 8k test images),
                   scored on test. The Tissue digital capacity-sweep and end-to-end
                   topologies are this probe's winners.
                   -> outputs/paper_results/linear_probing/tissue_mnist_topology_probe.csv
  analog-topology  validation-selected single-kernel topology for the analog
                   sources (Analog-Z and Analog-ZZ), AUC on validation and test,
                   selection on validation. Prints only.
  all-encodings    one cache (--cache digital or analog), all 8 topologies, Z and
                   Z+ZZ, full splits, test AUC. Rows are written after every topology.
                   -> outputs/paper_results/linear_probing/tissue_mnist_probe_<cache>.csv

Every mode loads whole TissueMNIST caches (up to 47 GB): run on a large-memory node,
never locally.

    python experiments/probe_tissue.py topologies
    python experiments/probe_tissue.py analog-topology
    python experiments/probe_tissue.py all-encodings --cache digital
"""
import argparse, os, sys, json, csv, gc, time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.kernel_mapping import build_kernel_to_channels_map
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset, load_cached_quantum_dataset,
)

QD = "data/quantum_datasets"
CACHES = {
    "digital": f"{QD}/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz.npz",
    "analog":  f"{QD}/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz_analog.npz",
}
TOPOS = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
OUTDIR = "outputs/paper_results/linear_probing"

# topologies mode: subsample sizes
N_TR, N_TE = 15000, 8000
MAX_STACK_TR, MAX_STACK_TE = 30000, 15000  # rows read before subsampling
SEED = 0


def probe(Xtr, ytr, evals):
    """Fit on train; macro one-vs-rest AUC on each (X, y) in evals, as a list."""
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1).fit(sc.transform(Xtr), ytr)
    aucs = []
    for X, y in evals:
        proba = clf.predict_proba(sc.transform(X.reshape(X.shape[0], -1)))
        aucs.append(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    return aucs


# ---------------------------------------------------------------- topologies

def stack(loader, max_n=None):
    """Loader -> numpy, stopping once max_n rows are read."""
    xs, ys, n = [], [], 0
    for x, y in loader:
        xs.append(x.float()); ys.append(y.long().squeeze()); n += len(x)
        if max_n and n >= max_n:
            break
    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


def subsample(X, y, n, rng):
    if len(y) <= n:
        return X, y
    idx = rng.choice(len(y), n, replace=False)
    return X[idx], y[idx]


def run_topologies():
    t0 = time.time()
    print("loading cache (all 8 topologies)...", flush=True)
    tr, _, te, _, meta = load_cached_quantum_dataset(CACHES["digital"], batch_size=4096, num_workers=0,
                                                     requested_kernels=TOPOS)
    Xtr, ytr = stack(tr, MAX_STACK_TR); Xte, yte = stack(te, MAX_STACK_TE)
    print(f"  stacked train{Xtr.shape} test{Xte.shape} in {time.time()-t0:.0f}s "
          f"classes={sorted(set(ytr.tolist()))}", flush=True)
    rng = np.random.default_rng(SEED)
    Xtr, ytr = subsample(Xtr, ytr, N_TR, rng)
    Xte, yte = subsample(Xte, yte, N_TE, rng)
    print(f"  subsampled: train{Xtr.shape} test{Xte.shape}", flush=True)

    rows = []
    print(f"\n{'topology':12} {'Z(9)':>8} {'ZZ(45)':>8}")
    for k, topo in enumerate(TOPOS):
        blk = slice(k * 45, (k + 1) * 45)
        z = probe(Xtr[:, k * 45:k * 45 + 9], ytr, [(Xte[:, k * 45:k * 45 + 9], yte)])[0]
        zz = probe(Xtr[:, blk], ytr, [(Xte[:, blk], yte)])[0]
        print(f"  {topo:12} {z:8.4f} {zz:8.4f}")
        rows.append({"topology": topo, "auc_z": round(z, 4), "auc_zz": round(zz, 4)})

    best_zz = sorted(rows, key=lambda r: -r["auc_zz"])[:4]
    sets = {
        "best4_zz": [r["topology"] for r in best_zz],
        "g_c_h_kings": ["grid", "chain", "horizontal", "kings"],
    }
    print(f"\n4-kernel ZZ sets:")
    for label, ts in sets.items():
        idx = np.concatenate([np.arange(TOPOS.index(t) * 45, (TOPOS.index(t) + 1) * 45) for t in ts])
        zz4 = probe(Xtr[:, idx], ytr, [(Xte[:, idx], yte)])[0]
        print(f"  {label:14} {ts} -> {zz4:.4f}")
        rows.append({"topology": f"4k:{label}:{'+'.join(ts)}", "auc_z": "", "auc_zz": round(zz4, 4)})

    out = f"{OUTDIR}/tissue_mnist_topology_probe.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topology", "auc_z", "auc_zz"]); w.writeheader(); w.writerows(rows)
    bz = max((r for r in rows if r["auc_z"] != ""), key=lambda r: r["auc_z"])
    bzz = max((r for r in rows if not str(r["topology"]).startswith("4k")), key=lambda r: r["auc_zz"])
    print(f"\n>>> BEST digital_z topology: {bz['topology']} ({bz['auc_z']})")
    print(f">>> BEST digital_zz topology: {bzz['topology']} ({bzz['auc_zz']})")
    print(f"saved {out}")


# ----------------------------------------------------------- analog-topology

ENCODINGS = [("Analog-Z", "analog", False), ("Analog-ZZ", "analog", True)]


def collect(loader):
    xs, ys = [], []
    for x, y in loader:
        x = x.numpy() if isinstance(x, torch.Tensor) else x
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        xs.append(x.reshape(x.shape[0], -1))
        ys.append(np.asarray(y).squeeze())
    return np.concatenate(xs), np.concatenate(ys)


def cfg(topo, enc, corr):
    """Minimal config for the sidecar lookup of a single-topology cache."""
    return {"dataset": {"name": "tissue_mnist", "data_root": "./data", "batch_size": 2000,
                        "num_workers": 0, "download": False, "color_space": "GRAYSCALE"},
            "model": {"num_classes": 8, "kernel_size": 3, "stride": 3,
                      "kernel_topology_names": [topo], "scaling_factor": 1.0,
                      "evolution_time": 2.5, "mode": "trotter", "encoding_mode": enc,
                      "include_correlators": corr, "quantum_device": "default.qubit",
                      "classical_device": "auto"}}


def run_analog_topology():
    for name, enc, corr in ENCODINGS:
        print(f"\n===== {name} =====", flush=True)
        rows = []
        for t in TOPOS:
            cp = find_cached_quantum_dataset(cfg(t, enc, corr))
            if cp is None:
                print(f"  {t:<11s} no cache", flush=True)
                continue
            tr, va, te, _, _ = load_cached_quantum_dataset(
                cp, batch_size=2000, num_workers=0, requested_kernels=[t])
            Xtr, ytr = collect(tr); Xva, yva = collect(va); Xte, yte = collect(te)
            va_auc, te_auc = probe(Xtr, ytr, [(Xva, yva), (Xte, yte)])
            rows.append((t, va_auc, te_auc))
            print(f"  {t:<11s} val={va_auc:.4f} test={te_auc:.4f}", flush=True)
            del Xtr, Xva, Xte
        if rows:
            vb = max(rows, key=lambda r: r[1])
            tb = max(rows, key=lambda r: r[2])
            print(f"  => {name} VAL-selected={vb[0]} (val={vb[1]:.4f} test={vb[2]:.4f}) | "
                  f"test-selected={tb[0]} (test={tb[2]:.4f}) | "
                  f"inflation={tb[2]-vb[2]:+.4f}", flush=True)


# ------------------------------------------------------------- all-encodings

def write_csv(rows, enc, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topology", f"{enc}_z", f"{enc}_zz"])
        w.writeheader()
        w.writerows(rows)


def run_all_encodings(enc):
    path = CACHES[enc]
    out_csv = f"{OUTDIR}/tissue_mnist_probe_{enc}.csv"
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    print(f"===== {enc} :: {os.path.basename(path)} =====", flush=True)
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["metadata"]))
    k2c = build_kernel_to_channels_map(meta["channel_kernel_map"])
    Xtr_all = data["train_features"]
    ytr = np.asarray(data["train_labels"]).squeeze()
    Xte_all = data["test_features"]
    yte = np.asarray(data["test_labels"]).squeeze()
    print(f"  loaded train{Xtr_all.shape} test{Xte_all.shape} in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for topo in TOPOS:
        ch = k2c[topo]
        z_ch, zz_ch = ch[:9], ch
        tt = time.time()
        z = probe(Xtr_all[:, z_ch], ytr, [(Xte_all[:, z_ch], yte)])[0]
        zz = probe(Xtr_all[:, zz_ch], ytr, [(Xte_all[:, zz_ch], yte)])[0]
        rows.append({"topology": topo, f"{enc}_z": round(z, 4), f"{enc}_zz": round(zz, 4)})
        write_csv(rows, enc, out_csv)
        print(f"  {topo:12} z={z:.4f} zz={zz:.4f}  "
              f"[{len(rows)}/{len(TOPOS)} done, {time.time()-tt:.0f}s]", flush=True)

    del Xtr_all, Xte_all, data
    gc.collect()
    print(f"\n{enc} means: "
          f"z={np.mean([r[f'{enc}_z'] for r in rows]):.4f} "
          f"zz={np.mean([r[f'{enc}_zz'] for r in rows]):.4f}")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["topologies", "analog-topology", "all-encodings"])
    ap.add_argument("--cache", choices=list(CACHES), help="all-encodings mode: which cache")
    args = ap.parse_args()
    if args.mode == "topologies":
        run_topologies()
    elif args.mode == "analog-topology":
        run_analog_topology()
    else:
        if args.cache is None:
            ap.error("all-encodings needs --cache digital or --cache analog")
        run_all_encodings(args.cache)
