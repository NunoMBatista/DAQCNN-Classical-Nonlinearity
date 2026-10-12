# Cluster job scripts

The SLURM scripts used to run the heavy jobs (TissueMNIST and the BreastMNIST
classical-nonlinear searches) on an HPC cluster. They are kept as a record of
how those results were produced and are not meant to run unchanged: they assume
the repository at `~/MoT-DAQCNN`, a virtual environment at `.venv`, and a
placeholder account (`<your-account>`). The ZZ-cache jobs need about 200 GB of
memory.

| Script | Job |
|---|---|
| `mcc_tissue_e2e.sh`, `mcc_tissue_e2e_launch.sh` | TissueMNIST end-to-end HP searches (digital and classical rows), one job per model |
| `mcc_tissue_analog_e2e.sh` | TissueMNIST end-to-end analog-ZZ rows |
| `mcc_tissue_analog_capsweep.sh` | analog sources of the TissueMNIST capacity sweep |
| `mcc_tissue_classical_cache.sh` | TissueMNIST poly-2 / RFF feature caches |
| `mcc_tissue_analog_merge.sh` | merge the TissueMNIST analog chunks into one cache |
| `mcc_tissue_analog_verify.sh` | value check of the merged cache (NaN, Inf, range) |
| `mcc_tissue_derive_z.sh` | analog-Z cache sliced from the analog-ZZ cache |
| `mcc_tissue_analog_topo_probe.sh` | validation-selected analog topology probe |
| `mcc_tissue_all_enc_probe.sh` | full-data TissueMNIST probe of one cache |
| `mcc_breast_classical.sh` | BreastMNIST poly-2 / RFF end-to-end HP searches |
