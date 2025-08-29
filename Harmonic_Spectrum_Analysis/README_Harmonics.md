# Harmonics Analysis and EMI Risk Flags

This module ingests per-measurement harmonic spectra, cleans outliers, computes EMI-relevant features (THD, decay slope, band power, peak counts), and exports both per-file and master feature tables. It also includes notebooks/snippets that reconstruct time-domain waveforms from the first N harmonics and plot spectra.

---

## Inputs

- Folder: `Harmonics_Data/`
- Files: `harmonics_*.csv`

Each CSV row should contain (at minimum):

- `Measurement` (e.g., `I(V5)` or `V(node)`)
- Harmonic triplets:
  - `Frequency_k_Hz` for k = 1…K
  - `Mag_k` (linear amplitude units)
  - `Phase_k` (degrees)
- Optional simulation descriptors (kept if present):  
  `Vbus, Rg, Ls4, Ls5, Ls6, Ls7, Ls8, Ls9, Ls10, Ls11`

> Fundamental is assumed to be `k = 1`.

---

## What the Script Does

1. **Locate files**  
   Scans `Harmonics_Data/` for `harmonics_*.csv`.

2. **IQR outlier filtering (optional)**  
   - Applies Tukey IQR rule (default *k* = 1.5) to chosen columns (`IQR_COLS`, default `["Mag_1"]`).  
   - Drops any row that is an outlier in **any** selected column.  
   - Saves counts per file to `processed/iqr_summary.csv`.

3. **Feature computation per row**
   - **THD_percent**  
     \[
     \mathrm{THD}\% = 100 \times \frac{\sqrt{\sum_{k=2}^{K} \mathrm{Mag}_k^2}}{\mathrm{Mag}_1}
     \]
   - **dB relative to fundamental**  
     \[
     \mathrm{dB}_k = 20\log_{10}\left(\frac{\mathrm{Mag}_k}{\mathrm{Mag}_1}\right)
     \]
   - **Noise floor (dB)**: median of \(\mathrm{dB}_k\) for \(k \ge 30\).  
   - **DecaySlope_dB_per_decade**: least-squares slope of \(\mathrm{dB}_k\) vs \(\log_{10}(k)\) for \(k=3..30\).  
   - **Band power (dB)** in three EMI bands (default):  
     - 100 kHz–3 MHz  
     - 3–10 MHz  
     - 10–30 MHz  
     Computed by summing magnitudes (relative to fundamental) within band and converting to dB.
   - **PeakCount_gt_floor+10dB** and **PeakMax_dB**: local peaks above noise floor + 10 dB (for \(k \ge 3\)).

4. **EMI risk flags**  
   Adds boolean/classification flags using configurable thresholds:
   - `THD_percent > 150`
   - `DecaySlope_dB_per_decade > -4`
   - `PeakMax_dB > -8`
   - `PeakCount_gt_floor+10dB ≥ 4`
   - Band power limits (default):
     - `100000-3000000Hz > -15 dB`
     - `3000000-10000000Hz > -12 dB`
     - `10000000-30000000Hz > -10 dB`
   The `EMI_Flag` column stores a semicolon-joined reason string; `EMI_Risky` is True/False.

5. **Exports**
   - Cleaned copy of each input: `processed/<file>_clean.csv`
   - Per-file features: `processed/<file>_features.csv`
   - IQR summary: `processed/iqr_summary.csv`
   - Master feature table (all files): `processed/emi_master_features.csv`

---

## Configuration (top of script)

```python
ROOT = Path("Harmonics_Data")
PATTERN = "harmonics_*.csv"
OUTDIR = ROOT / "processed"

IQR_K = 1.5
IQR_COLS = ["Mag_1"]

BANDS = ((1e5, 3e6), (3e6, 10e6), (10e6, 30e6))

THR = dict(
  thd=150.0, slope=-4.0, peak_max=-8.0, peak_count=4,
  band_db={
    "100000-3000000Hz": -15,
    "3000000-10000000Hz": -12,
    "10000000-30000000Hz": -10
  }
)
