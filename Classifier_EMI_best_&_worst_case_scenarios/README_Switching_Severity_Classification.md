# Physics-Informed Switching Severity Analysis (Per Device)

This script ranks switching scenarios per MOSFET device using a physics-informed score derived from the 13 EMI waveform targets. It labels each row as **Best**, **Neutral**, or **Worst**, and produces concise tables and plots for reporting.

1. Loads `merged_6_MOSFETs.csv` with a `Part_Number` column.
2. Ensures the 13 targets are numeric and drops rows with missing targets.
3. Standardizes the 13 targets per device (z-scores).
4. Computes a physics-informed **Severity_Score**:
   - Faster voltage/current rise/fall → worse (−z)
   - Larger over/undershoot → worse (+z)
   - Lower ringing frequency → worse (−z)
5. Labels scenarios using device-wise quantiles (default: Best = 10th %, Worst = 90th %).
6. Exports top-5 Best/Worst scenarios and visual summaries.
7. Optionally saves per-row contribution breakdowns.

## Inputs
- `INPUT_FILE`: path to `merged_6_MOSFETs.csv`
- Required columns:
  - `Part_Number`
  - 13 targets:
    `voltage_rise_time_pulse1/2`, `voltage_fall_time_pulse1/2`,
    `current_rise_time_pulse1/2`, `current_fall_time_pulse1/2`,
    `overshoot_pulse_1/2`, `undershoot_pulse_1/2`, `ringing_frequency_MHz`

## Key settings
- `BEST_Q = 0.10`, `WORST_Q = 0.90` (quantile thresholds)
- `SAVE_PER_ROW_CONTRIBUTIONS = True`
- `OUTPUT_DIR`: root folder for per-device artifacts

## How to run
```bash
python switching_severity_analysis.py
