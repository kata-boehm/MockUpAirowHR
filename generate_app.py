"""
generate_app.py
---------------
Reads model result pkl files and regenerates src/App.jsx
with the data for the chosen session embedded.

USAGE:
  1. Set the paths to your pkl files below
  2. Set the SESSION name you want to display
  3. Run:  python generate_app.py
  4. Refresh your browser at http://localhost:5173
"""

import pickle
import json
import numpy as np
import re
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

BILSTM_PKL  = r"data\bilstm_results.pkl"
NN_PKL      = r"data\nn_results.pkl"
XGBOOST_PKL = r"data\xgboost_results.pkl"

# Session to display — must exist in all three pkl files
# Run with SESSION = None first to print available sessions
SESSION = "12069656901_Schwitzen_im_Sitzen"


# Path to your App.jsx (relative to this script, or absolute)
APP_JSX = Path(__file__).parent / "src" / "App.jsx"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_session(pkl_path, session_name):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    results = data["results"]
    match = next((r for r in results if r["session"] == session_name), None)
    if match is None:
        available = [r["session"] for r in results]
        raise ValueError(f"Session '{session_name}' not found.\nAvailable sessions:\n" +
                         "\n".join(f"  {s}" for s in sorted(available)))
    return match

def get_boundaries(y_true):
    return [int(i) for i in np.where(np.array(y_true) == 1)[0].tolist()]

def get_predictions(y_pred):
    return [int(i) for i in np.where(np.array(y_pred) == 1)[0].tolist()]

def build_dual_data(df, step=5):
    """Downsample HR and power to every `step` seconds."""
    hr_col  = "heart_rate" if "heart_rate" in df.columns else "HR"
    pwr_col = "power_roll_avg" if "power_roll_avg" in df.columns else "power"
    result = []
    for i in range(0, len(df), step):
        hr  = df[hr_col].iloc[i]
        pwr = df[pwr_col].iloc[i]
        if not np.isnan(hr) and not np.isnan(pwr):
            result.append({"t": int(i), "hr": round(float(hr), 1), "pwr": round(float(pwr), 1)})
    return result

def session_sport(session_name):
    """Guess sport type from session name — adjust if needed."""
    name = session_name.lower()
    if "row" in name or "ruder" in name:
        return "ROWING"
    elif "cycl" in name or "bike" in name or "rad" in name:
        return "CYCLING"
    else:
        return "SESSION"

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # If SESSION is None, just print available sessions and exit
    if SESSION is None:
        with open(BILSTM_PKL, "rb") as f:
            data = pickle.load(f)
        sessions = sorted(r["session"] for r in data["results"])
        print("Available sessions:")
        for s in sessions:
            print(f"  {s}")
        return

    print(f"Loading session: {SESSION}")

    bilstm_r  = load_session(BILSTM_PKL,  SESSION)
    ffnn_r    = load_session(NN_PKL,      SESSION)
    xgboost_r = load_session(XGBOOST_PKL, SESSION)

    # Extract data
    true_boundaries = get_boundaries(bilstm_r["y_true"])
    bilstm_preds    = get_predictions(bilstm_r["y_pred"])
    ffnn_preds      = get_predictions(ffnn_r["y_pred"])
    xgboost_preds   = get_predictions(xgboost_r["y_pred"])

    df       = bilstm_r["df"]
    duration = len(df)
    dual_data = build_dual_data(df)

    # Session metadata
    athlete  = bilstm_r.get("athlete", "Unknown")
    sport    = session_sport(SESSION)
    dur_min  = round(duration / 60)

    # F-beta scores
    fb_bilstm  = round(bilstm_r.get("f_beta", 0), 3)
    fb_ffnn    = round(ffnn_r.get("f_beta", 0), 3)
    fb_xgboost = round(xgboost_r.get("f_beta", 0), 3)

    print(f"  Duration:        {duration}s ({dur_min} min)")
    print(f"  True boundaries: {len(true_boundaries)}")
    print(f"  BiLSTM preds:    {len(bilstm_preds)}  (F_β={fb_bilstm})")
    print(f"  FFNN preds:      {len(ffnn_preds)}  (F_β={fb_ffnn})")
    print(f"  XGBoost preds:   {len(xgboost_preds)}  (F_β={fb_xgboost})")

    # Serialize to JSON strings for embedding
    j = lambda x: json.dumps(x, separators=(',', ':'))

    true_boundaries_js = j(true_boundaries)
    bilstm_preds_js    = j(bilstm_preds)
    ffnn_preds_js      = j(ffnn_preds)
    xgboost_preds_js   = j(xgboost_preds)
    dual_data_js       = j(dual_data)
    duration_js        = str(duration)

    # Read existing App.jsx
    if not APP_JSX.exists():
        print(f"ERROR: App.jsx not found at {APP_JSX}")
        return

    content = APP_JSX.read_text(encoding="utf-8")

    # Replace data constants using regex
    replacements = [
        (r'const TRUE_BOUNDARIES\s*=\s*\[.*?\];',
         f'const TRUE_BOUNDARIES  = {true_boundaries_js};'),
        (r'const DURATION\s*=\s*\d+;',
         f'const DURATION = {duration_js};'),
        (r'const DUAL_DATA\s*=\s*\[.*?\];',
         f'const DUAL_DATA = {dual_data_js};'),
        # MODEL_PREDS block (multiline)
        (r'const MODEL_PREDS\s*=\s*\{[^}]*\};',
         f'const MODEL_PREDS = {{\n'
         f'  bilstm:  {bilstm_preds_js},\n'
         f'  xgboost: {xgboost_preds_js},\n'
         f'  ffnn:    {ffnn_preds_js},\n'
         f'}};'),
    ]

    for pattern, replacement in replacements:
        new_content, n = re.subn(pattern, replacement, content, flags=re.DOTALL)
        if n == 0:
            print(f"WARNING: pattern not matched — {pattern[:60]}")
        else:
            content = new_content

    # Replace header metadata (session name, sport, athlete, duration)
    session_short = SESSION.split("_")[-1] if "_" in SESSION else SESSION
    content = re.sub(
        r'ROWING · \w+|CYCLING · \w+|SESSION · \w+',
        f'{sport} · {session_short}',
        content
    )
    content = re.sub(r'ATH_\d+|Athlete\d+', athlete, content)
    content = re.sub(r'\d+ min</b>', f'{dur_min} min</b>', content)

    # Write back
    APP_JSX.write_text(content, encoding="utf-8")
    print(f"\nApp.jsx updated successfully → {APP_JSX}")
    print("Refresh your browser at http://localhost:5173")


if __name__ == "__main__":
    main()
