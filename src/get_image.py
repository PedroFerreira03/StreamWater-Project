"""
simulate_and_plot.py
────────────────────
1. Load full dataset + subgroup variables
2. Mask Aug 1-14 2024 consumption & cumulative for the two target contacts
   (simulating missing meter readings)
3. Run BatchImputationEngine on contacts belonging to the same
   (calibre, tipo_consumo) pairs as the target contacts only
4. Plot three lines for each target contact over Aug 1-14:
     • Blue  – original real values
     • Red   – engine-imputed values  
     • Yellow– corrected / normalised values 
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from collections import defaultdict
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INPUT_CSV  = "./Data/input_ts.csv"
VARIABLES  = "./variables/variables.pkl"
OUTPUT_DIR = "images"

CONTACTS  = ["LC41848", "LC42435"]
GROUPS    = ["Comércio", "Rega-Doméstico"]
CALIBRE   = ["15", "20"]
AUG_START = pd.Timestamp("2024-08-01")
AUG_END   = pd.Timestamp("2024-08-14")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH IMPUTATION ENGINE  
# ══════════════════════════════════════════════════════════════════════════════
class BatchImputationEngine:
    def __init__(self, subgroups_list, idx2pair, pair2idx,
                 warmup_contact=0, warmup_subgroup=0, weight_contact=0.7,
                 ewma_alpha=0.1, contact_params=None):
        self.subgroups_list  = subgroups_list
        self.idx2pair        = idx2pair
        self.pair2idx        = pair2idx
        self.warmup_contact  = warmup_contact
        self.warmup_subgroup = warmup_subgroup
        self.weight_contact  = weight_contact
        self.ewma_alpha      = ewma_alpha
        self.contact_params  = contact_params or {}

        _def_stats = lambda: {
            'pop_mean': None, 'ewma_mean': None,
            'pop_var':  None, 'ewma_var':  None,
            'ewma_std': None, 'count': 0
        }
        self.subgroup_stats     = defaultdict(_def_stats)
        self.contact_stats      = defaultdict(_def_stats)
        self.contact_pair_stats = defaultdict(_def_stats)
        self.pair_stats         = defaultdict(_def_stats)

        self.last_real_cumulative = {}
        self.last_real_meter_id   = {}
        self.last_real_idx        = {}

        self.subgroup_lookup = {}
        self._build_subgroup_lookup()

    def _build_subgroup_lookup(self):
        for pair_idx, subgroups in enumerate(self.subgroups_list):
            for subgroup_idx, subgroup in enumerate(subgroups):
                for month in subgroup['months']:
                    for day in subgroup['days']:
                        for hour in subgroup['hours']:
                            self.subgroup_lookup[(pair_idx, month, day, int(hour))] = subgroup_idx

    def _get_subgroup_index(self, pair_idx, month, day, hour):
        return self.subgroup_lookup.get((pair_idx, month, day, int(hour)), None)

    def _get_pair_from_row(self, row):
        if 'calibre' in row.index and 'tipo_consumo' in row.index:
            c, t = row['calibre'], row['tipo_consumo']
            if pd.notna(c) and pd.notna(t):
                return (c, t)
        return None

    def _update_stats(self, s, v, alpha=None):
        a = alpha if alpha is not None else self.ewma_alpha
        if s['pop_mean'] is None:
            s.update({'pop_mean': v, 'ewma_mean': v, 'pop_var': 0,
                      'ewma_var': 0, 'ewma_std': 0, 'count': 1})
        else:
            n = s['count'] + 1
            s['count'] = n
            old_pm, old_em = s['pop_mean'], s['ewma_mean']
            old_pv, old_ev = s['pop_var'],  s['ewma_var']
            new_pm = (1 - 1/n) * old_pm + (1/n) * v
            s['pop_mean'] = new_pm
            s['pop_var']  = old_pv*(n-1)/n + (v - new_pm)*(v - old_pm)/n
            new_em = a * old_em + (1 - a) * v
            s['ewma_mean'] = new_em
            new_ev = (a**2) * old_ev + ((1-a)**2) * s['pop_var']
            s['ewma_var'] = new_ev
            s['ewma_std'] = np.sqrt(new_ev)

    def _get_contador_id(self, row):
        return row['id'] if 'id' in row.index else None

    def _is_valid_real_value(self, contact_id, cumulative, meter_id, consumption):
        if pd.isna(cumulative):
            return False, False
        if contact_id not in self.last_real_cumulative:
            self.last_real_cumulative[contact_id] = cumulative
            if meter_id is not None:
                self.last_real_meter_id[contact_id] = meter_id
            return True, False
        if meter_id is not None:
            last_m = self.last_real_meter_id.get(contact_id)
            if last_m is not None and meter_id != last_m:
                self.last_real_cumulative[contact_id] = cumulative
                self.last_real_meter_id[contact_id] = meter_id
                return True, False
        last_c = self.last_real_cumulative[contact_id]
        if last_c is None:
            self.last_real_cumulative[contact_id] = cumulative
            return True, False
        if cumulative < last_c:
            return False, False
        return True, not pd.isna(consumption)

    def _impute_value(self, contact_id, pair_idx, subgroup_idx):
        params = self.contact_params.get(contact_id, {})
        wc = params.get('weight_contact', self.weight_contact)
        ws = 1.0 - wc

        sg_key  = (pair_idx, subgroup_idx)
        csg_key = (contact_id, pair_idx, subgroup_idx)
        cp_key  = (contact_id, pair_idx)

        ss  = self.subgroup_stats[sg_key]
        css = self.contact_stats[csg_key]
        cps = self.contact_pair_stats[cp_key]
        ps  = self.pair_stats[pair_idx]

        has_csg = css['count'] >= self.warmup_contact and css['ewma_mean'] is not None
        has_sg  = ss['count']  >= self.warmup_subgroup and ss['ewma_mean'] is not None
        has_cp  = cps['count'] >= self.warmup_contact  and cps['ewma_mean'] is not None
        has_p   = ps['count']  >= self.warmup_subgroup and ps['ewma_mean'] is not None

        if has_csg and has_sg:
            mean = wc * css['ewma_mean'] + ws * ss['ewma_mean']
            var  = (wc**2 * (css['ewma_std']**2 if css['ewma_std'] else 0) +
                    ws**2 * (ss['ewma_std']**2  if ss['ewma_std']  else 0))
            return mean, np.sqrt(var), 'contact_subgroup'
        elif has_sg:
            return ss['ewma_mean'],  ss['ewma_std'],  'only_subgroup'
        elif has_cp:
            return cps['ewma_mean'], cps['ewma_std'], 'only_contact'
        elif has_p:
            return ps['ewma_mean'],  ps['ewma_std'],  'only_group'
        return None, None, 'insufficient_data'

    def _normalize_segment(self, df_subset, start_idx, end_idx):
        if start_idx is None or end_idx is None:
            return
        start_row, end_row = df_subset.loc[start_idx], df_subset.loc[end_idx]
        sm = self._get_contador_id(start_row)
        em = self._get_contador_id(end_row)
        if sm is not None and em is not None and sm != em:
            return
        start_c = start_row['cumulative_value']
        end_c   = end_row['cumulative_value']
        is_anchor = end_row['imputation_source'].startswith('anchor_')
        if is_anchor:
            actual_diff = end_c - start_c
            segment = df_subset.loc[start_idx+1:end_idx]
        else:
            actual_diff = end_c - start_c - end_row['consumption']
            segment = df_subset.loc[start_idx+1:end_idx-1]
        imp_mask = segment['imputation_source'].str.startswith(('imputed_', 'anchor_'))
        if not imp_mask.any():
            return
        if actual_diff <= 1e-6:
            for i in segment[imp_mask].index:
                df_subset.at[i, 'corrected_consumption'] = 0
                df_subset.at[i, 'correction'] = 'Yes'
            return
        sum_imp = segment.loc[imp_mask, 'consumption'].sum()
        if sum_imp <= 1e-6:
            share = actual_diff / imp_mask.sum()
            for i in segment[imp_mask].index:
                df_subset.at[i, 'corrected_consumption'] = share
                df_subset.at[i, 'correction'] = 'Yes'
            return
        for i in segment[imp_mask].index:
            df_subset.at[i, 'corrected_consumption'] = (
                df_subset.at[i, 'consumption'] / sum_imp * actual_diff
            )
            df_subset.at[i, 'correction'] = 'Yes'

    def _flush_batch(self, batch_dict, pair_batch_dict):
        for sg_key, data in batch_dict.items():
            vals = data.get('values', [])
            if vals:
                self._update_stats(self.subgroup_stats[sg_key], np.mean(vals))
        for pair_idx, vals in pair_batch_dict.items():
            if vals:
                self._update_stats(self.pair_stats[pair_idx], np.mean(vals))

    def process_dataframe(self, df):
        print(f"Processing {len(df):,} rows for "
              f"{df['contact_id'].nunique():,} unique contacts...")
        df = df.copy()
        df['imputation_source']     = 'unknown'
        df['correction']            = 'No'
        df['corrected_consumption'] = np.nan
        df['ewma_std']              = np.nan
        df = df.sort_values(['data', 'hour']).reset_index(drop=True)

        print("Pass 1: statistics + imputation...")
        current_batch      = defaultdict(lambda: defaultdict(list))
        current_pair_batch = defaultdict(list)
        last_date_hour     = None

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            contact_id = row['contact_id']
            pair = self._get_pair_from_row(row)
            if pair is None or pair not in self.pair2idx:
                df.at[idx, 'imputation_source'] = 'unknown_pair'
                df.at[idx, 'correction']        = 'Not Applicable'
                continue
            pair_idx     = self.pair2idx[pair]
            subgroup_idx = self._get_subgroup_index(
                pair_idx, row['month_name'], row['day_name'], row['hour'])
            if subgroup_idx is None:
                subgroup_idx = 0
            date_hour = (row['data'], row['hour'])
            if last_date_hour is not None and date_hour != last_date_hour:
                self._flush_batch(current_batch, current_pair_batch)
                current_batch      = defaultdict(lambda: defaultdict(list))
                current_pair_batch = defaultdict(list)
            last_date_hour = date_hour

            meter_id = self._get_contador_id(row)
            if meter_id is not None:
                last_m = self.last_real_meter_id.get(contact_id)
                if last_m is not None and meter_id != last_m:
                    self.last_real_cumulative[contact_id] = None
                    self.last_real_idx[contact_id]        = None
                    if not pd.isna(row['cumulative_value']):
                        self.last_real_meter_id[contact_id]   = meter_id
                        self.last_real_cumulative[contact_id] = row['cumulative_value']
                        self.last_real_idx[contact_id]        = idx
                        df.at[idx, 'imputation_source'] = 'meter_change'
                    else:
                        df.at[idx, 'imputation_source'] = 'meter_change_no_reading'
                    df.at[idx, 'correction']            = 'Not Applicable'
                    df.at[idx, 'corrected_consumption'] = np.nan
                    continue

            is_real_c, has_real_cons = self._is_valid_real_value(
                contact_id, row['cumulative_value'], meter_id, row['consumption'])

            if not is_real_c:
                if contact_id not in self.last_real_idx:
                    df.at[idx, 'imputation_source'] = 'no_readings'
                    df.at[idx, 'correction']        = 'Not Applicable'
                    continue
                iv, std, src = self._impute_value(contact_id, pair_idx, subgroup_idx)
                if iv is not None:
                    df.at[idx, 'consumption']           = iv
                    df.at[idx, 'imputation_source']     = f'imputed_{src}'
                    df.at[idx, 'correction']            = 'No'
                    df.at[idx, 'corrected_consumption'] = iv
                    df.at[idx, 'ewma_std']              = std
                else:
                    df.at[idx, 'imputation_source'] = 'insufficient_data'
                    df.at[idx, 'correction']        = 'Not Applicable'
                continue

            is_baseline = contact_id not in self.last_real_idx
            if is_baseline:
                df.at[idx, 'imputation_source']     = 'baseline'
                df.at[idx, 'correction']            = 'Not Applicable'
                df.at[idx, 'corrected_consumption'] = np.nan
                self.last_real_idx[contact_id]      = idx
            elif has_real_cons:
                cons = row['consumption']
                alpha = self.contact_params.get(contact_id, {}).get(
                    'ewma_alpha', self.ewma_alpha)

                current_batch[(pair_idx, subgroup_idx)]['values'].append(cons)
                self._update_stats(
                    self.contact_stats[(contact_id, pair_idx, subgroup_idx)], cons, alpha=alpha)
                self._update_stats(
                    self.contact_pair_stats[(contact_id, pair_idx)], cons, alpha=alpha)
                current_pair_batch[pair_idx].append(cons)
                self.last_real_cumulative[contact_id] = row['cumulative_value']
                self.last_real_idx[contact_id]        = idx
                df.at[idx, 'imputation_source']     = 'real'
                df.at[idx, 'correction']            = 'Not Applicable'
                df.at[idx, 'corrected_consumption'] = cons
            else:
                iv, std, src = self._impute_value(contact_id, pair_idx, subgroup_idx)
                if iv is not None and iv >= 0:
                    df.at[idx, 'consumption']           = iv
                    df.at[idx, 'imputation_source']     = f'anchor_{src}'
                    df.at[idx, 'correction']            = 'No'
                    df.at[idx, 'corrected_consumption'] = iv
                    df.at[idx, 'ewma_std']              = std
                    self.last_real_cumulative[contact_id] = row['cumulative_value']
                    self.last_real_idx[contact_id]        = idx
                else:
                    df.at[idx, 'imputation_source'] = 'insufficient_data'
                    df.at[idx, 'correction']        = 'Not Applicable'

        self._flush_batch(current_batch, current_pair_batch)

        print("Pass 2: normalising segments...")
        ANCHOR_SOURCES = {'real', 'baseline', 'meter_change',
                          'anchor_contact_subgroup', 'anchor_only_subgroup',
                          'anchor_only_contact', 'anchor_only_group'}
        for contact_id in tqdm(CONTACTS, desc="Normalising segments for target contacts"):
            mask   = df['contact_id'] == contact_id
            c_df   = df[mask].copy()
            if len(c_df) < 2:
                continue
            r_idxs = c_df[c_df['imputation_source'].isin(ANCHOR_SOURCES)].index.tolist()
            for i in range(len(r_idxs) - 1):
                si, ei = r_idxs[i], r_idxs[i+1]
                if ei - si > 1:
                    self._normalize_segment(c_df, si, ei)
            df.loc[mask, 'corrected_consumption'] = c_df['corrected_consumption']
            df.loc[mask, 'correction']            = c_df['correction']

        return df


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def build_datetime(df):
    return pd.to_datetime(df["data"]) + pd.to_timedelta(df["hour"], unit="h")


def make_sparse(series, mask):
    out = series.copy().astype(float)
    out[~mask] = np.nan
    return out


def plot_contact(contact_id, result_aug, original_real, out_dir):
    df = result_aug.sort_values(["data", "hour"]).reset_index(drop=True)
    df["datetime"] = build_datetime(df)

    imputed_mask   = df["imputation_source"].str.startswith("imputed_", na=False)
    corrected_mask = df["correction"] == "Yes"

    orig = original_real[["data", "hour", "consumption"]].copy()
    orig["data"] = pd.to_datetime(orig["data"])
    df_orig = df[["datetime", "data", "hour"]].merge(
        orig.rename(columns={"consumption": "real_consumption"}),
        on=["data", "hour"], how="left"
    )
    real_vals      = df_orig["real_consumption"]
    imputed_vals   = make_sparse(df["consumption"],            imputed_mask)
    corrected_vals = make_sparse(df["corrected_consumption"],  corrected_mask)

    fig, ax = plt.subplots(figsize=(18, 5))

    ax.plot(df["datetime"], real_vals,
            color="#2196F3", linewidth=1.8, label="Real", zorder=3)
    ax.plot(df["datetime"], imputed_vals,
            color="#F44336", linewidth=1.5, linestyle="--",
            label="Imputed", zorder=2)
    ax.plot(df["datetime"], corrected_vals,
            color="#FFC107", linewidth=1.5, linestyle=":",
            label="Normalised", zorder=4)

    row0         = df.iloc[0]
    calibre      = row0.get("calibre",      "?")
    tipo_consumo = row0.get("tipo_consumo", "?")

    if contact_id == "LC42435":
        ax.set_title(
            f"Nominal Diameter={calibre}  |  Type of Consumption=Domestic irrigation\n",
            fontsize=11, pad=10
        )
    else:
        ax.set_title(
            f"{contact_id}  |  calibre={calibre}  |  tipo_consumo={tipo_consumo}\n",
            fontsize=11, pad=10
        )
    ax.set_xlabel("Date / Hour", fontsize=10)
    ax.set_ylabel("Consumption", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.4)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_xlim(df["datetime"].iloc[0], df["datetime"].iloc[-1])

    fig.tight_layout()
    path = os.path.join(out_dir, f"{contact_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load variables ────────────────────────────────────────────────────────
    print("Loading variables...")
    with open(VARIABLES, "rb") as f:
        data = pickle.load(f)
    idx2pair  = data["idx2pair"]
    pair2idx  = data["pair2idx"]
    subgroups = data["subgroups"]

    # ── Load per-contact hyperparams from JSON ────────────────────────────────
    print("Loading contact hyperparams from info/best_hyperparams.json...")
    with open("info/best_hyperparams.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)

    contact_params = {}
    for contact_id, group, calibre in zip(CONTACTS, GROUPS, CALIBRE):
        contact_info = json_data.get(group, {})
        wc, alpha = contact_info.get(calibre, [0.7, 0.1])
        contact_params[contact_id] = {'weight_contact': wc, 'ewma_alpha': alpha}
        print(f"  {contact_id}: weight_contact={wc}, ewma_alpha={alpha}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df["data"] = pd.to_datetime(df["data"])
    print(f"  {len(df):,} rows  |  {df['contact_id'].nunique():,} unique contacts")

    # ── Determine target (calibre, tipo_consumo) pairs ────────────────────────
    target_pairs = set(zip(CALIBRE, GROUPS))   # {("15", "Comércio"), ("20", "Rega-Doméstic")}
    pair_mask = df.apply(
        lambda r: (str(r.get("calibre", "")), str(r.get("tipo_consumo", ""))) in target_pairs,
        axis=1
    )
    df = df[pair_mask].copy()
    print(f"  Filtered to {len(target_pairs)} pair(s) {target_pairs}: "
          f"{len(df):,} rows  |  {df['contact_id'].nunique():,} unique contacts")

    # ── Save originals before masking ─────────────────────────────────────────
    aug_mask = (
        df["contact_id"].isin(CONTACTS) &
        (df["data"] >= AUG_START) &
        (df["data"] <= AUG_END)
    )
    original_real = df.loc[aug_mask & df["consumption"].notna(),
                           ["contact_id", "data", "hour", "consumption"]].copy()

    print(f"\nMasking Aug 1–14 2024 for {CONTACTS}  ({aug_mask.sum()} rows)...")
    df.loc[aug_mask, "consumption"]      = np.nan
    df.loc[aug_mask, "cumulative_value"] = np.nan

    # ── Run engine ────────────────────────────────────────────────────────────
    engine = BatchImputationEngine(
        subgroups_list=subgroups,
        idx2pair=idx2pair,
        pair2idx=pair2idx,
        warmup_contact=0,
        warmup_subgroup=0,
        weight_contact=0.7,
        ewma_alpha=0.1,
        contact_params=contact_params,
    )
    result_df = engine.process_dataframe(df)

    # ── Filter to target contacts + August window for plotting ────────────────
    result_aug = result_df[
        result_df["contact_id"].isin(CONTACTS) &
        (result_df["data"] >= AUG_START) &
        (result_df["data"] <= AUG_END)
    ].copy()

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nPlotting...")
    for contact_id in CONTACTS:
        print(f"  {contact_id}")
        c_result = result_aug[result_aug["contact_id"] == contact_id].copy()
        c_orig   = original_real[original_real["contact_id"] == contact_id].copy()
        if c_result.empty:
            print(f"    [WARN] No data in Aug window — skipping.")
            continue
        plot_contact(contact_id, c_result, c_orig, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()