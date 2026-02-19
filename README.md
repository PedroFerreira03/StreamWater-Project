# Streamwater Project — DISCO

**DISCO (DIstributional Subgroup Clustering for Online Imputation)** is a fast and scalable framework for **online time-series imputation** of water consumption data.

The method leverages **frequency-distribution clustering of behavioural subgroups** across **hourly, daily, weekly and monthly patterns**.

---

# 📦 Project Structure

```text
.
├── src
│   ├── data_analysis.ipynb      # Clustering (model warm-up)
│   └── impute_ts.py             # Online imputation pipeline
│
├── Data
│   ├── Contadores.xlsx
│   └── TelemetriaConsumosVilaSol_v2.csv
│
├── variables
│   └── variables.pkl            # Generated warm-up artefacts
│
└── README.md
```

---

# 1️⃣ Clustering — `src/data_analysis.ipynb`

This notebook prepares the historical dataset and builds the **distributional subgroup clusters** used during online imputation.

## Input datasets

### `Contadores.xlsx`
Metadata for each **Point of Consumption (LC)**:
- `contact_id`
- `tipo_consumo` (Type of Consumption)

### `TelemetriaConsumosVilaSol_v2.csv`
Hourly cumulative meter readings:

| Column | Description |
|---|---|
| id | Meter ID |
| contact_id | Point of Consumption ID |
| calibre | Nominal diameter (mm) |
| data_instalacao | Installation date |
| data | Day of consumption |
| index_0 – index_23 | Hourly cumulative values |

---

## 🔄 Data Transformation

Daily cumulative readings are converted into an **hourly time-series dataset**:

`seq_all_df` → exported as `inputs_ts`

### Final schema (1 row per hour)

**Metadata**
- id  
- contact_id  
- calibre  
- data_instalacao  
- tipo_consumo  

**Time features**
- hour  
- month / month_name  
- day_of_week / day_name  

**Consumption**
- cumulative_value  
- consumption (hourly usage)

---

## 🧠 Clustering Strategy

Clustering is performed using a **hierarchical grouping**.

### Group level
(Type of Consumption, Calibre)

### Subgroup level
(hour, day_of_week, month)

The output is a list of lists, where each index corresponds to a group. Each index also contains a list with a dictionary comprising:
- Months
- Days
- Hours
- Dataframe of subgroup

### Stored artefacts
Saved to: `variables/variables.pkl`

Contents:
- List of subgroup's DataFrames lists
- `idx2pair` → index → (tipo_consumo, calibre)  
- `pair2idx` → (tipo_consumo, calibre) → index

This file represents the **model warm-up** required for online imputation.

---

# 2️⃣ Online Imputation — `src/impute_ts.py`

## Belief Update Logic

The belief update for a **Point of Consumption (LC)** and its corresponding **Subgroup** follows an Exponential Moving Average (EMA) model to balance historical data with new observations.

### 1. LC Belief Update
For each **LC**, the belief is updated at each time step `t` as follows:

`μ(t)_LC = (1 - α)μ(t-1)_LC + α x(t)_LC`

* **μ(t)_LC**: The updated belief at time step `t`.
* **x(t)_LC**: The true hourly consumption value of the LC at time step `t`.
* **α (Alpha)**: A decay rate in the range (0, 1) that controls the influence of historical information.

> **Note:** Each LC maintains a distinct belief for each subgroup it belongs to.

### 2. Subgroup Belief Update
Similarly, the belief associated with the subgroup is updated as:

`μ(t)_subgroup = (1 - α)μ(t-1)_subgroup + α x̄(t)_subgroup`

* **x̄(t)_subgroup**: The mean consumption value of all LCs within the subgroup at time `t`.

---

## ⚙️ Script Behaviour & Inputs

This script performs **real-time imputation** and optional **post-processing**.

| Parameter | Description | Default |
|---|---|---|
| data | `variables.pkl` warm-up file | required |
| input_csv | Hourly time-series file | required |
| weight_contact | Weight of contact history | 0.7 |
| ewma_alpha | EWMA update factor (α) | 0.1 |
| skip | Skip corrections + cumulative rebuild | False |

**The script performs the following:**
1. Performs online imputation + updates (warmup) the stored values.
2. Applies EWMA corrections (optional).
3. Rebuilds cumulative values (optional).

*Set `skip=True` to run **warm-up only**.*

---

## 🔁 Imputation Fallback Strategy

DISCO performs imputation using a **hierarchical fallback mechanism** to guarantee robustness when data is missing or insufficient at finer levels of granularity. 

Each imputed value is returned **together with the level used**, allowing full traceability and quality assessment.

### Imputation Levels (from most specific to most general)

| Level | Name | Description |
|---|---|---|
| **Level 1 (Default)** | `contact_subgroup` | Uses Contact's history restricted to the subgroup + Subgroups general data. |
| **Level 2** | `only_subgroup` | Uses the subgroup distribution of the **group (tipo_consumo + calibre)**, ignoring contact history. |
| **Level 3** | `only_contact` | Uses the historical behaviour of the **contact_id across all subgroups**. |
| **Level 4** | `only_group` | Uses the global distribution of the **group (tipo_consumo + calibre)**. This is the most generic fallback. |

If no level can produce a value, the consumption is left **missing**.

---

# 📊 Best Hyperparameters per Group

Only groups with validated weights are shown.

| Type of Consumption | Calibre | weight_contact | ewma_alpha |
|---|---|---|---|
| Rega Inframoura | 40 | 0.1 | 0.1 |
| Comércio | 15 | 0.9 | 0.9 |
| Doméstico | 15 | 0.9 | 0.9 |
| Serviços-Condomínio | 15 | 0.3 | 0.3 |
| Indústria | 15 | 0.7 | 0.9 |
| Serviços-Condomínio | 20 | 0.7 | 0.5 |
| Indústria | 50 | 0.5 | 0.9 |
| Indústria | 40 | 0.9 | 0.9 |
| Indústria | 20 | 0.7 | 0.7 |
| Comércio | 40 | 0.1 | 0.1 |
| Rega Inframoura | 15 | 0.1 | 0.1 |
| Serviços 2º Contador – Rega Condomínio | 50 | 0.7 | 0.9 |
| Indústria sem RSU | 80 | 0.1 | 0.7 |
| Serviços-Condomínio | 65 | 0.7 | 0.3 |
| Serviços-Condomínio | 100 | 0.9 | 0.7 |
| Rega-Doméstico | 20 | 0.9 | 0.3 |
| Doméstico | 20 | 0.9 | 0.9 |
| Não Doméstico – Rega | 20 | 0.9 | 0.7 |
| Rega-Doméstico | 15 | 0.1 | 0.1 |
| Rega Inframoura | 30 | 0.1 | 0.3 |
| Indústria | 30 | 0.3 | 0.9 |
| Instituições de Utilidade Pública | 15 | 0.1 | 0.1 |
| Rega Inframoura | 20 | 0.1 | 0.5 |
| Serviços 2º Contador – Rega Condomínio | 80 | 0.1 | 0.3 |
| Serviços 2º Contador – Rega Condomínio | 40 | 0.1 | 0.3 |
| Indústria | 25 | 0.1 | 0.1 |
| Indústria Inframoura | 15 | 0.1 | 0.5 |
| Obras | 15 | 0.1 | 0.1 |
| Serviços-Condomínio | 80 | 0.1 | 0.9 |
| Obras | 20 | 0.1 | 0.7 |

---

# 🚀 Summary

DISCO enables:
- **Behaviour-aware clustering** for precise profiling.
- **Real-time imputation** for continuous data streams.
- **Adaptive EWMA corrections** to follow consumption shifts.
- **Scalable deployment for smart-meter time series.**
