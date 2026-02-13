# Time Series Cleaning Project

## Overview
This project provides a pipeline for **cleaning univariate time series data with additional multivariate information**.  
Its main goal is to ensure that any missing or anomalous values are automatically corrected, so that the resulting dataset is consistent, reliable, and ready for further analysis or modeling.

## Objective
Given a new time series:
- **Detect** missing values or anomalies.  
- **Correct** these values by imputing realistic replacements.  
- **Leverage group information** to improve the correction process.  

Corrections are not made in isolation. Instead, the process takes into account similar series that share the same characteristics (`tipo_consumo`, `calibre`, `month`, `week_day`, and `hour`).

## How It Works
1. Input a new time series dataset.  
2. The system checks for irregularities:  
   - Missing values  
   - Anomalous values (values that don’t fit expected patterns)  
3. Each irregularity is corrected using information from comparable groups of data.  
4. The output is a **cleaned time series** with corrected values.  

## Usage
1. Prepare your dataset with the required columns:  
   - `tipo_consumo`  
   - `calibre`  
   - `month`  
   - `week_day`  
   - `hour`  
   - Value column (the actual measurement over time)  

2. Run the cleaning pipeline (still needs to be implemented).  

3. The result will be a cleaned version of the original dataset, with missing and anomalous values imputed.  

## Why This Matters
- Ensures data quality before analysis or reporting.  
- Reduces the risk of biased results due to gaps or anomalies.  
- Makes downstream tasks (visualization, modeling, or decision-making) more reliable.  
