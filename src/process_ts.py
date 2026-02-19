import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle


class BatchImputationEngine:
    """
    Process entire dataset for all contacts, imputing missing consumption values.
    Handles multiple (calibre, tipo_consumo) groups and performs online imputation.
    """
    
    def __init__(self, subgroups_list, idx2pair, pair2idx,
                 warmup_contact=0, warmup_subgroup=0, weight_contact=0.7,
                 ewma_alpha=0.1):
        """
        Initialize batch imputation engine for all contacts.
        
        Parameters:
        -----------
        subgroups_list : list
            Output from divide_dataframes() - list of subgroups per (calibre, tipo_consumo)
        idx2pair : dict
            Maps index -> (calibre, tipo_consumo)
        pair2idx : dict
            Maps (calibre, tipo_consumo) -> index
        warmup_contact : int
            Minimum real observations needed for contact stats
        warmup_subgroup : int
            Minimum real observations needed for subgroup stats
        weight_contact : float
            Weight for contact stats (0.7 = 70% contact, 30% subgroup)
        ewma_alpha : float
            EWMA smoothing factor (lower = more smoothing)
        """
        self.subgroups_list = subgroups_list
        self.idx2pair = idx2pair
        self.pair2idx = pair2idx
        self.warmup_contact = warmup_contact
        self.warmup_subgroup = warmup_subgroup
        self.weight_contact = weight_contact
        self.weight_subgroup = 1 - weight_contact
        self.ewma_alpha = ewma_alpha
        
        # Statistics per (pair_idx, subgroup_idx) -> Subgroup for each Group
        self.subgroup_stats = defaultdict(lambda: {
            'pop_mean': None,
            'ewma_mean': None,
            'pop_var': None,
            'ewma_var': None,
            'ewma_std': None,
            'count': 0
        })
        
        # Statistics per (contact_id, pair_idx, subgroup_idx) -> Subgroup for each LC
        self.contact_stats = defaultdict(lambda: {
            'pop_mean': None,
            'ewma_mean': None,
            'pop_var': None,
            'ewma_var': None,
            'ewma_std': None,
            'count': 0
        })
        
        # Track last real cumulative per contact
        self.last_real_cumulative = {}
        self.last_real_meter_id = {}  # Track meter ID (from 'id' column)
        self.last_real_idx = {}  # Track index of last real value per contact
        
        # Subgroup lookup
        self.subgroup_lookup = {}
        self._build_subgroup_lookup()
    
    def _build_subgroup_lookup(self):
        """Create mapping from (pair_idx, month, day, hour) to subgroup_idx."""
        for pair_idx, subgroups in enumerate(self.subgroups_list):
            for subgroup_idx, subgroup in enumerate(subgroups):
                months = subgroup['months']
                days = subgroup['days']
                hours = subgroup['hours']
                
                for month in months:
                    for day in days:
                        for hour in hours:
                            key = (pair_idx, month, day, int(hour))
                            self.subgroup_lookup[key] = subgroup_idx
    
    def _get_subgroup_index(self, pair_idx, month, day, hour):
        """Get subgroup index for given parameters."""
        key = (pair_idx, month, day, int(hour))
        return self.subgroup_lookup.get(key, None)
    
    def _get_pair_from_row(self, row):
        """
        Extract (calibre, tipo_consumo) pair from row data.
        
        Parameters:
        -----------
        row : Series
            Row containing 'calibre' and 'tipo_consumo' columns
            
        Returns:
        --------
        tuple or None: (calibre, tipo_consumo) if available, None otherwise
        """
        if 'calibre' in row.index and 'tipo_consumo' in row.index:
            calibre = row['calibre']
            tipo_consumo = row['tipo_consumo']
            
            # Check for missing values
            if pd.notna(calibre) and pd.notna(tipo_consumo):
                return (calibre, tipo_consumo)
        
        return None
    
    def _update_stats(self, stats_dict, new_value):
        """Update statistics with new observation"""
        alpha = self.ewma_alpha
        
        if stats_dict['pop_mean'] is None:
            stats_dict['pop_mean'] = new_value
            stats_dict['ewma_mean'] = new_value
            stats_dict['pop_var'] = 0
            stats_dict['ewma_var'] = 0
            stats_dict['ewma_std'] = 0
            stats_dict['count'] = 1
        else:
            count = stats_dict['count']
            old_pop_mean = stats_dict['pop_mean']
            old_ewma_mean = stats_dict['ewma_mean']
            old_pop_var = stats_dict['pop_var']
            old_ewma_var = stats_dict['ewma_var']
            
            count += 1
            stats_dict['count'] = count
            
            new_pop_mean = (1 - 1/count) * old_pop_mean + (1/count) * new_value
            stats_dict['pop_mean'] = new_pop_mean
            
            new_pop_var = old_pop_var*(count-1)/count + (new_value - new_pop_mean) * (new_value - old_pop_mean)/count
            stats_dict['pop_var'] = new_pop_var
            
            new_ewma_mean = alpha * old_ewma_mean + (1 - alpha) * new_value
            stats_dict['ewma_mean'] = new_ewma_mean
            
            new_ewma_var = (alpha**2) * old_ewma_var + ((1 - alpha)**2) * new_pop_var
            stats_dict['ewma_var'] = new_ewma_var
            stats_dict['ewma_std'] = np.sqrt(new_ewma_var)
    
    def _get_contador_id(self, row):
        """Safely get meter id from row, return None if column doesn't exist."""
        if 'id' in row.index:
            return row['id']
        return None
    
    def _is_valid_real_value(self, contact_id, cumulative, meter_id, consumption):
        """
        Check if a value should be treated as real based on:
        1. Has non-null cumulative value
        2. For first value: just needs cumulative (establishes baseline)
        3. For subsequent values: cumulative must be non-decreasing
        4. Meter ID (from 'id' column) hasn't changed (if available)
        
        Returns: tuple (is_real_cumulative, has_real_consumption)
        """
        if pd.isna(cumulative):
            return False, False
        
        # First real value for this contact - valid as baseline
        if contact_id not in self.last_real_cumulative:
            self.last_real_cumulative[contact_id] = cumulative
            if meter_id is not None:
                self.last_real_meter_id[contact_id] = meter_id
            # First value: real cumulative, but no consumption to compute
            return True, False
        
        # Check if meter changed (only if meter_id column exists)
        if meter_id is not None:
            last_meter = self.last_real_meter_id.get(contact_id)
            if last_meter is not None and meter_id != last_meter:
                # Meter changed - new baseline
                self.last_real_cumulative[contact_id] = cumulative
                self.last_real_meter_id[contact_id] = meter_id
                return True, False
        
        # Check if cumulative is non-decreasing
        last_cumulative = self.last_real_cumulative[contact_id]
        if cumulative < last_cumulative:
            return False, False  # Invalid cumulative
        
        # Valid cumulative, check if consumption is also valid
        has_real_consumption = not pd.isna(consumption)
        
        return True, has_real_consumption
    
    def _impute_value(self, contact_id, pair_idx, subgroup_idx):
        """
        Impute consumption value using weighted combination of contact and subgroup stats.
        
        Returns:
        --------
        tuple: (imputed_consumption, ewma_std) or (None, None)
        """
        # Get stats
        subgroup_key = (pair_idx, subgroup_idx)
        contact_key = (contact_id, pair_idx, subgroup_idx)  # CHANGED: now includes pair_idx
        
        subgroup_stats = self.subgroup_stats[subgroup_key]
        contact_stats = self.contact_stats[contact_key]
        
        # Check warmup requirements
        if (contact_stats['count'] < self.warmup_contact or 
            subgroup_stats['count'] < self.warmup_subgroup):
            return None, None
        
        if contact_stats['ewma_mean'] is None or subgroup_stats['ewma_mean'] is None:
            return None, None
        
        # Weighted combination
        imputed_mean = (self.weight_contact * contact_stats['ewma_mean'] + 
                       self.weight_subgroup * subgroup_stats['ewma_mean'])
        
        # Combined variance
        contact_ewma_var = contact_stats['ewma_std']**2 if contact_stats['ewma_std'] else 0
        subgroup_ewma_var = subgroup_stats['ewma_std']**2 if subgroup_stats['ewma_std'] else 0
        
        combined_ewma_var = (self.weight_contact**2 * contact_ewma_var + 
                            self.weight_subgroup**2 * subgroup_ewma_var)
        
        combined_ewma_std = np.sqrt(combined_ewma_var)
        
        return imputed_mean, combined_ewma_std
    
    def _normalize_segment(self, df_subset, start_idx, end_idx):
        """
        Normalize imputed values between two real CUMULATIVE values.
        Works with both real and anchor_imputed endpoints.
        """
        if start_idx is None or end_idx is None:
            return
        
        # Get real values at boundaries
        start_row = df_subset.loc[start_idx]
        end_row = df_subset.loc[end_idx]
        
        # Check if meter changed (only if column exists)
        start_meter = self._get_contador_id(start_row)
        end_meter = self._get_contador_id(end_row)
        if start_meter is not None and end_meter is not None:
            if start_meter != end_meter:
                return  # Don't normalize across meter changes
        
        start_cumulative = start_row['cumulative_value']
        end_cumulative = end_row['cumulative_value']
        
        # Get the end consumption (might be real or imputed)
        end_consumption = end_row['consumption']
        
        # Determine segment based on end row type
        if end_row['imputation_type'] in ['imputed', 'fully_imputed', 'anchor_imputed']:
            actual_diff = end_cumulative - start_cumulative
            segment = df_subset.loc[start_idx+1:end_idx]  # Include end_idx
        else:
            actual_diff = end_cumulative - start_cumulative - end_consumption
            segment = df_subset.loc[start_idx+1:end_idx-1]  # Exclude end_idx

        # Get imputed values in segment
        imputed_mask = segment['imputation_type'].isin(['imputed', 'fully_imputed', 'anchor_imputed'])
        
        if not imputed_mask.any():
            return
        
        # Handle edge case: actual_diff is zero or negative
        if actual_diff <= 1e-6:  # Essentially zero (handle floating point errors)
            # Set all imputed values to zero
            for idx in segment[imputed_mask].index:
                df_subset.at[idx, 'corrected_consumption'] = 0
                if idx != end_idx:
                    df_subset.at[idx, 'imputation_type'] = 'corrected'
            return
        
        # Normal case: normalize proportionally
        sum_imputed = segment.loc[imputed_mask, 'consumption'].sum()
        
        # Handle case where sum_imputed is essentially zero
        if sum_imputed <= 1e-6:
            # Distribute actual_diff equally among imputed values
            num_imputed = imputed_mask.sum()
            equal_share = actual_diff / num_imputed
            
            for idx in segment[imputed_mask].index:
                df_subset.at[idx, 'corrected_consumption'] = equal_share
                if idx != end_idx:
                    df_subset.at[idx, 'imputation_type'] = 'corrected'
            return
        
        # Standard proportional normalization
        for idx in segment[imputed_mask].index:
            old_consumption = df_subset.at[idx, 'consumption']
            new_consumption = (old_consumption / sum_imputed) * actual_diff
            df_subset.at[idx, 'corrected_consumption'] = new_consumption
            if idx != end_idx:
                df_subset.at[idx, 'imputation_type'] = 'corrected'
    
    def process_dataframe(self, df):
        """
        Process entire dataframe with all contacts.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe with columns: contact_id, data, hour, month_name, day_name,
            consumption, cumulative_value, calibre, tipo_consumo, id (meter ID, optional)
        
        Returns:
        --------
        DataFrame with added columns: imputation_type, corrected_consumption, ewma_std
        """
        print(f"Processing {len(df)} rows for {df['contact_id'].nunique()} unique contacts...")
        
        # Add new columns
        df = df.copy()
        df['imputation_type'] = 'unknown'
        df['corrected_consumption'] = np.nan
        df['ewma_std'] = np.nan
        
        # Sort by time
        df = df.sort_values(['data', 'hour']).reset_index(drop=True)
        
        # Process in two passes:
        # Pass 1: Update statistics and perform initial imputation
        # Pass 2: Normalize segments
        # Pass 3: Compute cumulatives

        print("Pass 1: Updating statistics and imputing values...")
        
        # Group by (date, hour) for batch updates
        current_batch = defaultdict(lambda: defaultdict(list))
        last_date_hour = None
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            contact_id = row['contact_id']
            
            # Get pair from row data (CHANGED: dynamic extraction)
            pair = self._get_pair_from_row(row)
            
            if pair is None:
                df.at[idx, 'imputation_type'] = 'unknown_pair'
                continue
            
            if pair not in self.pair2idx:
                df.at[idx, 'imputation_type'] = 'unknown_pair'
                continue
            pair_idx = self.pair2idx[pair]
            
            # Get subgroup
            subgroup_idx = self._get_subgroup_index(
                pair_idx, row['month_name'], row['day_name'], row['hour']
            )
            
            if subgroup_idx is None:
                df.at[idx, 'imputation_type'] = 'unknown_subgroup'
                continue
            
            date_hour = (row['data'], row['hour']) 
            
            # Flush batch if new time window
            if last_date_hour is not None and date_hour != last_date_hour:
                self._flush_batch(current_batch)
                current_batch = defaultdict(lambda: defaultdict(list))
            
            last_date_hour = date_hour
            
            # Check if value is valid real
            meter_id = self._get_contador_id(row)
            is_real_cumulative, has_real_consumption = self._is_valid_real_value(
                contact_id, 
                row['cumulative_value'],
                meter_id,
                row['consumption']
            )
            
            if not is_real_cumulative:
                # Invalid or missing cumulative - needs full imputation
                imputed_value, ewma_std = self._impute_value(contact_id, pair_idx, subgroup_idx)
                
                if imputed_value is not None:
                    df.at[idx, 'consumption'] = imputed_value
                    df.at[idx, 'imputation_type'] = 'fully_imputed'
                    df.at[idx, 'corrected_consumption'] = imputed_value
                    df.at[idx, 'ewma_std'] = ewma_std
                else:
                    df.at[idx, 'imputation_type'] = 'insufficient_data'
                continue
            
            # Has valid cumulative value
            is_baseline = contact_id not in self.last_real_idx
            
            if is_baseline:
                # First value or after meter change
                df.at[idx, 'imputation_type'] = 'baseline'
                df.at[idx, 'corrected_consumption'] = np.nan
                self.last_real_idx[contact_id] = idx
                
            elif has_real_consumption:
                # Real cumulative AND real consumption
                consumption = row['consumption']
                
                # Update statistics
                subgroup_key = (pair_idx, subgroup_idx)
                current_batch[subgroup_key]['values'].append(consumption)
                
                contact_key = (contact_id, pair_idx, subgroup_idx)  # CHANGED: now includes pair_idx
                self._update_stats(self.contact_stats[contact_key], consumption)
                
                # Update tracking
                self.last_real_cumulative[contact_id] = row['cumulative_value']
                self.last_real_idx[contact_id] = idx
                
                df.at[idx, 'imputation_type'] = 'real'
                df.at[idx, 'corrected_consumption'] = consumption
                
            else:
                # Real cumulative but MISSING consumption - impute but mark as anchor
                imputed_value, ewma_std = self._impute_value(contact_id, pair_idx, subgroup_idx)
                
                if imputed_value is not None and imputed_value >= 0:
                    df.at[idx, 'consumption'] = imputed_value
                    df.at[idx, 'imputation_type'] = 'anchor_imputed'  # Special marker!
                    df.at[idx, 'corrected_consumption'] = imputed_value
                    df.at[idx, 'ewma_std'] = ewma_std
                    
                    # Update tracking - this cumulative is real!
                    self.last_real_cumulative[contact_id] = row['cumulative_value']
                    self.last_real_idx[contact_id] = idx
                else:
                    df.at[idx, 'imputation_type'] = 'insufficient_data'
        
        # Flush final batch
        self._flush_batch(current_batch)
        
        print("Pass 2: Normalizing segments...")

        for contact_id in tqdm(df['contact_id'].unique()):
            contact_mask = df['contact_id'] == contact_id
            contact_df = df[contact_mask].copy()
            
            if len(contact_df) < 2:
                continue
            
            # Find all real value indices (including anchor_imputed)
            real_indices = contact_df[
                contact_df['imputation_type'].isin(['real', 'baseline', 'anchor_imputed', 'corrected'])
            ].index.tolist()
            
            # Normalize between consecutive real values
            for i in range(len(real_indices) - 1):
                start_idx = real_indices[i]
                end_idx = real_indices[i + 1]
                
                if end_idx - start_idx > 1:  # Has imputed values between
                    self._normalize_segment(contact_df, start_idx, end_idx)
            
            # Write changes back to main dataframe
            df.loc[contact_mask, 'corrected_consumption'] = contact_df['corrected_consumption']
            df.loc[contact_mask, 'imputation_type'] = contact_df['imputation_type']
        
        print("Pass 3: Recomputing cumulative values...")

        for contact_id in tqdm(df['contact_id'].unique()):
            contact_mask = df['contact_id'] == contact_id
            contact_indices = df[contact_mask].index
            
            if len(contact_indices) == 0:
                continue
            
            # Sort indices to process in chronological order
            contact_indices = sorted(contact_indices)
            
            # Track cumulative value
            current_cumulative = None
            current_meter_id = None
            
            for idx in contact_indices:
                row = df.loc[idx]
                
                # Check if meter changed (only if column exists)
                row_meter = self._get_contador_id(row)
                if current_meter_id is not None and row_meter is not None:
                    if row_meter != current_meter_id:
                        current_cumulative = None  # Reset on meter change
                
                if row_meter is not None:
                    current_meter_id = row_meter
                
                # If we have a real cumulative value, use it as anchor
                if row['imputation_type'] in ['real', 'baseline', 'anchor_imputed']:
                    if not pd.isna(row['cumulative_value']):
                        current_cumulative = row['cumulative_value']
                        continue
                
                # Otherwise, compute cumulative from previous + consumption
                if current_cumulative is not None:
                    # Use corrected consumption if available, otherwise use raw consumption
                    consumption = row['corrected_consumption'] if not pd.isna(row['corrected_consumption']) else row['consumption']
                    
                    if not pd.isna(consumption):
                        current_cumulative += consumption
                        df.at[idx, 'cumulative_value'] = current_cumulative

        return df
    
    def _flush_batch(self, batch_dict):
        """Update subgroup statistics with batch means."""
        for subgroup_key, data in batch_dict.items():
            values = data.get('values', [])
            if values:
                batch_mean = np.mean(values)
                self._update_stats(self.subgroup_stats[subgroup_key], batch_mean)


def process_full_dataset(input_csv, output_csv, subgroups_list, idx2pair, pair2idx, **kwargs):
    """
    Process full dataset from CSV and save results.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    output_csv : str
        Path to save output CSV file
    subgroups_list, idx2pair, pair2idx : 
        Subgroup and mapping data structures
    **kwargs : 
        Additional parameters for BatchImputationEngine
    """
    # Load data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loaded {len(df)} rows")
    print(f"Unique contacts: {df['contact_id'].nunique()}")
    print(f"Missing consumption values: {df['consumption'].isna().sum()}")
    
    # Initialize engine (CHANGED: removed mapa_contact_id parameter)
    engine = BatchImputationEngine(
        subgroups_list=subgroups_list,
        idx2pair=idx2pair,
        pair2idx=pair2idx,
        **kwargs
    )
    
    # Process
    result_df = engine.process_dataframe(df)
    
    # Summary statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total rows processed: {len(result_df)}")
    print("\nImputation type breakdown:")
    print(result_df['imputation_type'].value_counts())
    print(f"\nRows with corrected consumption: {result_df['corrected_consumption'].notna().sum()}")
    
    # Save
    print(f"\nSaving results to {output_csv}...")
    result_df.to_csv(output_csv, index=False)
    print("Done!")
    
    return result_df


if __name__ == "__main__":
    # Load variables
    with open("./variables/variables.pkl", "rb") as f:
        data = pickle.load(f)
    
    idx2pair = data["idx2pair"]
    pair2idx = data["pair2idx"]
    subgroups = data["subgroups"]
    
    # Process full dataset (CHANGED: removed mapa_contact_id)
    result_df = process_full_dataset(
        input_csv="./Data/input_ts.csv",
        output_csv="./Data/output_ts.csv",
        subgroups_list=subgroups,
        idx2pair=idx2pair,
        pair2idx=pair2idx,
        warmup_contact=0,
        warmup_subgroup=0,
        weight_contact=0.7,
        ewma_alpha=0.1
    )
