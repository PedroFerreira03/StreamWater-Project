import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle


class BatchImputationEngine:
    """
    Process entire dataset for all contacts, imputing missing consumption values.
    Handles multiple (calibre, tipo_consumo) groups and performs online imputation.
    Enhanced with fallback strategies and proper meter change handling.
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
        
        # Statistics per (pair_idx, subgroup_idx)
        self.subgroup_stats = defaultdict(lambda: {
            'pop_mean': None,
            'ewma_mean': None,
            'pop_var': None,
            'ewma_var': None,
            'ewma_std': None,
            'count': 0
        })
        
        # Statistics per (contact_id, pair_idx, subgroup_idx)
        self.contact_stats = defaultdict(lambda: {
            'pop_mean': None,
            'ewma_mean': None,
            'pop_var': None,
            'ewma_var': None,
            'ewma_std': None,
            'count': 0
        })
        
        # Statistics per (contact_id, pair_idx) - all subgroups combined
        self.contact_pair_stats = defaultdict(lambda: {
            'pop_mean': None,
            'ewma_mean': None,
            'pop_var': None,
            'ewma_var': None,
            'ewma_std': None,
            'count': 0
        })
        
        # Statistics per pair_idx - all subgroups combined
        self.pair_stats = defaultdict(lambda: {
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
        
        # If last_cumulative is None (after meter change with no reading), treat current as new baseline
        if last_cumulative is None:
            self.last_real_cumulative[contact_id] = cumulative
            return True, False  # Valid cumulative, but no consumption to compute
        
        if cumulative < last_cumulative:
            return False, False  # Invalid cumulative
        
        # Valid cumulative, check if consumption is also valid
        has_real_consumption = not pd.isna(consumption)
        
        return True, has_real_consumption
    
    def _impute_value(self, contact_id, pair_idx, subgroup_idx):
        """
        Impute consumption value using fallback hierarchy:
        1. Contact + Subgroup (weighted combination)
        2. Only Subgroup (no contact history)
        3. Only Contact + Pair (contact history without subgroup discrimination)
        4. Only Pair (group statistics)
        5. Insufficient data
        
        Returns:
        --------
        tuple: (imputed_consumption, ewma_std, imputation_source) or (None, None, None)
        """
        # Get stats for all levels
        subgroup_key = (pair_idx, subgroup_idx)
        contact_subgroup_key = (contact_id, pair_idx, subgroup_idx)
        contact_pair_key = (contact_id, pair_idx)
        pair_key = pair_idx
        
        subgroup_stats = self.subgroup_stats[subgroup_key]
        contact_subgroup_stats = self.contact_stats[contact_subgroup_key]
        contact_pair_stats = self.contact_pair_stats[contact_pair_key]
        pair_stats = self.pair_stats[pair_key]
        
        # Check warmup requirements for each level
        has_contact_subgroup = (contact_subgroup_stats['count'] >= self.warmup_contact and 
                                contact_subgroup_stats['ewma_mean'] is not None)
        has_subgroup = (subgroup_stats['count'] >= self.warmup_subgroup and 
                       subgroup_stats['ewma_mean'] is not None)
        has_contact_pair = (contact_pair_stats['count'] >= self.warmup_contact and 
                           contact_pair_stats['ewma_mean'] is not None)
        has_pair = (pair_stats['count'] >= self.warmup_subgroup and 
                   pair_stats['ewma_mean'] is not None)
        
        # Fallback hierarchy
        if has_contact_subgroup and has_subgroup:
            # Best case: both contact+subgroup and subgroup available
            imputed_mean = (self.weight_contact * contact_subgroup_stats['ewma_mean'] + 
                           self.weight_subgroup * subgroup_stats['ewma_mean'])
            
            contact_ewma_var = contact_subgroup_stats['ewma_std']**2 if contact_subgroup_stats['ewma_std'] else 0
            subgroup_ewma_var = subgroup_stats['ewma_std']**2 if subgroup_stats['ewma_std'] else 0
            
            combined_ewma_var = (self.weight_contact**2 * contact_ewma_var + 
                                self.weight_subgroup**2 * subgroup_ewma_var)
            combined_ewma_std = np.sqrt(combined_ewma_var)
            
            return imputed_mean, combined_ewma_std, 'contact_subgroup'
        
        elif has_subgroup:
            # No contact history, but subgroup available
            return subgroup_stats['ewma_mean'], subgroup_stats['ewma_std'], 'only_subgroup'
        
        elif has_contact_pair:
            # No subgroup data, use contact's overall history for this pair
            return contact_pair_stats['ewma_mean'], contact_pair_stats['ewma_std'], 'only_contact'
        
        elif has_pair:
            # No contact data at all, use overall pair statistics
            return pair_stats['ewma_mean'], pair_stats['ewma_std'], 'only_group'
        
        else:
            # Insufficient data at all levels
            return None, None, 'insufficient_data'
    
    def _normalize_segment(self, df_subset, start_idx, end_idx):
        """
        Normalize imputed values between two real CUMULATIVE values.
        Works with anchor_imputed endpoints.
        Updates correction status to 'Yes' for normalized values.
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
        
        # Get the end consumption (might be real or anchor_imputed)
        end_consumption = end_row['consumption']
        
        # Check if end row is anchor imputed
        is_anchor_imputed = end_row['imputation_source'].startswith('anchor_') 
        
        if is_anchor_imputed:
            # End is anchor_imputed - include it in normalization
            actual_diff = end_cumulative - start_cumulative
            segment = df_subset.loc[start_idx+1:end_idx]  # Include end_idx
        else:
            # End is real - exclude it from normalization
            actual_diff = end_cumulative - start_cumulative - end_consumption
            segment = df_subset.loc[start_idx+1:end_idx-1]  # Exclude end_idx

        # Get imputed values in segment (both fully imputed and anchor imputed)
        imputed_mask = segment['imputation_source'].str.startswith(('imputed_', 'anchor_'))
        
        if not imputed_mask.any():
            return
        
        # Handle edge case: actual_diff is zero or negative
        if actual_diff <= 1e-6:  # Essentially zero (handle floating point errors)
            # Set all imputed values to zero
            for idx in segment[imputed_mask].index:
                df_subset.at[idx, 'corrected_consumption'] = 0
                df_subset.at[idx, 'correction'] = 'Yes'
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
                df_subset.at[idx, 'correction'] = 'Yes'
            return
        
        # Standard proportional normalization
        for idx in segment[imputed_mask].index:
            old_consumption = df_subset.at[idx, 'consumption']
            new_consumption = (old_consumption / sum_imputed) * actual_diff
            df_subset.at[idx, 'corrected_consumption'] = new_consumption
            df_subset.at[idx, 'correction'] = 'Yes'
    
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
        DataFrame with added columns: imputation_source, correction, corrected_consumption, ewma_std
        """
        print(f"Processing {len(df)} rows for {df['contact_id'].nunique()} unique contacts...")
        
        # Add new columns
        df = df.copy()
        df['imputation_source'] = 'unknown'
        df['correction'] = 'No'
        df['corrected_consumption'] = np.nan
        df['ewma_std'] = np.nan
        
        # Sort by time
        df = df.sort_values(['data', 'hour']).reset_index(drop=True)
        
        # Process in three passes:
        # Pass 1: Update statistics and perform initial imputation
        # Pass 2: Normalize segments
        # Pass 3: Compute cumulatives

        print("Pass 1: Updating statistics and imputing values...")
        
        # Group by (date, hour) for batch updates
        current_batch = defaultdict(lambda: defaultdict(list))
        current_pair_batch = defaultdict(list)
        last_date_hour = None
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            contact_id = row['contact_id']
            
            # Get pair from row data
            pair = self._get_pair_from_row(row)
            
            if pair is None:
                df.at[idx, 'imputation_source'] = 'unknown_pair'
                df.at[idx, 'correction'] = 'Not Applicable'
                continue
            
            if pair not in self.pair2idx:
                df.at[idx, 'imputation_source'] = 'unknown_pair'
                df.at[idx, 'correction'] = 'Not Applicable'
                continue

            pair_idx = self.pair2idx[pair]
            
            # Get subgroup (might be None, but that's okay now with fallback)
            subgroup_idx = self._get_subgroup_index(
                pair_idx, row['month_name'], row['day_name'], row['hour']
            )
            
            # If no subgroup, use a placeholder (we'll rely on fallback)
            if subgroup_idx is None:
                subgroup_idx = 0  # Use first subgroup as placeholder for key purposes
            
            date_hour = (row['data'], row['hour']) 
            
            # Flush batch if new time window
            if last_date_hour is not None and date_hour != last_date_hour:
                self._flush_batch(current_batch, current_pair_batch)
                current_batch = defaultdict(lambda: defaultdict(list))
                current_pair_batch = defaultdict(list)
            
            last_date_hour = date_hour
            
            # Check for meter change FIRST (before validity check)
            meter_id = self._get_contador_id(row)
            if meter_id is not None:
                last_meter = self.last_real_meter_id.get(contact_id)
                if last_meter is not None and meter_id != last_meter:
                    # Meter changed - reset tracking regardless of data validity
                    self.last_real_cumulative[contact_id] = None  # Reset
                    self.last_real_idx[contact_id] = None  # Reset
                    
                    # If we have a valid cumulative, use it as new baseline
                    if not pd.isna(row['cumulative_value']):
                        self.last_real_meter_id[contact_id] = meter_id # Only update is real to not lose track
                        self.last_real_cumulative[contact_id] = row['cumulative_value']
                        self.last_real_idx[contact_id] = idx
                        df.at[idx, 'imputation_source'] = 'meter_change'
                    
                    else:
                        # Meter changed but no reading yet
                        df.at[idx, 'imputation_source'] = 'meter_change_no_reading'
                    
                    df.at[idx, 'correction'] = 'Not Applicable'
                    df.at[idx, 'corrected_consumption'] = np.nan
                    continue
            
            # Check if value is valid real
            is_real_cumulative, has_real_consumption = self._is_valid_real_value(
                contact_id, 
                row['cumulative_value'],
                meter_id,
                row['consumption']
            )
            
            if not is_real_cumulative:
                if contact_id not in self.last_real_idx:
                    df.at[idx, 'imputation_source'] = 'no_readings'
                    df.at[idx, 'correction'] = 'Not Applicable'
                    continue

                # Invalid or missing cumulative - needs full imputation
                imputed_value, ewma_std, source = self._impute_value(contact_id, pair_idx, subgroup_idx)
                
                if imputed_value is not None:
                    df.at[idx, 'consumption'] = imputed_value
                    df.at[idx, 'imputation_source'] = f'imputed_{source}'  # e.g., 'imputed_contact_subgroup'
                    df.at[idx, 'correction'] = 'No'  # Not yet corrected
                    df.at[idx, 'corrected_consumption'] = imputed_value
                    df.at[idx, 'ewma_std'] = ewma_std
                else:
                    df.at[idx, 'imputation_source'] = 'insufficient_data'
                    df.at[idx, 'correction'] = 'Not Applicable'

                continue
            
            # Has valid cumulative value
            is_baseline = contact_id not in self.last_real_idx
            
            if is_baseline:
                # First value
                df.at[idx, 'imputation_source'] = 'baseline'
                df.at[idx, 'correction'] = 'Not Applicable'
                df.at[idx, 'corrected_consumption'] = np.nan
                self.last_real_idx[contact_id] = idx
                
            elif has_real_consumption:
                # Real cumulative AND real consumption
                consumption = row['consumption']
                
                # Update statistics at ALL levels
                # 1. Subgroup level
                subgroup_key = (pair_idx, subgroup_idx)
                current_batch[subgroup_key]['values'].append(consumption)
                
                # 2. Contact + Subgroup level
                contact_subgroup_key = (contact_id, pair_idx, subgroup_idx)
                self._update_stats(self.contact_stats[contact_subgroup_key], consumption)
                
                # 3. Contact + Pair level (all subgroups)
                contact_pair_key = (contact_id, pair_idx)
                self._update_stats(self.contact_pair_stats[contact_pair_key], consumption)
                
                # 4. Pair level (all subgroups)
                current_pair_batch[pair_idx].append(consumption)
                
                # Update tracking
                self.last_real_cumulative[contact_id] = row['cumulative_value']
                self.last_real_idx[contact_id] = idx
                
                df.at[idx, 'imputation_source'] = 'real'
                df.at[idx, 'correction'] = 'Not Applicable'
                df.at[idx, 'corrected_consumption'] = consumption
                
            else:
                # Real cumulative but MISSING consumption - ANCHOR IMPUTED
                imputed_value, ewma_std, source = self._impute_value(contact_id, pair_idx, subgroup_idx)
                
                if imputed_value is not None and imputed_value >= 0:
                    df.at[idx, 'consumption'] = imputed_value
                    df.at[idx, 'imputation_source'] = f'anchor_{source}'  # e.g., 'anchor_contact_subgroup'
                    df.at[idx, 'correction'] = 'No'  # Will become 'Yes' after normalization
                    df.at[idx, 'corrected_consumption'] = imputed_value
                    df.at[idx, 'ewma_std'] = ewma_std
                    
                    # Update tracking - this cumulative is real!
                    self.last_real_cumulative[contact_id] = row['cumulative_value']
                    self.last_real_idx[contact_id] = idx
                else:
                    df.at[idx, 'imputation_source'] = 'insufficient_data'
                    df.at[idx, 'correction'] = 'Not Applicable'
        
        # Flush final batch
        self._flush_batch(current_batch, current_pair_batch)
        
        print("Pass 2: Normalizing segments...")

        for contact_id in tqdm(df['contact_id'].unique()):
            contact_mask = df['contact_id'] == contact_id
            contact_df = df[contact_mask].copy()
            
            if len(contact_df) < 2:
                continue
            
            # Anchor
            anchor_sources = ['real', 'baseline', 'meter_change',
                  'anchor_contact_subgroup', 'anchor_only_subgroup',
                  'anchor_only_contact', 'anchor_only_group'] # Only ones with real cumulative

            # Also include any anchor_imputed variants
            real_indices = contact_df[
                (contact_df['imputation_source'].isin(anchor_sources))
            ].index.tolist()
            
            # Normalize between consecutive anchor values
            for i in range(len(real_indices) - 1):
                start_idx = real_indices[i]
                end_idx = real_indices[i + 1]
                
                if end_idx - start_idx > 1:  # Has imputed values between
                    self._normalize_segment(contact_df, start_idx, end_idx)
            
            # Write changes back to main dataframe
            df.loc[contact_mask, 'corrected_consumption'] = contact_df['corrected_consumption']
            df.loc[contact_mask, 'correction'] = contact_df['correction']
        
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
                
                # Handle meter_change_no_reading - reset cumulative
                if row['imputation_source'] == 'meter_change_no_reading':
                    current_cumulative = None
                    continue
                
                # If we have a real cumulative value, use it as anchor
                # These sources have trustworthy cumulative values
                anchor_sources = ['real', 'baseline', 'meter_change']
                if (row['imputation_source'] in anchor_sources or 
                    row['imputation_source'].startswith('anchor_')):
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
    
    def _flush_batch(self, batch_dict, pair_batch_dict):
        """Update subgroup and pair statistics with batch means."""
        # Update subgroup statistics
        for subgroup_key, data in batch_dict.items():
            values = data.get('values', [])
            if values:
                batch_mean = np.mean(values)
                self._update_stats(self.subgroup_stats[subgroup_key], batch_mean)
        
        # Update pair-level statistics
        for pair_idx, values in pair_batch_dict.items():
            if values:
                batch_mean = np.mean(values)
                self._update_stats(self.pair_stats[pair_idx], batch_mean)


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
    
    # Initialize engine
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
    print("\nImputation source breakdown:")
    print(result_df['imputation_source'].value_counts())
    print("\nCorrection status breakdown:")
    print(result_df['correction'].value_counts())
    print(f"\nRows with corrected consumption: {result_df['corrected_consumption'].notna().sum()}")
    
    # Cross-tabulation for detailed analysis
    print("\nCross-tabulation (Imputation Source vs Correction):")
    print(pd.crosstab(result_df['imputation_source'], result_df['correction']))
    
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
    
    # Process full dataset
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