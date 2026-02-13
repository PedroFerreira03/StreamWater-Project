import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time


class OnlineImputationEngine:
    def __init__(self, subgroups_list, idx2pair, pair2idx, contact_id,
                 warmup_contact=10, warmup_subgroup=50, weight_contact=0.7,
                 ewma_alpha=0.1):
        """
        Initialize the online imputation engine for a specific contact.
        
        Parameters:
        -----------
        subgroups_list : list
            Output from divide_dataframes() - list of subgroups per (calibre, tipo_consumo)
        idx2pair : dict
            Maps index -> (calibre, tipo_consumo)
        pair2idx : dict
            Maps (calibre, tipo_consumo) -> index
        contact_id : str
            The contact ID we're processing
        warmup_contact : int
            Minimum real observations needed for contact stats per subgroup
        warmup_subgroup : int
            Minimum real observations needed for subgroup stats
        weight_contact : float
            Weight for contact stats (0.7 means 70% contact, 30% subgroup)
        ewma_alpha : float
            Smoothing factor for exponentially weighted moving average (0 < alpha <= 1)
            Lower values = more smoothing, higher values = more reactive
        """
        self.subgroups_list = subgroups_list
        self.idx2pair = idx2pair
        self.pair2idx = pair2idx
        self.contact_id = contact_id
        self.warmup_contact = warmup_contact
        self.warmup_subgroup = warmup_subgroup
        self.weight_contact = weight_contact
        self.weight_subgroup = 1 - weight_contact
        self.ewma_alpha = ewma_alpha
        
        # Initialize statistics for subgroups (updated online)
        # Now tracking BOTH population mean E[X] and EWMA mean μ separately
        self.subgroup_stats = defaultdict(lambda: {
            'pop_mean': None,       # E[X]_t (population mean for variance calculation)
            'ewma_mean': None,      # μ_t (EWMA mean used for imputation)
            'pop_var': None,        # V(X)_t (population variance)
            'ewma_var': None,       # V(μ)_t (variance of EWMA estimator)
            'ewma_std': None,       # sqrt(V(μ)_t) (uncertainty in mean estimate)
            'count': 0
        })
        
        # CHANGED: Initialize contact statistics per (pair_idx, subgroup_idx)
        # This ensures we don't mix statistics from different groups
        self.contact_stats = defaultdict(lambda: {
            'pop_mean': None,
            'ewma_mean': None,
            'pop_var': None,
            'ewma_var': None,
            'ewma_std': None,
            'count': 0
        })
        
        # Results storage (only for target contact)
        self.results = []
        self.last_real_idx = -1
        self.last_cumulative = 0

    def _build(self):
        # Extract ALL contact data from subgroups_list (for all contacts in all pairs)
        self._extract_all_data()
        
        # Create subgroup lookup for all pairs
        self.subgroup_lookup = self._create_subgroup_lookup()
        
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
    
    def _extract_all_data(self):
        """
        Extract ALL data from all contacts in ALL pairs' subgroups.
        Creates a unified timeline with all observations.
        """
        all_rows = []
        target_contact_rows = []
        
        # Extract data from ALL pairs
        for pair_idx, subgroups in enumerate(self.subgroups_list):
            for subgroup_idx, subgroup in enumerate(subgroups):
                df = subgroup['df'].copy()
                df['pair_idx'] = pair_idx  # ADDED: Track which pair this belongs to
                df['subgroup_idx'] = subgroup_idx
                
                # All contacts data
                all_rows.append(df)
                
                # Target contact data
                target_df = df[df['contact_id'] == self.contact_id].copy()
                if len(target_df) > 0:
                    target_contact_rows.append(target_df)
        
        if not target_contact_rows:
            raise ValueError(f"Contact {self.contact_id} not found in any subgroups")
        
        # Combine all data and sort by timestamp
        self.all_data = pd.concat(all_rows, ignore_index=True)
        self.all_data = self.all_data.sort_values(by=['data', 'hour']).reset_index(drop=True)
        self.all_data['is_real'] = ~self.all_data['consumption'].isna()
        self.all_data['is_target'] = self.all_data['contact_id'] == self.contact_id
        self.target_num = self.all_data['is_target'].sum()
        
        # Get unique pairs for target contact
        target_data = pd.concat(target_contact_rows, ignore_index=True)
        target_pairs = set()
        for _, row in target_data.iterrows():
            pair = self._get_pair_from_row(row)
            if pair:
                target_pairs.add(pair)
        
        print(f"Total data points (all contacts): {len(self.all_data)}")
        print(f"Target contact ({self.contact_id}) data points: {self.target_num}")
        print(f"Target contact belongs to {len(target_pairs)} group(s): {target_pairs}")
        print(f"Number of unique contacts across all pairs: {self.all_data['contact_id'].nunique()}")

    def _create_subgroup_lookup(self):
        """
        Create mapping from (pair_idx, month, day, hour) to subgroup index.
        """
        lookup = {}
        
        for pair_idx, subgroups in enumerate(self.subgroups_list):
            for subgroup_idx, subgroup in enumerate(subgroups):
                months = subgroup['months']
                days = subgroup['days']
                hours = subgroup['hours']
                
                # Map all combinations to this subgroup
                for month in months:
                    for day in days:
                        for hour in hours:
                            key = (pair_idx, month, day, int(hour))
                            lookup[key] = subgroup_idx
        
        return lookup
    
    def _update_stats(self, stats_dict, new_value):
        """
        Update both population statistics and EWMA statistics with a new observation.
        
        Based on:
        - Population mean: E[X]_t = (1 - 1/n)E[X]_{t-1} + (1/n)x_t
        - Population variance: V(X)_t = (V(X)_{t-1} + (x_t - E[X]_t)(x_t - E[X]_{t-1})/n
        - EWMA mean: μ_t = α μ_{t-1} + (1-α) x_t
        - EWMA variance: V(μ_t) = α² V(μ_{t-1}) + (1-α)² V(X)_t
        
        Parameters:
        -----------
        stats_dict : dict
            Dictionary with 'pop_mean', 'ewma_mean', 'pop_var', 'ewma_var', etc.
        new_value : float
            New observation to incorporate
        """
        alpha = self.ewma_alpha
        
        if stats_dict['pop_mean'] is None:
            # First observation - initialize everything
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
            
            # Update count first
            count += 1
            stats_dict['count'] = count
            
            # Update population mean
            new_pop_mean = (1 - 1/count) * old_pop_mean + (1/count) * new_value
            stats_dict['pop_mean'] = new_pop_mean
            
            # Update population variance
            new_pop_var = old_pop_var*(count-1)/count + (new_value - new_pop_mean) * (new_value - old_pop_mean)/count
            stats_dict['pop_var'] = new_pop_var
            
            # Update EWMA mean
            new_ewma_mean = alpha * old_ewma_mean + (1 - alpha) * new_value
            stats_dict['ewma_mean'] = new_ewma_mean
            
            # Update EWMA variance
            new_ewma_var = (alpha**2) * old_ewma_var + ((1 - alpha)**2) * new_pop_var
            stats_dict['ewma_var'] = new_ewma_var
            stats_dict['ewma_std'] = np.sqrt(new_ewma_var)
    
    def _get_subgroup_index(self, row):
        """Get subgroup index for a given row."""
        pair_idx = row.get('pair_idx')
        if pair_idx is None:
            return None
        key = (pair_idx, row['month_name'], row['day_name'], int(row['hour']))
        return self.subgroup_lookup.get(key, None)
    
    def _update_contact_stats(self, pair_idx, subgroup_idx, consumption):
        """
        Update contact statistics for a (pair, subgroup) combination with a new real consumption value.
        
        CHANGED: Now uses (pair_idx, subgroup_idx) as key instead of just subgroup_idx
        """
        contact_key = (pair_idx, subgroup_idx)
        self._update_stats(self.contact_stats[contact_key], consumption)
    
    def _impute_value(self, pair_idx, subgroup_idx):
        """
        Impute a consumption value using weighted combination of contact and subgroup EWMA means.
        
        CHANGED: Now uses pair_idx to look up correct contact stats
        
        Returns:
        --------
        tuple: (imputed_consumption, combined_ewma_std) or (None, None) if insufficient data
        """
        # Get subgroup stats (indexed by pair_idx, subgroup_idx)
        subgroup_key = (pair_idx, subgroup_idx)
        subgroup_ewma_mean = self.subgroup_stats[subgroup_key]['ewma_mean']
        subgroup_ewma_std = self.subgroup_stats[subgroup_key]['ewma_std']
        subgroup_count = self.subgroup_stats[subgroup_key]['count']
        
        # Get contact stats for this (pair, subgroup) combination
        contact_key = (pair_idx, subgroup_idx)
        contact_ewma_mean = self.contact_stats[contact_key]['ewma_mean']
        contact_ewma_std = self.contact_stats[contact_key]['ewma_std']
        contact_count = self.contact_stats[contact_key]['count']
        
        # Check if we have enough data
        if contact_count < self.warmup_contact or subgroup_count < self.warmup_subgroup:
            return None, None
        
        if contact_ewma_mean is None or subgroup_ewma_mean is None:
            return None, None
        
        # Weighted combination of EWMA means
        imputed_mean = (self.weight_contact * contact_ewma_mean + 
                       self.weight_subgroup * subgroup_ewma_mean)
        
        # Combined EWMA variance 
        contact_ewma_var = contact_ewma_std**2 if contact_ewma_std else 0
        subgroup_ewma_var = subgroup_ewma_std**2 if subgroup_ewma_std else 0
        
        combined_ewma_var = (self.weight_contact**2 * contact_ewma_var + 
                            self.weight_subgroup**2 * subgroup_ewma_var)
        
        combined_ewma_std = np.sqrt(combined_ewma_var)
        
        return imputed_mean, combined_ewma_std
    
    def _normalize_segment(self):
        """
        Normalize imputed values between two real values.
        
        Logic:
        - Get actual difference: real_cumulative_end - real_cumulative_start (always >= 0)
        - For each imputed value: new_consumption = (na_consumption / sum_na_consumption) * actual_difference
        - This redistributes the actual consumption proportionally
        
        IMPORTANT: Only normalize if the contador_id (meter ID) is the same at both ends!
        If the meter changes, the scale changes, so normalization is invalid.
        """
        start_idx = self.last_real_idx
        end_idx = len(self.results) - 1
        
        # Check if contador_id changed between start and end
        contador_start = self.results[start_idx].get('contador_id')
        contador_end = self.results[end_idx].get('contador_id')
        
        if contador_start != contador_end:
            return
        
        # Get real cumulative values at boundaries
        real_cumulative_start = self.results[start_idx]['cumulative']
        real_cumulative_end = self.results[end_idx]['cumulative']
        
        # Actual difference between real readings
        cons_value = self.results[end_idx]['consumption']
        type_data = self.results[end_idx]['type']
        if cons_value and type_data == 'real':
            actual_diff = real_cumulative_end - real_cumulative_start - cons_value
        else:
            actual_diff = real_cumulative_end - real_cumulative_start
        
        if actual_diff < 0:
            print("Negative value. Skipping normalization... FIX")
            return
        
        segment = self.results[start_idx+1:end_idx+1]

        # Sum of all imputed (NA) consumption values in this segment
        sum_na_consumption = sum(r['consumption'] for r in segment if r['type'] == 'imputed')
        
        if sum_na_consumption <= 0:
            return
        
        # Normalize each imputed value proportionally
        cumulative = real_cumulative_start
        for i in range(start_idx + 1, len(self.results)):
            if self.results[i]['type'] == 'imputed':
                na_consumption = self.results[i]['consumption']
                new_consumption = (na_consumption / sum_na_consumption) * actual_diff
                cumulative += new_consumption
                
                self.results[i]['corrected_consumption'] = new_consumption
                self.results[i]['corrected_cumulative'] = cumulative
    
    def process_all_points(self):
        """
        Process all data points in chronological order.
        Updates subgroup stats for ALL contacts using batched means, performs imputation only for target contact.
        
        Returns:
        --------
        Generator yielding (all_data_idx, is_target, result) tuples
        """
        # Track which target data points we've processed
        target_idx = 0
        
        # Buffer for accumulating observations from same (date, hour)
        current_batch = {
            'date': None,
            'hour': None,
            'observations': defaultdict(list)  # (pair_idx, subgroup_idx) -> list of consumption values
        }
        
        def flush_batch():
            """Process accumulated observations and update subgroup stats."""
            if current_batch['observations']:
                for subgroup_key, consumptions in current_batch['observations'].items():
                    if consumptions:
                        # Calculate mean of all observations for this subgroup in this time window
                        batch_mean = np.mean(consumptions)
                        # Update stats with the batch mean
                        self._update_stats(self.subgroup_stats[subgroup_key], batch_mean)
                
                # Clear the batch
                current_batch['observations'].clear()
        
        for all_idx in range(len(self.all_data)):
            row = self.all_data.iloc[all_idx]
            subgroup_idx = self._get_subgroup_index(row)
            
            if subgroup_idx is None:
                continue
            
            pair_idx = row.get('pair_idx')
            if pair_idx is None:
                continue
            
            # Check if we've moved to a new (date, hour) window
            row_date = row['data']
            row_hour = row['hour']
            
            if (current_batch['date'] != row_date or current_batch['hour'] != row_hour):
                # Flush previous batch before starting new one
                flush_batch()
                current_batch['date'] = row_date
                current_batch['hour'] = row_hour
            
            is_target = row['is_target']
            
            # Accumulate real observations for batch processing
            # CHANGED: Key now includes pair_idx
            if row['is_real']:
                subgroup_key = (pair_idx, subgroup_idx)
                current_batch['observations'][subgroup_key].append(row['consumption'])
            
            # Process target contact for imputation and results
            if is_target:
                result = self._process_target_point(row, pair_idx, subgroup_idx)
                target_idx += 1
                yield (all_idx, True, result)
            else:
                # Non-target contact: just accumulate for batch update, no result
                yield (all_idx, False, None)
        
        # Flush any remaining batch at the end
        flush_batch()
    
    def _process_target_point(self, row, pair_idx, subgroup_idx):
        """
        Process a single point for the target contact (imputation logic).
        
        CHANGED: Now receives pair_idx as parameter
        """
        result = {
            'data': row['data'],
            'month': row['month_name'],
            'day': row['day_name'],
            'hour': row['hour'],
            'pair_idx': pair_idx,  # ADDED: Track which pair this point belongs to
            'subgroup_idx': subgroup_idx,
            'corrected_consumption': None,
            'corrected_cumulative': None,
            'index': len(self.results),
            'contador_id': row.get('contador_id', 'default')
        }
        
        if row['is_real']:
            # Real value
            consumption = row['consumption']
            cumulative = row['cumulative_value']
            
            # Update contact statistics (CHANGED: now includes pair_idx)
            self._update_contact_stats(pair_idx, subgroup_idx, consumption)
            
            result['type'] = 'real'
            result['diff'] = consumption
            result['consumption'] = consumption
            result['cumulative'] = cumulative
            result['ewma_std'] = None

            self.results.append(result)
            
        else:
            # Missing value - try to impute (CHANGED: now includes pair_idx)
            imputed_consumption, combined_ewma_std = self._impute_value(pair_idx, subgroup_idx)

            if imputed_consumption is not None:
                cumulative = row['cumulative_value'] if not pd.isna(row['cumulative_value']) else self.last_cumulative + imputed_consumption
                result['type'] = 'imputed'
                result['diff'] = imputed_consumption
                if imputed_consumption < 0:
                    print("ERRORRRRR, FIX")

                result['consumption'] = imputed_consumption
                result['cumulative'] = cumulative
                result['ewma_std'] = combined_ewma_std
                self.results.append(result)
                    
            else:
                # Skip this point - not enough data to impute
                result['type'] = 'skipped'
                result['diff'] = None
                result['consumption'] = None
                result['cumulative'] = None
                result['ewma_std'] = None
                self.results.append(result)
                return result
        
        if not pd.isna(row['cumulative_value']): 
            if self.last_real_idx >= 0 and self.last_real_idx < len(self.results) - 2:
                self._normalize_segment()
            
            self.last_real_idx = len(self.results) - 1
            self.last_cumulative = cumulative
            
        return result


class SlidingWindowVisualizer:
    def __init__(self, engine, delay=0.1, window_size=100, title=None):
        """
        Create online visualization with sliding window of the imputation process.
        
        Parameters:
        -----------
        engine : OnlineImputationEngine
            The imputation engine to visualize
        delay : float
            Delay between points in seconds
        window_size : int
            Number of points to show in the sliding window
        title : str
            Plot title (default uses contact_id)
        """
        self.engine = engine
        self.delay = delay
        self.window_size = window_size
        self.title = title or f"Online Imputation - {engine.contact_id}"
        
        # Setup interactive plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        
        # Storage for all data (for final summary)
        self.all_data = {
            'real': [],
            'imputed': [],
            'corrected': []
        }
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Initialize plot appearance."""
        self.ax.set_xlabel('Time Index', fontsize=12)
        self.ax.set_ylabel('Consumption Difference', fontsize=12)
        self.ax.set_title(self.title, fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper left')
    
    def _get_window_data(self, current_target_idx):
        """
        Get data for the current window (only target contact points).
        
        Returns:
        --------
        dict with lists for real, imputed, and corrected points in window
        """
        start_idx = max(0, current_target_idx - self.window_size + 1)
        
        window_data = {
            'real_x': [], 'real_y': [], 'real_ts': [],
            'imputed_x': [], 'imputed_y': [], 'imputed_yerr': [], 'imputed_ts': [],
            'corrected_x': [], 'corrected_y': [], 'corrected_ts': []
        }
        
        for result in self.engine.results[start_idx:current_target_idx+1]:
            idx = result['index']
            
            if result['type'] == 'real':
                window_data['real_x'].append(idx)
                window_data['real_y'].append(result['diff'])
                window_data['real_ts'].append(result['data'])
                
            elif result['type'] == 'imputed':
                window_data['imputed_x'].append(idx)
                window_data['imputed_y'].append(result['diff'])
                
                # Use EWMA uncertainty (2 standard deviations for ~95% confidence)
                ewma_std = result['ewma_std'] if result['ewma_std'] else 0
                lower_err = 2 * ewma_std if result['diff'] - 2*ewma_std >= 0 else result['diff']
                upper_err = 2 * ewma_std

                window_data['imputed_yerr'].append((lower_err, upper_err))
                window_data['imputed_ts'].append(result['data'])
            
            # Check for corrected values
            if result['corrected_consumption'] is not None:
                if idx not in window_data['corrected_x']:
                    window_data['corrected_x'].append(idx)
                    window_data['corrected_y'].append(result['corrected_consumption'])
                    window_data['corrected_ts'].append(result['data'])
        
        window_data['imputed_yerr'] = np.array(window_data['imputed_yerr']).T
        return window_data
    
    def _update_plot(self, current_target_idx, all_processed, target_processed):
        """Update the plot with current window data."""
        self.ax.clear()
        
        # Get window data
        window = self._get_window_data(current_target_idx)
        
        # Plot real points (blue)
        if window['real_x']:
            self.ax.scatter(window['real_x'], window['real_y'], 
                          c='blue', s=50, label='Real', zorder=3, alpha=0.8)
        
        # Plot imputed points (red) with error bars
        if window['imputed_x']:
            self.ax.errorbar(window['imputed_x'], window['imputed_y'],
                           yerr=window['imputed_yerr'],
                           fmt='o', c='red', markersize=6, label='Imputed (±2σ EWMA)',
                           alpha=0.6, capsize=3, zorder=2)
        
        # Plot corrected points (yellow)
        if window['corrected_x']:
            self.ax.scatter(window['corrected_x'], window['corrected_y'],
                          c='yellow', s=50, label='Corrected',
                          zorder=4, alpha=0.8, edgecolors='orange', linewidths=1.5)
        
        # Set title with progress
        self.ax.set_title(
            f'{self.title} | All: {all_processed}/{len(self.engine.all_data)} | Target: {target_processed}/{self.engine.target_num}',
            fontsize=14
        )
        
        self.ax.set_xlabel('Date', fontsize=12)
        self.ax.set_ylabel('Consumption Difference', fontsize=12)
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        
        # Set x-axis limits for sliding window
        start_idx = max(0, current_target_idx - self.window_size + 1)
        self.ax.set_xlim(start_idx - 5, current_target_idx + 5)
        
        # Create x-axis labels showing dates
        all_x = window['real_x'] + window['imputed_x'] + window['corrected_x']
        all_ts = window['real_ts'] + window['imputed_ts'] + window['corrected_ts']
        
        if all_x and all_ts:
            sorted_data = sorted(zip(all_x, all_ts), key=lambda x: x[0])
            
            date_labels = {}
            prev_date = None
            
            for x, ts in sorted_data:
                current_date = pd.Timestamp(ts).strftime('%b %d')
                if current_date != prev_date:
                    date_labels[x] = current_date
                    prev_date = current_date
            
            if date_labels:
                tick_positions = list(date_labels.keys())
                tick_labels = list(date_labels.values())
                self.ax.set_xticks(tick_positions)
                self.ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # Auto-scale y-axis
        all_y = window['real_y'] + window['imputed_y'] + window['corrected_y']
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            padding = max((y_max - y_min) * 0.1, 0.1)
            self.ax.set_ylim(y_min - padding, y_max + padding)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.000001)
    
    def run(self):
        """
        Run the online visualization with sliding window.
        """
        print(f"Starting online imputation for {self.engine.contact_id}")
        print(f"Total points to process (all contacts): {len(self.engine.all_data)}")
        print(f"Target contact points: {self.engine.target_num}")
        print(f"Window size: {self.window_size} points")
        print(f"Delay: {self.delay}s per point")
        print(f"EWMA alpha: {self.engine.ewma_alpha}")
        print("-" * 60)
        
        all_processed = 0
        target_processed = 0
        skipped_count = 0
        
        for all_idx, is_target, result in self.engine.process_all_points():
            all_processed += 1
            
            if is_target:
                target_processed += 1
                if result is None:
                    skipped_count += 1
                    continue
                  
                # Store for summary
                if result['type'] == 'real':
                    self.all_data['real'].append(result)
                elif result['type'] == 'imputed':
                    self.all_data['imputed'].append(result)
                
                # Update plot
                if len(self.engine.results) > 0:
                    self._update_plot(len(self.engine.results) - 1, all_processed, target_processed)
                
                # Delay
                time.sleep(self.delay)
        
        # Collect corrected points
        for r in self.engine.results:
            if r['corrected_consumption'] is not None:
                self.all_data['corrected'].append(r)
        
        print("-" * 60)
        print("Processing complete!")
        print(f"Total points processed (all contacts): {all_processed}")
        print(f"Target contact processed: {target_processed}")
        print(f"Skipped points: {skipped_count}")
        print(f"Real points: {len(self.all_data['real'])}")
        print(f"Imputed points: {len(self.all_data['imputed'])}")
        print(f"Corrected points: {len(self.all_data['corrected'])}")
        
        plt.ioff()
        plt.show()
        
        return self.engine.results


if __name__ == "__main__":
    import pickle

    # Load all variables
    with open("./variables/variables.pkl", "rb") as f:
        data = pickle.load(f)

    idx2pair = data["idx2pair"]
    pair2idx = data["pair2idx"]
    subgroups = data["subgroups"]

    # Initialize online engine for a specific contact
    # CHANGED: Removed mapa_contact_id parameter
    contact_id = 'LC43022'
    
    engine = OnlineImputationEngine(
        subgroups_list=subgroups,
        idx2pair=idx2pair,
        pair2idx=pair2idx,
        contact_id=contact_id,
        warmup_contact=0,
        warmup_subgroup=0,
        weight_contact=0.7,
        ewma_alpha=0.1 
    )

    engine._build()

    # Run online visualization with sliding window
    visualizer = SlidingWindowVisualizer(engine, delay=0, window_size=100)
    results = visualizer.run()