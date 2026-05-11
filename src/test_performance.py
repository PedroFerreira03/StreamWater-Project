import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from animator import OnlineImputationEngine, SlidingWindowVisualizer


class OnlineImputationTester(OnlineImputationEngine):
    """
    Extended engine that evaluates imputation performance by treating 
    specified values as missing and comparing predictions to ground truth.
    Now includes fallback imputation strategy from BatchImputationEngine.
    """
    
    def __init__(self, subgroups_list,  
                 contact_id, test_config_path, warmup_contact=0, 
                 warmup_subgroup=0, weight_contact=0.7, ewma_alpha=0.1):
        """
        Initialize tester with configuration for synthetic missing data.
        
        Parameters
        ----------
        test_config_path : str
            Path to JSON file mapping contact IDs to test periods
            Format: {contact_id: {date_string: "start_hour-end_hour", ...}}
        """
        # Load and validate test configuration
        with open(test_config_path, 'r') as f:
            test_config = json.load(f)
        
        if contact_id not in test_config:
            raise ValueError(f"Contact {contact_id} not in test configuration")
        
        self.test_dates = test_config[contact_id]
        self.missing_map = self._build_missing_map()
        
        super().__init__(
            subgroups_list, contact_id,
            warmup_contact, warmup_subgroup, weight_contact, ewma_alpha
        )
        
        # Performance tracking - now with breakdown by imputation type
        self.imputed_errors = []
        self.corrected_errors = []
        
        # Track imputation counts by type
        self.imputation_counts = {
            'contact_subgroup': 0,
            'only_subgroup': 0,
            'only_contact': 0,
            'only_group': 0,
            'insufficient_data': 0
        }
        
        # Map result index to ground truth for quick lookup
        self.ground_truth_map = {}
        
        # Additional statistics for fallback strategy
        # Statistics per (contact_id, pair_idx) - all subgroups combined
        self.contact_pair_stats = {}
        
        # Statistics per pair_idx - all subgroups combined
        self.pair_stats = {}
  

    def _build_missing_map(self):
        """
        Convert test configuration to set of (date, hour) tuples.
        
        Returns
        -------
        set of (pd.Timestamp, int) tuples
        """
        missing_set = set()
        
        for date_str, hour_range in self.test_dates.items():
            date = pd.to_datetime(date_str, format='%d/%m/%Y')
            start_hour, end_hour = map(int, hour_range.split('-'))
            
            for hour in range(start_hour, end_hour + 1):
                missing_set.add((date, hour))
        
        print(f"Test configuration: {len(missing_set)} synthetic missing points")
        return missing_set
    
    def _is_test_missing(self, row):
        """Check if row should be treated as missing for testing."""
        date = pd.to_datetime(row['data'])
        hour = int(row['hour'])
        return (date, hour) in self.missing_map
    
    def _build(self):
        """Override to initialize additional statistics structures."""
        super()._build()
        
        # Initialize contact_pair_stats and pair_stats
        for pair_idx in range(len(self.subgroups_list)):
            self.pair_stats[pair_idx] = {
                'pop_mean': None,
                'ewma_mean': None,
                'pop_var': None,
                'ewma_var': None,
                'ewma_std': None,
                'count': 0
            }
            
            # Initialize for target contact
            contact_pair_key = (self.contact_id, pair_idx)
            self.contact_pair_stats[contact_pair_key] = {
                'pop_mean': None,
                'ewma_mean': None,
                'pop_var': None,
                'ewma_var': None,
                'ewma_std': None,
                'count': 0
            }
    
    def _extract_all_data(self):
        super()._extract_all_data()
        
        # Identify test missing points
        self.all_data['is_test_missing'] = self.all_data.apply(
            lambda row: self._is_test_missing(row) if row['is_target'] else False,
            axis=1
        )
        
        test_mask = self.all_data['is_test_missing']
        n_test = test_mask.sum()
        print(f"\n{'='*60}")
        print(f"TEST SETUP: Marked {n_test} points as test missing")
        print(f"{'='*60}\n")
        
        # Store ground truth values BEFORE marking as missing
        self.all_data['ground_truth_consumption'] = np.nan
        self.all_data.loc[test_mask, 'ground_truth_consumption'] = \
            self.all_data.loc[test_mask, 'consumption'].values
        
        self.all_data.loc[test_mask, 'consumption'] = np.nan
        self.all_data.loc[test_mask, 'is_real'] = False
        self.all_data.loc[test_mask, 'cumulative_value'] = np.nan
    
    def _update_statistics(self, pair_idx, subgroup_idx, new_value):
        """
        Override to also update contact_pair_stats and pair_stats.
        """
        # Call parent's update (updates subgroup_stats and contact_stats)
        super()._update_statistics(pair_idx, subgroup_idx, new_value)
        
        # Update contact_pair_stats (all subgroups for this contact and pair)
        contact_pair_key = (self.contact_id, pair_idx)
        if contact_pair_key in self.contact_pair_stats:
            self._update_stats_dict(self.contact_pair_stats[contact_pair_key], new_value)
        
        # Update pair_stats (all subgroups for this pair, all contacts)
        if pair_idx in self.pair_stats:
            self._update_stats_dict(self.pair_stats[pair_idx], new_value)
    
    def _update_stats_dict(self, stats_dict, new_value):
        """Helper method to update a statistics dictionary."""
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
    
    def _impute_value(self, pair_idx, subgroup_idx):
        """
        Override to implement fallback hierarchy:
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
        contact_subgroup_key = (self.contact_id, pair_idx, subgroup_idx)
        contact_pair_key = (self.contact_id, pair_idx)
        pair_key = pair_idx
        
        subgroup_stats = self.subgroup_stats.get(subgroup_key, {
            'pop_mean': None, 'ewma_mean': None, 'pop_var': None,
            'ewma_var': None, 'ewma_std': None, 'count': 0
        })
        contact_subgroup_stats = self.contact_stats.get(contact_subgroup_key, {
            'pop_mean': None, 'ewma_mean': None, 'pop_var': None,
            'ewma_var': None, 'ewma_std': None, 'count': 0
        })
        contact_pair_stats = self.contact_pair_stats.get(contact_pair_key, {
            'pop_mean': None, 'ewma_mean': None, 'pop_var': None,
            'ewma_var': None, 'ewma_std': None, 'count': 0
        })
        pair_stats = self.pair_stats.get(pair_key, {
            'pop_mean': None, 'ewma_mean': None, 'pop_var': None,
            'ewma_var': None, 'ewma_std': None, 'count': 0
        })
        
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
            
            self.imputation_counts['contact_subgroup'] += 1
            return imputed_mean, combined_ewma_std, 'contact_subgroup'
        
        elif has_subgroup:
            # No contact history, but subgroup available
            self.imputation_counts['only_subgroup'] += 1
            return subgroup_stats['ewma_mean'], subgroup_stats['ewma_std'], 'only_subgroup'
        
        elif has_contact_pair:
            # No subgroup data, use contact's overall history for this pair
            self.imputation_counts['only_contact'] += 1
            return contact_pair_stats['ewma_mean'], contact_pair_stats['ewma_std'], 'only_contact'
        
        elif has_pair:
            # No contact data at all, use overall pair statistics
            self.imputation_counts['only_group'] += 1
            return pair_stats['ewma_mean'], pair_stats['ewma_std'], 'only_group'
        
        else:
            # Insufficient data at all levels
            self.imputation_counts['insufficient_data'] += 1
            return None, None, 'insufficient_data'

    def _process_target_point(self, row, pair_idx, subgroup_idx):
        """
        Override to track ground truth and measure imputation errors.
        Now uses fallback imputation strategy.
        
        CHANGED: Now receives pair_idx parameter
        """
        is_test = row.get('is_test_missing', False)
        ground_truth = row.get('ground_truth_consumption', None)
        
        # Process normally (will be imputed since is_real=False)
        # CHANGED: Pass pair_idx to parent
        result = super()._process_target_point(row, pair_idx, subgroup_idx)
        
        # Track metrics if this was a test point
        if result and is_test and not np.isnan(ground_truth):
            result['is_test_missing'] = True
            result['ground_truth'] = ground_truth
            
            # Store in map for later correction tracking
            self.ground_truth_map[result['index']] = ground_truth
            
            # Record imputation error (initial prediction)
            if result['type'] in ['imputed', 'test_imputed']:
                imputed_val = result['consumption']
                squared_error = (imputed_val - ground_truth) ** 2
                
                # Get imputation source from result
                imputation_source = result.get('imputation_source', 'unknown')
                
                self.imputed_errors.append({
                    'date': row['data'],
                    'hour': row['hour'],
                    'ground_truth': ground_truth,
                    'imputed': imputed_val,
                    'error': squared_error,
                    'index': result['index'],
                    'source': imputation_source
                })
                
                print(f"[IMPUTE] Test point idx={result['index']} at {row['data']} {row['hour']}h: "
                      f"GT={ground_truth:.4f}, Imp={imputed_val:.4f}, Source={imputation_source}")
        
        return result
    
    def _normalize_segment(self):
        """
        Override to track correction errors for test points.
        
        After parent's normalization, check which test points got corrected
        and measure their error against ground truth.
        """
        # Find the segment BEFORE normalization
        start_idx = self.last_real_idx
        end_idx = len(self.results) - 1
        
        # Count test points in this segment
        test_points_in_segment = []
        for i in range(start_idx + 1, end_idx):
            result = self.results[i]
            if result['index'] in self.ground_truth_map:
                test_points_in_segment.append((i, result['index']))
        
        
        # Call parent to do the actual correction
        super()._normalize_segment()
        
        # Now check which test points got corrected
        n_corrected = 0
        for i, result_idx in test_points_in_segment:
            result = self.results[i]
            ground_truth = self.ground_truth_map[result_idx]
            corrected_val = result.get('corrected_consumption')
            
            # Measure correction performance
            if corrected_val is not None and not np.isnan(corrected_val):
                squared_error = (corrected_val - ground_truth) ** 2
                
                # Get imputation source
                imputation_source = result.get('imputation_source', 'unknown')
                
                self.corrected_errors.append({
                    'date': result['data'],
                    'hour': result['hour'],
                    'ground_truth': ground_truth,
                    'corrected': corrected_val,
                    'error': squared_error,
                    'index': result_idx,
                    'source': imputation_source
                })
                
                n_corrected += 1
                print(f"  ✓ Test point idx={result_idx}: GT={ground_truth:.4f}, "
                      f"Imp={result['consumption']:.4f}, Corr={corrected_val:.4f}, "
                      f"RMSE={np.sqrt(squared_error):.4f}, Source={imputation_source}")
    
    
    def get_performance_metrics(self):
        """
        Calculate comprehensive performance metrics.
        Now includes breakdown by imputation type.
        
        Returns
        -------
        dict with RMSE, SSE, counts, and improvement percentage
        """
        metrics = {
            'n_test_points': int(self.all_data['is_test_missing'].sum()),
            'n_imputed': len(self.imputed_errors),
            'n_corrected': len(self.corrected_errors)
        }
        
        # Add imputation counts by type
        metrics['imputation_breakdown'] = self.imputation_counts.copy()
        
        # Imputation metrics
        if self.imputed_errors:
            sse = sum(e['error'] for e in self.imputed_errors)
            metrics['imputed_rmse'] = np.sqrt(sse / len(self.imputed_errors))
            metrics['imputed_sse'] = sse
            metrics['imputed_mean_error'] = np.mean([e['imputed'] - e['ground_truth'] 
                                                      for e in self.imputed_errors])
            
            # Breakdown by imputation source
            metrics['imputed_by_source'] = {}
            source_counts = {}
            for e in self.imputed_errors:
                source = e.get('source', 'unknown')
                if source not in metrics['imputed_by_source']:
                    metrics['imputed_by_source'][source] = []
                    source_counts[source] = 0
                metrics['imputed_by_source'][source].append(e['error'])
                source_counts[source] += 1
            
            # Calculate RMSE for each source
            metrics['imputed_rmse_by_source'] = {}
            for source, errors in metrics['imputed_by_source'].items():
                rmse = np.sqrt(np.mean(errors))
                metrics['imputed_rmse_by_source'][source] = {
                    'rmse': rmse,
                    'count': source_counts[source]
                }
        else:
            metrics['imputed_rmse'] = None
            metrics['imputed_sse'] = None
            metrics['imputed_mean_error'] = None
            metrics['imputed_by_source'] = {}
            metrics['imputed_rmse_by_source'] = {}
        
        # Correction metrics
        if self.corrected_errors:
            sse = sum(e['error'] for e in self.corrected_errors)
            metrics['corrected_rmse'] = np.sqrt(sse / len(self.corrected_errors))
            metrics['corrected_sse'] = sse
            metrics['corrected_mean_error'] = np.mean([e['corrected'] - e['ground_truth'] 
                                                        for e in self.corrected_errors])
            
            # Breakdown by imputation source
            metrics['corrected_by_source'] = {}
            source_counts = {}
            for e in self.corrected_errors:
                source = e.get('source', 'unknown')
                if source not in metrics['corrected_by_source']:
                    metrics['corrected_by_source'][source] = []
                    source_counts[source] = 0
                metrics['corrected_by_source'][source].append(e['error'])
                source_counts[source] += 1
            
            # Calculate RMSE for each source
            metrics['corrected_rmse_by_source'] = {}
            for source, errors in metrics['corrected_by_source'].items():
                rmse = np.sqrt(np.mean(errors))
                metrics['corrected_rmse_by_source'][source] = {
                    'rmse': rmse,
                    'count': source_counts[source]
                }
        else:
            metrics['corrected_rmse'] = None
            metrics['corrected_sse'] = None
            metrics['corrected_mean_error'] = None
            metrics['corrected_by_source'] = {}
            metrics['corrected_rmse_by_source'] = {}
        
        # Improvement calculation
        if metrics['imputed_rmse'] and metrics['corrected_rmse']:
            improvement = (
                (metrics['imputed_rmse'] - metrics['corrected_rmse']) / 
                metrics['imputed_rmse'] * 100
            )
            metrics['improvement_pct'] = improvement
        else:
            metrics['improvement_pct'] = None
        
        return metrics


class TestingVisualizer(SlidingWindowVisualizer):
    """
    Extended visualizer that displays ground truth points and performance
    metrics in real-time, with intelligent fast-forwarding.
    """
    
    def __init__(self, engine, delay=0.1, window_size=100, title=None, 
                 fast_forward_distance=50):
        """
        Parameters
        ----------
        fast_forward_distance : int
            Skip visualization updates when farther than this from any
            ground truth point (improves performance)
        """
        super().__init__(engine, delay, window_size, title)
        self.fast_forward_distance = fast_forward_distance
        self.ground_truth_indices = set()
        
    def _get_window_data(self, current_target_idx):
        """Override to include ground truth points in window."""
        window_data = super()._get_window_data(current_target_idx)
        
        # Add ground truth series
        window_data['truth_x'] = []
        window_data['truth_y'] = []
        window_data['truth_ts'] = []
        
        start_idx = max(0, current_target_idx - self.window_size + 1)
        
        for result in self.engine.results[start_idx:current_target_idx + 1]:
            result_idx = result['index']
            if result_idx in self.engine.ground_truth_map:
                window_data['truth_x'].append(result_idx)
                window_data['truth_y'].append(self.engine.ground_truth_map[result_idx])
                window_data['truth_ts'].append(result['data'])
        
        return window_data
    
    def _update_plot(self, current_target_idx, all_processed, target_processed):
        """Override to display ground truth and live metrics."""
        self.ax.clear()
        window = self._get_window_data(current_target_idx)
        
        # Plot ground truth (green stars) - render first for background
        if window['truth_x']:
            self.ax.scatter(
                window['truth_x'], window['truth_y'], 
                c='lime', s=120, label='Ground Truth', 
                zorder=2, alpha=0.95, marker='*', 
                edgecolors='darkgreen', linewidths=2.5
            )
        
        # Plot real observations (blue)
        if window['real_x']:
            self.ax.scatter(
                window['real_x'], window['real_y'], 
                c='blue', s=50, label='Real', 
                zorder=3, alpha=0.8
            )
        
        # Plot imputed values with uncertainty (red)
        if window['imputed_x']:
            self.ax.errorbar(
                window['imputed_x'], window['imputed_y'],
                yerr=window['imputed_yerr'],
                fmt='o', c='red', markersize=6, 
                label='Imputed (±2σ EWMA)',
                alpha=0.6, capsize=3, zorder=4
            )
        
        # Plot corrected values (yellow) - HIGH z-order to appear on top
        if window['corrected_x']:
            self.ax.scatter(
                window['corrected_x'], window['corrected_y'],
                c='gold', s=90, label='Corrected',
                zorder=6, alpha=1.0, 
                edgecolors='darkorange', linewidths=2.5
            )
        
        # Build title with live metrics
        metrics = self.engine.get_performance_metrics()
        title_parts = [
            f'{self.title}',
            f'All: {all_processed}/{len(self.engine.all_data)}',
            f'Target: {target_processed}/{self.engine.target_num}'
        ]
        
        if metrics['imputed_rmse'] is not None:
            title_parts.append(f"Imp RMSE: {metrics['imputed_rmse']:.4f}")
        if metrics['corrected_rmse'] is not None:
            title_parts.append(f"Corr RMSE: {metrics['corrected_rmse']:.4f}")
        
        self.ax.set_title(' | '.join(title_parts), fontsize=14)
        self.ax.set_xlabel('Date', fontsize=12)
        self.ax.set_ylabel('Consumption Difference', fontsize=12)
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        
        # Configure x-axis
        start_idx = max(0, current_target_idx - self.window_size + 1)
        self.ax.set_xlim(start_idx - 5, current_target_idx + 5)
        
        # Format date labels
        all_x = (window['real_x'] + window['imputed_x'] + 
                 window['corrected_x'] + window['truth_x'])
        all_ts = (window['real_ts'] + window['imputed_ts'] + 
                  window['corrected_ts'] + window['truth_ts'])
        
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
                self.ax.set_xticks(list(date_labels.keys()))
                self.ax.set_xticklabels(
                    list(date_labels.values()), 
                    rotation=45, ha='right'
                )
        
        # Auto-scale y-axis with padding
        all_y = (window['real_y'] + window['imputed_y'] + 
                 window['corrected_y'] + window['truth_y'])
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            padding = max((y_max - y_min) * 0.1, 0.1)
            self.ax.set_ylim(y_min - padding, y_max + padding)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.000001)
    
    def _is_near_ground_truth(self, current_idx):
        """Check if current position is near any ground truth point."""
        for gt_idx in self.ground_truth_indices:
            if abs(current_idx - gt_idx) <= self.fast_forward_distance:
                return True
        return False
    
    def run(self):
        """
        Execute online visualization with intelligent fast-forwarding
        and comprehensive performance reporting.
        """
        print(f"\n{'='*60}")
        print(f"Starting Online Imputation Test: {self.engine.contact_id}")
        print(f"{'='*60}")
        print(f"Total points (all contacts): {len(self.engine.all_data)}")
        print(f"Target contact points: {self.engine.target_num}")
        print(f"Window size: {self.window_size}")
        print(f"Delay: {self.delay}s per update")
        print(f"Fast-forward threshold: {self.fast_forward_distance} points")
        print(f"EWMA alpha: {self.engine.ewma_alpha}")
        print(f"{'='*60}\n")
        
        all_processed = 0
        target_processed = 0
        skipped_count = 0
        fast_forwarded = 0
        
        # Process all points
        for all_idx, is_target, result in self.engine.process_all_points():
            all_processed += 1
            
            if is_target:
                target_processed += 1
                
                if result is None:
                    skipped_count += 1
                    continue
                
                # Track ground truth indices
                if result['index'] in self.engine.ground_truth_map:
                    self.ground_truth_indices.add(result['index'])
                
                # Store by type
                if result['type'] == 'real':
                    self.all_data['real'].append(result)
                elif result['type'] == 'imputed':
                    self.all_data['imputed'].append(result)
                
                # Decide whether to update visualization
                current_idx = len(self.engine.results) - 1
                is_near_gt = self._is_near_ground_truth(current_idx)
                
                should_update = (
                    is_near_gt or 
                    target_processed % 100 == 0
                )
                
                if should_update and len(self.engine.results) > 0:
                    self._update_plot(current_idx, all_processed, target_processed)
                    
                    if self.delay and is_near_gt:
                        time.sleep(self.delay)
                else:
                    fast_forwarded += 1
        
        # Collect corrected points
        for r in self.engine.results:
            if r['corrected_consumption'] is not None:
                self.all_data['corrected'].append(r)
        
        # Final plot update
        if len(self.engine.results) > 0:
            final_idx = len(self.engine.results) - 1
            self._update_plot(final_idx, all_processed, target_processed)
        
        # Print summary
        self._print_summary(
            all_processed, target_processed, 
            skipped_count, fast_forwarded
        )
        
        # Show error analysis
        self._plot_error_distribution()
        
        plt.ioff()
        plt.show()
        
        return self.engine.results
    
    def _print_summary(self, all_proc, target_proc, skipped, fast_fwd):
        """Print comprehensive performance summary with fallback breakdown."""
        metrics = self.engine.get_performance_metrics()
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed (all contacts): {all_proc}")
        print(f"Target contact processed: {target_proc}")
        print(f"Skipped: {skipped}")
        print(f"Fast-forwarded (not visualized): {fast_fwd}")
        print(f"Real points: {len(self.all_data['real'])}")
        print(f"Imputed points: {len(self.all_data['imputed'])}")
        print(f"Corrected points: {len(self.all_data['corrected'])}")
        
        print(f"\n{'='*60}")
        print("IMPUTATION BREAKDOWN BY TYPE")
        print(f"{'='*60}")
        breakdown = metrics['imputation_breakdown']
        for imputation_type, count in breakdown.items():
            print(f"  {imputation_type}: {count}")
        print(f"  TOTAL: {sum(breakdown.values())}")
        
        print(f"\n{'='*60}")
        print("PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"Test points: {metrics['n_test_points']}")
        print(f"Successfully imputed: {metrics['n_imputed']}")
        print(f"Successfully corrected: {metrics['n_corrected']}")
        
        if metrics['n_corrected'] < metrics['n_imputed']:
            missing = metrics['n_imputed'] - metrics['n_corrected']
            print(f"⚠️  WARNING: {missing} test points were imputed but NOT corrected!")
        
        if metrics['imputed_rmse'] is not None:
            print(f"\nImputation Performance:")
            print(f"  Overall RMSE: {metrics['imputed_rmse']:.6f}")
            print(f"  Overall SSE:  {metrics['imputed_sse']:.6f}")
            print(f"  Mean Error: {metrics['imputed_mean_error']:.6f}")
            
            if metrics['imputed_rmse_by_source']:
                print(f"\n  Breakdown by imputation source:")
                for source, data in metrics['imputed_rmse_by_source'].items():
                    print(f"    {source}: RMSE={data['rmse']:.6f}, n={data['count']}")
        else:
            print(f"\nImputation Performance: N/A")
        
        if metrics['corrected_rmse'] is not None:
            print(f"\nCorrection Performance:")
            print(f"  Overall RMSE: {metrics['corrected_rmse']:.6f}")
            print(f"  Overall SSE:  {metrics['corrected_sse']:.6f}")
            print(f"  Mean Error: {metrics['corrected_mean_error']:.6f}")
            
            if metrics['corrected_rmse_by_source']:
                print(f"\n  Breakdown by imputation source:")
                for source, data in metrics['corrected_rmse_by_source'].items():
                    print(f"    {source}: RMSE={data['rmse']:.6f}, n={data['count']}")
        else:
            print(f"\nCorrection Performance: N/A (no corrections computed)")
        
        if metrics.get('improvement_pct') is not None:
            print(f"\n{'🎯' if metrics['improvement_pct'] > 0 else '⚠️ '} "
                  f"Improvement: {metrics['improvement_pct']:+.2f}%")
        
        print(f"{'='*60}")
        print("\n📊 FINAL RESULTS:")
        if metrics['imputed_rmse']:
            print(f"   Imputed RMSE:   {metrics['imputed_rmse']:.6f}")
        if metrics['corrected_rmse']:
            print(f"   Corrected RMSE: {metrics['corrected_rmse']:.6f}")
        print(f"{'='*60}\n")
    
    def _plot_error_distribution(self):
        """Generate comprehensive error analysis plots."""
        imputed_errs = self.engine.imputed_errors
        corrected_errs = self.engine.corrected_errors
        
        if not imputed_errs and not corrected_errs:
            print("⚠️  No error data available for plotting")
            return
        
        # Calculate actual errors (signed)
        imputed_actual = [e['imputed'] - e['ground_truth'] for e in imputed_errs] if imputed_errs else []
        corrected_actual = [e['corrected'] - e['ground_truth'] for e in corrected_errs] if corrected_errs else []
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Error Distribution Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Imputed errors histogram
        ax = axes[0, 0]
        if imputed_actual:
            ax.hist(imputed_actual, bins=30, color='red', 
                   alpha=0.7, edgecolor='black')
            ax.axvline(0, color='black', linestyle='--', 
                      linewidth=2, label='Zero Error')
            mean_imp = np.mean(imputed_actual)
            ax.axvline(mean_imp, color='darkred', linestyle='-', 
                      linewidth=2, label=f'Mean: {mean_imp:.4f}')
            
            ax.set_xlabel('Error (Imputed - Ground Truth)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Imputed Errors (n={len(imputed_actual)})', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics box
            stats = (
                f'Mean: {mean_imp:.4f}\n'
                f'Std: {np.std(imputed_actual):.4f}\n'
                f'Median: {np.median(imputed_actual):.4f}\n'
                f'RMSE: {np.sqrt(np.mean([e**2 for e in imputed_actual])):.4f}'
            )
            ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No Imputed Data', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Imputed Errors', fontweight='bold')
        
        # Plot 2: Corrected errors histogram
        ax = axes[0, 1]
        if corrected_actual:
            ax.hist(corrected_actual, bins=30, color='orange',
                   alpha=0.7, edgecolor='black')
            ax.axvline(0, color='black', linestyle='--',
                      linewidth=2, label='Zero Error')
            mean_corr = np.mean(corrected_actual)
            ax.axvline(mean_corr, color='darkorange', linestyle='-',
                      linewidth=2, label=f'Mean: {mean_corr:.4f}')
            
            ax.set_xlabel('Error (Corrected - Ground Truth)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Corrected Errors (n={len(corrected_actual)})',
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics box
            stats = (
                f'Mean: {mean_corr:.4f}\n'
                f'Std: {np.std(corrected_actual):.4f}\n'
                f'Median: {np.median(corrected_actual):.4f}\n'
                f'RMSE: {np.sqrt(np.mean([e**2 for e in corrected_actual])):.4f}'
            )
            ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No Corrected Data\n(Check if test points are between real points)', 
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_title('Corrected Errors', fontweight='bold')
        
        # Plot 3: Squared errors comparison
        ax = axes[1, 0]
        if imputed_errs and corrected_errs:
            imp_se = [e['error'] for e in imputed_errs]
            corr_se = [e['error'] for e in corrected_errs]
            
            ax.hist(imp_se, bins=30, color='red', alpha=0.5,
                   edgecolor='black', label=f'Imputed (n={len(imp_se)})')
            ax.hist(corr_se, bins=30, color='orange', alpha=0.5,
                   edgecolor='black', label=f'Corrected (n={len(corr_se)})')
            ax.set_title('Squared Error Distribution', fontweight='bold')
            ax.set_xlabel('Squared Error')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        elif imputed_errs:
            imp_se = [e['error'] for e in imputed_errs]
            ax.hist(imp_se, bins=30, color='red', alpha=0.7, edgecolor='black')
            ax.set_title(f'Imputed Squared Error (n={len(imp_se)})', 
                        fontweight='bold')
            ax.set_xlabel('Squared Error')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        elif corrected_errs:
            corr_se = [e['error'] for e in corrected_errs]
            ax.hist(corr_se, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax.set_title(f'Corrected Squared Error (n={len(corr_se)})',
                        fontweight='bold')
            ax.set_xlabel('Squared Error')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Error Data', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Squared Error Distribution', fontweight='bold')
        
        # Plot 4: Absolute error comparison (box plot)
        ax = axes[1, 1]
        if imputed_actual and corrected_actual:
            abs_imp = [abs(e) for e in imputed_actual]
            abs_corr = [abs(e) for e in corrected_actual]
            
            bp = ax.boxplot([abs_imp, abs_corr], 
                           tick_labels=['Imputed', 'Corrected'],
                           patch_artist=True, showmeans=True)
            
            bp['boxes'][0].set_facecolor('red')
            bp['boxes'][0].set_alpha(0.5)
            if len(bp['boxes']) > 1:
                bp['boxes'][1].set_facecolor('orange')
                bp['boxes'][1].set_alpha(0.5)
            
            ax.set_ylabel('Absolute Error')
            ax.set_title('Absolute Error Comparison', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            stats = (
                f'Imputed:\n'
                f'  Mean: {np.mean(abs_imp):.4f}\n'
                f'  Median: {np.median(abs_imp):.4f}\n'
                f'Corrected:\n'
                f'  Mean: {np.mean(abs_corr):.4f}\n'
                f'  Median: {np.median(abs_corr):.4f}'
            )
            ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        elif imputed_actual:
            abs_imp = [abs(e) for e in imputed_actual]
            bp = ax.boxplot([abs_imp], tick_labels=['Imputed'],
                           patch_artist=True, showmeans=True)
            bp['boxes'][0].set_facecolor('red')
            bp['boxes'][0].set_alpha(0.5)
            ax.set_ylabel('Absolute Error')
            ax.set_title('Absolute Error (Imputed Only)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        elif corrected_actual:
            abs_corr = [abs(e) for e in corrected_actual]
            bp = ax.boxplot([abs_corr], tick_labels=['Corrected'],
                           patch_artist=True, showmeans=True)
            bp['boxes'][0].set_facecolor('orange')
            bp['boxes'][0].set_alpha(0.5)
            ax.set_ylabel('Absolute Error')
            ax.set_title('Absolute Error (Corrected Only)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No Error Data', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Absolute Error Comparison', fontweight='bold')
        
        plt.tight_layout()
        plt.show(block=False)


# Example usage
if __name__ == "__main__":
    import pickle

    # Load pre-processed variables
    with open("./variables/variables.pkl", "rb") as f:
        data = pickle.load(f)

    subgroups = data["subgroups"]

    # Configuration
    contact_id = 'LC41847'
    test_config_path = 'info/test_config.json'
    
    # Initialize testing engine 
    engine = OnlineImputationTester(
        subgroups_list=subgroups,
        contact_id=contact_id,
        test_config_path=test_config_path,
        warmup_contact=0,
        warmup_subgroup=0,
        weight_contact=0.7,
        ewma_alpha=0.3
    )
    engine._build()

    # Run visualization with intelligent fast-forwarding
    visualizer = TestingVisualizer(
        engine, 
        delay=0.01,
        window_size=100,
        fast_forward_distance=50
    )
    
    results = visualizer.run()