import numpy as np
import pandas as pd
import pickle
import sys
import io
import json
from itertools import product
from test_performance import OnlineImputationTester


def run_silent(engine):
    """
    Run the engine without any visualization or prints.
    
    Parameters
    ----------
    engine : OnlineImputationTester
        The engine to run
        
    Returns
    -------
    list
        Results from the engine
    """
    # Suppress all prints by redirecting stdout to a dummy buffer
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        for all_idx, is_target, result in engine.process_all_points():
            pass  # Just process, no visualization
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    return engine.results


def hyperparameter_tuning_single_lc(subgroups_list, contact_id, tipo_consumo, calibre,
                                     test_config_path, warmup_contact=0, warmup_subgroup=0):
    """
    Perform grid search over weight_contact and ewma_alpha for a single LC.
    
    Parameters
    ----------
    subgroups_list : list
        List of subgroups
    contact_id : str
        Contact ID to test
    tipo_consumo : str
        Type of consumption
    calibre : str
        Caliber
    test_config_path : str
        Path to test config JSON
    warmup_contact : int
        Warmup for contact
    warmup_subgroup : int
        Warmup for subgroup
    
    Returns
    -------
    dict
        Best hyperparameters and metrics for this LC
    """
    # Define hyperparameter grid
    weight_contacts = np.arange(0.0, 1.1, 0.1)
    ewma_alphas = np.arange(0.0, 1.1, 0.1)
    
    best_result = None
    best_rmse = float('inf')
    
    # Grid search
    for weight_contact, ewma_alpha in product(weight_contacts, ewma_alphas):
        try:
            engine = OnlineImputationTester(
                subgroups_list=subgroups_list,
                contact_id=contact_id,
                test_config_path=test_config_path,
                warmup_contact=warmup_contact,
                warmup_subgroup=warmup_subgroup,
                weight_contact=weight_contact,
                ewma_alpha=ewma_alpha
            )
            engine._build()
            
            # Run without visualization
            run_silent(engine)
            
            # Get metrics
            metrics = engine.get_performance_metrics()
            
            # Check if this is the best so far
            if metrics['corrected_rmse'] is not None and metrics['corrected_rmse'] < best_rmse:
                best_rmse = metrics['corrected_rmse']
                
                # Calculate percentage errors based on consumption range
                # Get consumption values from results to calculate range
                consumptions = []
                for result in engine.results:
                    if result['is_target'] and result['true_value'] is not None:
                        consumptions.append(result['true_value'])
                
                consumption_range = max(consumptions) - min(consumptions) if consumptions else None
                
                imputed_pct_error = None
                corrected_pct_error = None
                if consumption_range and consumption_range > 0:
                    imputed_pct_error = (metrics['imputed_rmse'] / consumption_range) * 100
                    corrected_pct_error = (metrics['corrected_rmse'] / consumption_range) * 100
                
                best_result = {
                    'tipo_consumo': tipo_consumo,
                    'calibre': calibre,
                    'contact_id': contact_id,
                    'best_weight_contact': weight_contact,
                    'best_ewma_alpha': ewma_alpha,
                    'rmse_imputed': metrics['imputed_rmse'],
                    'rmse_corrected': metrics['corrected_rmse'],
                    'pct_error_imputed': imputed_pct_error,
                    'pct_error_corrected': corrected_pct_error,
                    'consumption_range': consumption_range,
                    'improvement_pct': metrics['improvement_pct'],
                    'n_test_points': metrics['n_test_points'],
                    'n_imputed': metrics['n_imputed'],
                    'n_corrected': metrics['n_corrected']
                }
                
        except Exception as e:
            continue  # Skip this combination if it fails
    
    return best_result


def hyperparameter_tuning_all_lcs(subgroups_list, test_config_path, 
                                   warmup_contact=0, warmup_subgroup=0):
    """
    Perform hyperparameter tuning for all LCs in the test config.
    
    Parameters
    ----------
    subgroups_list : list
        List of subgroups
    test_config_path : str
        Path to test config JSON
    warmup_contact : int
        Warmup for contact
    warmup_subgroup : int
        Warmup for subgroup
    
    Returns
    -------
    pd.DataFrame
        Results for all LCs
    """
    # Load test config
    with open(test_config_path, 'r') as f:
        test_config = json.load(f)
    
    total_lcs = len(test_config)
    print(f"{'='*70}")
    print(f"HYPERPARAMETER TUNING FOR ALL LCS")
    print(f"{'='*70}")
    print(f"Total LCs to process: {total_lcs}")
    print(f"Hyperparameter grid: weight_contact [0.0-1.0], ewma_alpha [0.0-1.0] (step 0.1)")
    print(f"Total combinations per LC: {11 * 11} = 121")
    print(f"{'='*70}\n")
    
    results = []
    
    for idx, (contact_id, lc_info) in enumerate(test_config.items(), 1):
        tipo_consumo = lc_info['tipo_consumo']
        calibre = lc_info['calibre']
        
        print(f"[{idx}/{total_lcs}] Processing LC: {contact_id}")
        print(f"  tipo_consumo={tipo_consumo}, calibre={calibre}")
        
        # Run hyperparameter tuning for this LC
        best_result = hyperparameter_tuning_single_lc(
            subgroups_list=subgroups_list,
            contact_id=contact_id,
            tipo_consumo=tipo_consumo,
            calibre=calibre,
            test_config_path=test_config_path,
            warmup_contact=warmup_contact,
            warmup_subgroup=warmup_subgroup
        )
        
        if best_result:
            results.append(best_result)
            print(f"  ✓ Best: weight_contact={best_result['best_weight_contact']:.1f}, "
                  f"ewma_alpha={best_result['best_ewma_alpha']:.1f}")
            print(f"  RMSE: imputed={best_result['rmse_imputed']:.4f}, "
                  f"corrected={best_result['rmse_corrected']:.4f} "
                  f"(improvement: {best_result['improvement_pct']:.2f}%)")
            if best_result['pct_error_corrected'] is not None:
                print(f"  %Error: imputed={best_result['pct_error_imputed']:.2f}%, "
                      f"corrected={best_result['pct_error_corrected']:.2f}%")
        else:
            print(f"  ✗ No valid results for this LC")
            results.append({
                'tipo_consumo': tipo_consumo,
                'calibre': calibre,
                'contact_id': contact_id,
                'best_weight_contact': None,
                'best_ewma_alpha': None,
                'rmse_imputed': None,
                'rmse_corrected': None,
                'pct_error_imputed': None,
                'pct_error_corrected': None,
                'consumption_range': None,
                'improvement_pct': None,
                'n_test_points': None,
                'n_imputed': None,
                'n_corrected': None
            })
        
        print()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    valid_results = results_df[results_df['rmse_corrected'].notna()]
    print(f"Total LCs processed: {total_lcs}")
    print(f"Successful tunings: {len(valid_results)}")
    print(f"Failed tunings: {total_lcs - len(valid_results)}")
    
    if len(valid_results) > 0:
        print(f"\nAverage RMSE (corrected): {valid_results['rmse_corrected'].mean():.4f}")
        print(f"Average improvement: {valid_results['improvement_pct'].mean():.2f}%")
        if valid_results['pct_error_corrected'].notna().any():
            print(f"Average %Error (corrected): {valid_results['pct_error_corrected'].mean():.2f}%")
        
        print(f"\nBest performing LC:")
        best_lc = valid_results.loc[valid_results['rmse_corrected'].idxmin()]
        print(f"  Contact ID: {best_lc['contact_id']}")
        print(f"  RMSE corrected: {best_lc['rmse_corrected']:.4f}")
        print(f"  Parameters: weight_contact={best_lc['best_weight_contact']:.1f}, "
              f"ewma_alpha={best_lc['best_ewma_alpha']:.1f}")
    
    print(f"{'='*70}\n")
    
    return results_df


if __name__ == "__main__":
    # Load pre-processed variables
    with open("./variables/variables.pkl", "rb") as f:
        data = pickle.load(f)

    subgroups = data["subgroups"]

    # Configuration
    test_config_path = 'info/test_config.json'
    
    # Run hyperparameter tuning for all LCs
    results_df = hyperparameter_tuning_all_lcs(
        subgroups_list=subgroups,
        test_config_path=test_config_path,
        warmup_contact=0,
        warmup_subgroup=0
    )
    
    # Save results
    output_file = 'hyperparameter_tuning_all_lcs.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'")
    
    # Save summary statistics by group
    valid_results = results_df[results_df['rmse_corrected'].notna()]
    
    if len(valid_results) > 0:
        summary_by_group = valid_results.groupby(['tipo_consumo', 'calibre']).agg({
            'rmse_corrected': ['mean', 'std', 'min', 'max'],
            'improvement_pct': ['mean', 'std'],
            'pct_error_corrected': ['mean', 'std'],
            'contact_id': 'count'
        }).round(4)
        
        summary_by_group.columns = ['_'.join(col).strip() for col in summary_by_group.columns.values]
        summary_by_group = summary_by_group.rename(columns={'contact_id_count': 'n_lcs'})
        
        summary_file = 'hyperparameter_summary_by_group.csv'
        summary_by_group.to_csv(summary_file)
        print(f"Summary by group saved to '{summary_file}'")