import numpy as np
import pandas as pd
import pickle
import sys
import os
import io
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


def hyperparameter_tuning(subgroups_list, idx2pair, pair2idx,
                         contact_id, test_config_path, 
                         warmup_contact=0, warmup_subgroup=0):
    """
    Perform grid search over weight_contact and ewma_alpha.
    
    Parameters
    ----------
    weight_contact : tested from 0.0 to 1.0 with step 0.1
    ewma_alpha : tested from 0.0 to 1.0 with step 0.1
    
    Returns
    -------
    pd.DataFrame with results for all combinations
    """
    # Define hyperparameter grid
    weight_contacts = np.arange(0.0, 1.1, 0.1)
    ewma_alphas = np.arange(0.0, 1.1, 0.1)
    
    # Store results
    results = []
    
    total_combinations = len(weight_contacts) * len(ewma_alphas)
    print(f"{'='*70}")
    print(f"HYPERPARAMETER TUNING")
    print(f"{'='*70}")
    print(f"Contact ID: {contact_id}")
    print(f"Total combinations to test: {total_combinations}")
    print(f"weight_contact: {weight_contacts[0]:.1f} to {weight_contacts[-1]:.1f} (step 0.1)")
    print(f"ewma_alpha: {ewma_alphas[0]:.1f} to {ewma_alphas[-1]:.1f} (step 0.1)")
    print(f"{'='*70}\n")
    
    current = 0

    # Grid search
    for weight_contact, ewma_alpha in product(weight_contacts, ewma_alphas):
        current += 1
        
        print(f"[{current}/{total_combinations}] Testing: "
              f"weight_contact={weight_contact:.1f}, ewma_alpha={ewma_alpha:.1f}", 
              end=" ... ")
        
        try:
            # CHANGED: Removed mapa_contact_id parameter
            engine = OnlineImputationTester(
                subgroups_list=subgroups_list,
                idx2pair=idx2pair,
                pair2idx=pair2idx,
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
            
            # Store results
            result_entry = {
                'weight_contact': weight_contact,
                'ewma_alpha': ewma_alpha,
                'corrected_rmse': metrics['corrected_rmse'],
                'imputed_rmse': metrics['imputed_rmse'],
                'n_test_points': metrics['n_test_points'],
                'n_imputed': metrics['n_imputed'],
                'n_corrected': metrics['n_corrected'],
                'improvement_pct': metrics['improvement_pct']
            }
            
            results.append(result_entry)
            
            # Print result
            if metrics['corrected_rmse'] is not None:
                print(f"OK Corrected RMSE: {metrics['corrected_rmse']:.6f}")
            else:
                print("X No corrected points")
                
        except Exception as e:
            print(f"X ERROR: {str(e)}")
            results.append({
                'weight_contact': weight_contact,
                'ewma_alpha': ewma_alpha,
                'corrected_rmse': None,
                'imputed_rmse': None,
                'n_test_points': None,
                'n_imputed': None,
                'n_corrected': None,
                'improvement_pct': None,
                'error': str(e)
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best parameters
    valid_results = results_df[results_df['corrected_rmse'].notna()]
    
    if len(valid_results) > 0:
        best_idx = valid_results['corrected_rmse'].idxmin()
        best_result = valid_results.loc[best_idx]
        
        print(f"\n{'='*70}")
        print("BEST HYPERPARAMETERS")
        print(f"{'='*70}")
        print(f"weight_contact: {best_result['weight_contact']:.1f}")
        print(f"ewma_alpha: {best_result['ewma_alpha']:.1f}")
        print(f"Corrected RMSE: {best_result['corrected_rmse']:.6f}")
        print(f"Imputed RMSE: {best_result['imputed_rmse']:.6f}")
        print(f"Improvement: {best_result['improvement_pct']:.2f}%")
        print(f"Test points corrected: {best_result['n_corrected']}/{best_result['n_test_points']}")
        print(f"{'='*70}\n")
        
        # Show top 5 configurations
        print("Top 5 Configurations:")
        print("-" * 70)
        top_5 = valid_results.nsmallest(5, 'corrected_rmse')[
            ['weight_contact', 'ewma_alpha', 'corrected_rmse', 'improvement_pct']
        ]
        print(top_5.to_string(index=False))
        print(f"{'='*70}\n")
    else:
        print("\n⚠️  WARNING: No valid results with corrected RMSE!")
    
    return results_df


if __name__ == "__main__":
    # Load pre-processed variables
    with open("./variables/variables.pkl", "rb") as f:
        data = pickle.load(f)

    idx2pair = data["idx2pair"]
    pair2idx = data["pair2idx"]
    subgroups = data["subgroups"]

    # Configuration
    contact_id = 'LC449452'
    test_config_path = 'info/test_config.json'
    
    # Run hyperparameter tuning (CHANGED: removed mapa_contact_id)
    results_df = hyperparameter_tuning(
        subgroups_list=subgroups,
        idx2pair=idx2pair,
        pair2idx=pair2idx,
        contact_id=contact_id,
        test_config_path=test_config_path,
        warmup_contact=0,
        warmup_subgroup=0
    )
    
    # Save results
    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    print(f"Results saved to 'hyperparameter_tuning_results.csv'")
    
    # Create pivot table for visualization
    valid_results = results_df[results_df['corrected_rmse'].notna()]
    
    if len(valid_results) > 0:
        pivot_table = valid_results.pivot_table(
            values='corrected_rmse',
            index='weight_contact',
            columns='ewma_alpha',
            aggfunc='mean'
        )
        
        print("\nPivot Table (Corrected RMSE):")
        print("=" * 70)
        print(pivot_table.to_string())
        
        # Save pivot table
        pivot_table.to_csv('hyperparameter_pivot_table.csv')
        print(f"\nPivot table saved to 'hyperparameter_pivot_table.csv'")