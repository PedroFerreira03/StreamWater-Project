import pandas as pd
import json
from datetime import datetime
from collections import defaultdict


def parse_column_datetime(col_name):
    """Parse column name in format 'YYYY-MM-DD_HH:MM' to datetime."""
    try:
        return datetime.strptime(col_name, '%Y-%m-%d_%H:%M')
    except:
        return None


def format_date_dmy(dt):
    """Format datetime to 'DD/MM/YYYY' format."""
    return dt.strftime('%d/%m/%Y')


def count_available_data(row, target_columns):
    """
    Count how many target columns have non-NA data.
    """
    count = 0
    for col in target_columns:
        if col in row.index and pd.notna(row[col]):
            count += 1
    return count


def count_nonzero_data(row, target_columns):
    """
    Count how many target columns have non-NA and non-zero data.
    """
    count = 0
    for col in target_columns:
        if col in row.index and pd.notna(row[col]) and row[col] != 0:
            count += 1
    return count


def generate_test_config(input_file, output_json, tipo_consumo_col='tipo_consumo', 
                         calibre_col='calibre', lc_col='contact_id'):
    """
    Generate test configuration JSON for validation from wide-format data.
    
    Parameters:
    -----------
    input_file : str
        Path to input Excel or CSV file where:
        - Rows = LCs
        - Columns = datetime values (YYYY-MM-DD_HH:MM format) containing consumption
        - Metadata columns: LC identifier, tipo_consumo, calibre
    output_json : str
        Path to save output JSON file
    """

    tipo_consumo_col = 'tipo_consumo'
    calibre_col = 'calibre'
    lc_col = 'contact_id'

    print(f"Loading data from {input_file}...")
    
    # Check file extension and read accordingly
    df = pd.read_excel(input_file, engine='openpyxl')
    print(f"✓ Loaded Excel file")
    
    # Identify metadata columns and datetime columns
    existing_metadata = [lc_col, tipo_consumo_col, calibre_col]
    
    # Parse datetime columns
    datetime_columns = {}
    for col in df.columns:
        if col not in existing_metadata:
            dt = parse_column_datetime(col)
            if dt is not None:
                datetime_columns[col] = dt
    
    # Define target dates and hours
    target_dates_hours = {
        '2024-11-05': list(range(1, 13)) + list(range(20, 24)),  # 1-12, 20-23
        '2024-11-08': list(range(1, 13)) + list(range(20, 24)),  # 1-12, 20-23
        '2024-08-05': list(range(1, 13)) + list(range(20, 24)),  # 1-12, 20-23
        '2024-08-08': list(range(1, 13)) + list(range(20, 24)),  # 1-12, 20-23
    }
    
    # Find column names for target datetime combinations
    target_columns_by_date = defaultdict(list)
    column_to_datetime = {}
    for col, dt in datetime_columns.items():
        date_key = dt.strftime('%Y-%m-%d')
        if date_key in target_dates_hours:
            if dt.hour in target_dates_hours[date_key]:
                target_columns_by_date[date_key].append(col)
                column_to_datetime[col] = dt
    
    # Flatten to get all required columns
    all_target_columns = []
    for cols in target_columns_by_date.values():
        all_target_columns.extend(cols)
    
    # Get unique pairs (exclude NAs)
    if tipo_consumo_col in df.columns and calibre_col in df.columns:
        # Filter out rows where either tipo_consumo or calibre is NA
        valid_pairs_df = df.dropna(subset=[tipo_consumo_col, calibre_col])
        pairs = valid_pairs_df[[tipo_consumo_col, calibre_col]].drop_duplicates()
        print(f"\nFound {len(pairs)} unique (tipo_consumo, calibre) pairs (excluding NAs)")
    
    config = {}
    
    for _, pair_row in pairs.iterrows():
        tipo_consumo = pair_row[tipo_consumo_col] if tipo_consumo_col in pair_row.index else 'Unknown'
        calibre = pair_row[calibre_col] if calibre_col in pair_row.index else 'Unknown'
        
        print(f"\nProcessing pair: (tipo_consumo={tipo_consumo}, calibre={calibre})")
        
        # Filter data for this pair
        if tipo_consumo_col in df.columns and calibre_col in df.columns:
            pair_data = df[
                (df[tipo_consumo_col] == tipo_consumo) & 
                (df[calibre_col] == calibre)
            ]
        else:
            pair_data = df
        
        print(f"  Found {len(pair_data)} LCs for this pair")
        
        # Find LC with most complete data, prioritizing available count then non-zero values
        # Group by contact_id to aggregate across multiple contadores
        best_lc = None
        best_nonzero_count = 0
        best_available_count = 0
        
        for lc_id in pair_data[lc_col].unique():
            # Get all rows for this LC (may have multiple contadores)
            lc_rows = pair_data[pair_data[lc_col] == lc_id]
            
            # Aggregate: a column has data if ANY row has non-NA data for it
            available_count = 0
            nonzero_count = 0
            
            for col in all_target_columns:
                # Check if any row has non-NA data for this column
                has_data = lc_rows[col].notna().any() if col in lc_rows.columns else False
                if has_data:
                    available_count += 1
                    # Check if any row has non-zero data
                    has_nonzero = (lc_rows[col].notna() & (lc_rows[col] != 0)).any()
                    if has_nonzero:
                        nonzero_count += 1
            
            # Prioritize: first by available count, then by non-zero count
            if (available_count > best_available_count) or \
               (available_count == best_available_count and nonzero_count > best_nonzero_count):
                best_nonzero_count = nonzero_count
                best_available_count = available_count
                best_lc = lc_id
        
        if best_lc is None:
            print(f"  ✗ No LC found with any data for this pair")
            continue
        
        coverage_pct = (best_available_count / len(all_target_columns) * 100) if all_target_columns else 0
        nonzero_pct = (best_nonzero_count / len(all_target_columns) * 100) if all_target_columns else 0
        
        if best_available_count == len(all_target_columns):
            print(f"  ✓ Selected LC: {best_lc} (100% coverage, {nonzero_pct:.1f}% non-zero)")
        else:
            print(f"  ✓ Selected LC: {best_lc} ({coverage_pct:.1f}% coverage, {nonzero_pct:.1f}% non-zero - {best_available_count}/{len(all_target_columns)} data points)")
        
        found_lc = best_lc
        
        # Get all rows for the selected LC (may have multiple contadores)
        # IMPORTANT: Only use rows that match the current (tipo_consumo, calibre) pair
        lc_rows = df[
            (df[lc_col] == found_lc) & 
            (df[tipo_consumo_col] == tipo_consumo) & 
            (df[calibre_col] == calibre)
        ]
        
        # Build configuration for this LC - only include hours with actual data
        lc_dates = {}
        
        for date_str in sorted(target_dates_hours.keys()):
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
            date_key = format_date_dmy(target_date)
            
            # Get columns for this date
            date_columns = target_columns_by_date.get(date_str, [])
            
            if not date_columns:
                print(f"    WARNING: No columns found for date {date_str}")
                continue
            
            # Find which hours have data for this LC (across all contadores)
            covered_hours = []
            for col in date_columns:
                if col not in column_to_datetime:
                    print(f"    WARNING: Column {col} not in column_to_datetime mapping")
                    continue
                
                # Check if ANY row for this LC has non-NA data for this column
                if col in lc_rows.columns and lc_rows[col].notna().any():
                    dt = column_to_datetime[col]
                    covered_hours.append(dt.hour)
            
            if not covered_hours:
                # Skip this date if no data is available
                continue
            
            # Sort hours
            covered_hours = sorted(covered_hours)
            
            # Find contiguous sequences
            sequences = []
            start = covered_hours[0]
            end = covered_hours[0]
            
            for i in range(1, len(covered_hours)):
                if covered_hours[i] == end + 1:
                    end = covered_hours[i]
                else:
                    sequences.append(f"{start}-{end}")
                    start = covered_hours[i]
                    end = covered_hours[i]
            
            # Add the last sequence
            sequences.append(f"{start}-{end}")
            
            # Add sequences for this date
            lc_dates[date_key] = sequences
        
        # Only add to config if at least one date has data
        if lc_dates:
            config[str(found_lc)] = {
                'tipo_consumo': str(tipo_consumo),
                'calibre': str(calibre),
                'dates': lc_dates
            }
        else:
            print(f"  ✗ LC {found_lc} has no data coverage in any target dates")
    
    # Save JSON
    print(f"\n{'='*60}")
    print(f"Saving configuration to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Configuration saved successfully")
    print(f"Total LCs selected: {len(config)}")


if __name__ == "__main__":
    input_file = "./Data/telem_widePedro.xlsx"  
    output_json = "./info/test_config.json"
    
    config = generate_test_config(
        input_file=input_file,
        output_json=output_json,
    )
    
    print("\n✓ Done!")