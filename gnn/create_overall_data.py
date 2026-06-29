#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

def create_overall_linear_plot_data():
    """Create overall linear-plot-group data - merge data from all experiments"""
    excel_file = Path("gnn/results/parameter_experiment_results.xlsx")

    # Mapping from parameter name to symbol
    param_symbol_mapping = {
        'main_loss_weight': 'λ_main',
        'aux_loss_weight': 'λ_aux',
        'adversarial_loss_weight': 'λ_adv',
        'correlation_loss_weight': 'λ_corr',
        'loss_weight': 'λ_reg'
    }

    # The four loss-weight parameters to be combined
    grouped_params = ['main_loss_weight', 'adversarial_loss_weight',
                     'correlation_loss_weight', 'loss_weight']

    # Load all data
    all_data = []
    with pd.ExcelFile(excel_file) as xls:
        for sheet_name in xls.sheet_names:
            if sheet_name.endswith('_Sensitivity'):
                df = pd.read_excel(xls, sheet_name)
                df['Source_Experiment'] = sheet_name.replace('_Sensitivity', '')
                all_data.append(df)
    
    # Merge all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Merged data shape: {combined_df.shape}")
    print(f"Experiments included: {combined_df['Source_Experiment'].unique()}")

    # Create overall data for each parameter type
    overall_data = []

    for param_type in grouped_params:
        if param_type in combined_df['Parameter_Type'].values:
            param_data = combined_df[combined_df['Parameter_Type'] == param_type].copy()

            # Get all unique parameter values
            unique_values = sorted(param_data['Parameter_Value'].unique())

            print(f"\n--- {param_symbol_mapping.get(param_type, param_type)} ({param_type}) ---")
            print(f"Parameter values: {unique_values}")

            # Compute the average for each parameter value (across all experiments)
            for param_value in unique_values:
                value_data = param_data[param_data['Parameter_Value'] == param_value]

                # Compute averages
                avg_male = value_data['MALE'].mean()
                avg_rmsle = value_data['RMSLE'].mean()
                avg_mae = value_data['MAE'].mean()
                avg_rmse = value_data['RMSE'].mean()
                avg_r2 = value_data['R2'].mean()
                avg_spearman = value_data['Spearman'].mean()
                avg_pearson = value_data['Pearson'].mean()
                avg_ndcg10 = value_data['NDCG@10'].mean()
                avg_ndcg20 = value_data['NDCG@20'].mean()
                
                print(f"  {param_value:>8} -> MALE: {avg_male:.6f}, RMSLE: {avg_rmsle:.6f}")

                # Create an overall data row
                overall_row = {
                    'Parameter_Type': param_type,
                    'Parameter_Value': param_value,
                    'MALE': avg_male,
                    'RMSLE': avg_rmsle,
                    'MAE': avg_mae,
                    'RMSE': avg_rmse,
                    'R2': avg_r2,
                    'Spearman': avg_spearman,
                    'Pearson': avg_pearson,
                    'NDCG@10': avg_ndcg10,
                    'NDCG@20': avg_ndcg20,
                    'Parameter_Symbol': param_symbol_mapping.get(param_type, param_type),
                    'Source': 'Overall_Average'
                }
                overall_data.append(overall_row)
    
    # Create the overall data DataFrame
    overall_df = pd.DataFrame(overall_data)

    # Save the overall data
    output_path = "Data/overall_linear_plot_data.csv"
    overall_df.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print("OVERALL data saved to:", output_path)
    print(f"Total rows: {len(overall_df)}")
    print(f"Parameter types: {overall_df['Parameter_Type'].unique()}")
    print(f"Parameter symbols: {overall_df['Parameter_Symbol'].unique()}")

    # Show a summary of the overall data
    print(f"\n{'='*50}")
    print("OVERALL data summary:")
    print(f"{'='*50}")

    for param_type in grouped_params:
        if param_type in overall_df['Parameter_Type'].values:
            param_data = overall_df[overall_df['Parameter_Type'] == param_type].copy()
            param_data = param_data.sort_values('Parameter_Value')

            print(f"\n--- {param_symbol_mapping.get(param_type, param_type)} ({param_type}) ---")
            print("Parameter value -> MALE, RMSLE")
            for _, row in param_data.iterrows():
                print(f"{row['Parameter_Value']:>8} -> {row['MALE']:>8.6f}, {row['RMSLE']:>8.6f}")
    
    return overall_df

if __name__ == "__main__":
    create_overall_linear_plot_data()


