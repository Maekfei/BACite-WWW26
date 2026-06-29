import json
import os
from pathlib import Path

def load_metrics_from_json(file_path):
    """Load metric data from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def calculate_overall_metrics(dataset_path, dataset_name):
    """Compute the overall metrics for a dataset"""
    print(f"\n{dataset_name} dataset results:")
    print("=" * 60)

    # Store the metrics for each year
    year_metrics = {}

    for year in range(1, 6):
        year_file = os.path.join(dataset_path, f"year_{year}.json")
        if os.path.exists(year_file):
            data = load_metrics_from_json(year_file)
            metrics = data.get("test_metrics", {})

            # Extract the needed metrics
            rmsle = metrics.get("RMSLE", 0)
            male = metrics.get("MALE", 0)
            ndcg10 = metrics.get("NDCG@10", 0)
            ndcg20 = metrics.get("NDCG@20", 0)
            
            year_metrics[year] = {
                'RMSLE': rmsle,
                'MALE': male,
                'NDCG@10': ndcg10,
                'NDCG@20': ndcg20
            }
            
            print(f"Year {year}:")
            print(f"  RMSLE: {rmsle:.6f}")
            print(f"  MALE:  {male:.6f}")
            print(f"  NDCG@10: {ndcg10:.6f}")
            print(f"  NDCG@20: {ndcg20:.6f}")
    
    # Compute the averages
    if year_metrics:
        avg_rmsle = sum(metrics['RMSLE'] for metrics in year_metrics.values()) / len(year_metrics)
        avg_male = sum(metrics['MALE'] for metrics in year_metrics.values()) / len(year_metrics)
        avg_ndcg10 = sum(metrics['NDCG@10'] for metrics in year_metrics.values()) / len(year_metrics)
        avg_ndcg20 = sum(metrics['NDCG@20'] for metrics in year_metrics.values()) / len(year_metrics)

        print(f"\n{dataset_name} average (Year 1-5):")
        print(f"  RMSLE: {avg_rmsle:.6f}")
        print(f"  MALE:  {avg_male:.6f}")
        print(f"  NDCG@10: {avg_ndcg10:.6f}")
        print(f"  NDCG@20: {avg_ndcg20:.6f}")
        
        return {
            'RMSLE': avg_rmsle,
            'MALE': avg_male,
            'NDCG@10': avg_ndcg10,
            'NDCG@20': avg_ndcg20
        }
    
    return None

def main():
    # Dataset paths
    qwen_path = "gnn/results/qwen"
    aminer_path = "gnn/results/Aminer_1"

    print("Computing the Year 1-5 Overall metrics for both datasets")
    print("=" * 80)

    # Compute the Qwen dataset
    qwen_metrics = calculate_overall_metrics(qwen_path, "Qwen")

    # Compute the Aminer dataset
    aminer_metrics = calculate_overall_metrics(aminer_path, "Aminer")

    # Comparison summary
    if qwen_metrics and aminer_metrics:
        print(f"\nComparison summary:")
        print("=" * 60)
        print(f"{'Metric':<10} {'Qwen':<15} {'Aminer':<15} {'Diff':<15}")
        print("-" * 60)
        
        for metric in ['RMSLE', 'MALE', 'NDCG@10', 'NDCG@20']:
            qwen_val = qwen_metrics[metric]
            aminer_val = aminer_metrics[metric]
            diff = qwen_val - aminer_val
            print(f"{metric:<10} {qwen_val:<15.6f} {aminer_val:<15.6f} {diff:<15.6f}")

if __name__ == "__main__":
    main()

