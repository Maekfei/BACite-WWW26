#!/usr/bin/env python3
"""
Inference with a trained BA-Cite model.

Loads a checkpoint, rebuilds the heterogeneous graph from the data splits, runs the
two-stage HGT, and writes the predicted 5-year citation count for each test paper.

Usage:
    python gnn/predict.py --checkpoint path/to/bacite_aminer.pth --out predictions.json

Notes:
    - BA-Cite is transductive: the full graph (train + eval + test papers) is needed to
      produce embeddings; predictions are returned for the `test` papers.
    - Data is read from $DATA_DIR/maindata/test1/{train,eval,test}_processsed.jsonl
      (see README -> Data preparation). Each paper carries the 8-dim feature
      [R, Q, C, H, Y, A1, A2, A3].
    - Output: predicted 5-year citation count per test paper.
"""

import os
import sys
import json
import argparse

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from counterfactual_config import config
from two_stage_hgt_counterfactual import (
    CounterfactualTwoStageHGT,
    load_separated_data_modified,
)


def load_checkpoint(model, path, device):
    """Accept either a raw state_dict or a {'model_state_dict': ...} checkpoint."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser(description="Run BA-Cite inference with a trained checkpoint.")
    ap.add_argument("--checkpoint", required=True, help="path to the trained .pth checkpoint")
    ap.add_argument("--out", default="predictions.json", help="where to write predictions")
    args = ap.parse_args()

    device = config.get_training_params()["device"]

    # 1. Build the heterogeneous graph from the data splits.
    paths = config.get_data_paths()
    builder = load_separated_data_modified(paths["train"], paths["eval"], paths["test"])
    data = builder.build_hetero_data().to(device)

    # 2. Build the model and load the trained weights.
    model = CounterfactualTwoStageHGT(**config.get_model_params()).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # 3. Forward pass over the whole graph. paper_features = the 8-dim paper node features.
    paper_features = data["paper"].x
    with torch.no_grad():
        output = model(data.x_dict, data.edge_index_dict, paper_features)

    citation = output["citation"]
    if "log_citation" in citation:                 # Log-MSE head (default): stored as log(1+y)
        pred = torch.expm1(citation["log_citation"])
    else:                                          # negative-binomial head: mean count
        pred = citation["mu"]
    pred = pred.clamp(min=0).cpu()

    # 4. Select the test papers (split == 'test') and write predictions.
    splits = data["paper"].split
    has_y = hasattr(data["paper"], "y")
    y = data["paper"].y.cpu() if has_y else None

    results = []
    for i, s in enumerate(splits):
        if s != "test":
            continue
        row = {"index": i, "predicted_citations_5y": round(float(pred[i]), 4)}
        if has_y:
            row["true_citations_5y"] = int(y[i])
        results.append(row)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} test-paper predictions to {args.out}")


if __name__ == "__main__":
    main()
