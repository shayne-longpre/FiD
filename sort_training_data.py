import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F

random.seed(0)


def sort_data_by_entropy(data: List[Dict]) -> List[Dict]:
    entropies = []

    for instance in data:
        scores = [float(context['score']) for context in instance['ctxs']]
        # Convert to probabilities
        probabilities = F.softmax(torch.FloatTensor(scores), dim=-1).tolist()
        # Add a small epsilon to each probability to prevent log errors if prob is 0
        probabilities = [p + 1e-10 for p in probabilities]
        # Don't make negative since we want lower to mean less confident
        entropy = -1*sum([p * math.log(p, 2) for p in probabilities])
        entropies.append(entropy)

    assert len(entropies) == len(data)

    # Sort the data from lowest to highest entropy
    data = [inst for _, inst in sorted(zip(entropies, data), key=lambda t: t[0])]
    return data


def sort_data_randomly(data: List[Dict]) -> List[Dict]:
    random.shuffle(data)
    return data


def sample_training_data(data_file: str, sampling_strategy: str, output_dir: str):
    data = json.load(open(data_file))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if sampling_strategy == "entropy":
        data = sort_data_by_entropy(data)
    elif sampling_strategy == "random":
        data = sort_data_randomly(data)
    else:
        print("Please supply an actual sampling strategy")
        sys.exit()

    # # Create growing subsets of data
    # num_subsets = 10
    # num_instances = len(data)
    # for i in range(num_subsets):
    #     num_subset_instances = math.ceil(num_instances/num_subsets*(i+1))
    #     data_subset = data[:num_subset_instances]

    output_file = os.path.join(output_dir, os.path.basename(data_file))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


def main():
    """Sorts data via some active-learning strategy."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Data file to sort.")
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        required=True,
        choices=["entropy", "random"],
        help="Sampling strategy to use to sample data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    sample_training_data(
        args.data_file, args.sampling_strategy, args.output_dir
    )


if __name__ == "__main__":
    main()
