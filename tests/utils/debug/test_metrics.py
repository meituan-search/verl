# Copyright 2025 Individual Contributor: TomQunChaoA
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics


class TestMetrics(unittest.TestCase):
    def test_calculate_debug_metrics(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor(
                    [
                        [-1.5085, -0.1200, -0.6650, -0.4823, -0.1426, -1.5557, -2.8532, -0.3919, -0.4294, -0.4700],
                        [-0.0585, -0.0573, -0.4681, -0.5187, -0.7451, -1.2737, -0.0682, -0.4284, -0.5754, -0.0611],
                    ]
                ),
                "old_log_probs": torch.tensor(
                    [
                        [-1.8636, -0.7863, -0.2136, -0.4376, -2.0257, -0.2579, -1.1547, -0.5203, -0.3802, -0.9872],
                        [-0.3507, -0.5426, -0.2725, -0.4637, -0.3577, -0.3733, -1.7560, -1.9542, -0.4229, -1.3098],
                    ]
                ),
                "loss_mask": torch.tensor([[1, 0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]]),
                "responses": torch.zeros((2, 10)),
            }
        )
        metrics = calculate_debug_metrics(data)
        print(metrics)
        assert metrics["training/rollout_probs_diff_valid"] == 1

    def test_calculate_debug_metrics_with_new_rollout(self):
        """When new_rollout_log_probs is present, Pair 2/3 metrics are emitted."""
        rollout = torch.tensor(
            [
                [-1.5085, -0.1200, -0.6650, -0.4823, -0.1426, -1.5557, -2.8532, -0.3919, -0.4294, -0.4700],
                [-0.0585, -0.0573, -0.4681, -0.5187, -0.7451, -1.2737, -0.0682, -0.4284, -0.5754, -0.0611],
            ]
        )
        old = torch.tensor(
            [
                [-1.8636, -0.7863, -0.2136, -0.4376, -2.0257, -0.2579, -1.1547, -0.5203, -0.3802, -0.9872],
                [-0.3507, -0.5426, -0.2725, -0.4637, -0.3577, -0.3733, -1.7560, -1.9542, -0.4229, -1.3098],
            ]
        )
        # new_rollout close to rollout (small staleness)
        new_rollout = rollout + 0.05 * torch.randn_like(rollout)
        data = DataProto.from_dict(
            {
                "rollout_log_probs": rollout,
                "old_log_probs": old,
                "new_rollout_log_probs": new_rollout,
                "loss_mask": torch.tensor([[1, 0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]]),
                "responses": torch.zeros((2, 10)),
            }
        )
        metrics = calculate_debug_metrics(data)

        # Pair 1: rollout vs old (always emitted)
        assert metrics["training/rollout_probs_diff_valid"] == 1
        assert "training/rollout_probs_diff_max" in metrics
        assert "training/rollout_actor_probs_pearson_corr" in metrics

        # Pair 2: new_rollout vs rollout (staleness)
        assert metrics["training/new_rollout_vs_rollout_probs_diff_valid"] == 1
        assert "training/new_rollout_vs_rollout_probs_diff_max" in metrics
        assert "training/new_rollout_vs_rollout_probs_diff_mean" in metrics
        assert "training/new_rollout_vs_rollout_probs_diff_std" in metrics
        assert "training/new_rollout_rollout_probs_pearson_corr" in metrics

        # Pair 3: old vs new_rollout (mismatch)
        assert metrics["training/old_vs_new_rollout_probs_diff_valid"] == 1
        assert "training/old_vs_new_rollout_probs_diff_max" in metrics
        assert "training/old_vs_new_rollout_probs_diff_mean" in metrics
        assert "training/old_vs_new_rollout_probs_diff_std" in metrics
        assert "training/old_new_rollout_probs_pearson_corr" in metrics

        # Sanity: when new_rollout ≈ rollout, Pair 2 diff should be small
        # relative to Pair 1 (combined) diff.
        assert metrics["training/new_rollout_vs_rollout_probs_diff_mean"] < metrics[
            "training/rollout_probs_diff_mean"
        ]


if __name__ == "__main__":
    unittest.main()
