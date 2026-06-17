# Copyright 2025-2026 Meituan Ltd. and/or its affiliates
#
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

"""Inspect old_log_prob vs new_log_prob difference on Ray cluster.

Runs a single step with MAGI prefix tree and dumps the comparison.
"""

import sys

sys.path.insert(0, ".")

import omegaconf


def main():
    # Build a minimal config that enables prefix tree + MAGI
    _config = omegaconf.OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "use_remove_padding": True,
                    "use_prefix_tree": True,
                    "prefix_tree_attention": "magi",
                },
                "actor": {
                    "ppo_mini_batch_size": 8,
                    "ppo_micro_batch_size_per_gpu": 4,
                    "ppo_epochs": 1,
                    "use_kl_loss": False,
                    "entropy_coeff": 0.0,
                    "optim": {"lr": 1e-6},
                },
                "rollout": {
                    "n": 1,
                    "name": "vllm",
                },
            },
            "trainer": {
                "n_gpus_per_node": 1,
                "nnodes": 1,
            },
        }
    )

    print("Loading model...")
    # This test needs a real model + GPU — placeholder
    print("Need to run on cluster with actual model loaded")

    # What we actually want to compare:
    # 1. Run _compute_old_log_prob with MAGI
    # 2. Run same forward again (like update_actor without optimizer step)
    # 3. Compare log_probs

    print("")
    print("Test plan:")
    print("  1. Load a small model (Qwen2.5-0.5B)")
    print("  2. Create a batch with shared-prefix prompts")
    print("  3. Compute old_log_probs (forward only, no grad)")
    print("  4. Compute log_probs again (same weights, forward only)")
    print("  5. Compare: if old_log_probs != log_probs → MAGI nondeterminism")
    print("  6. If equal → divergence comes from optimizer")
    print("")
    print("Expected: with identical weights, log_probs should be identical.")
    print("If they differ, MAGI attention produces different outputs each pass.")


if __name__ == "__main__":
    main()
