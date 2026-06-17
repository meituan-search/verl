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

"""Check if the 2 fixes are present in the codebase."""

import sys

sys.path.insert(0, ".")

print("=" * 60)
print("Check 1: prefix_tree_for_olp fix (engine_workers.py)")
print("=" * 60)

# Fix: prefix_tree_for_olp=None should inherit use_prefix_tree
# Check line 407-409: should handle None correctly

with open("verl/workers/engine_workers.py") as f:
    content = f.read()

# Look for the fix pattern
if "prefix_tree_for_olp" in content:
    # Extract relevant lines around line 407
    lines = content.split("\n")
    for i, _l in enumerate(lines[400:415], start=400):
        if "prefix_tree" in _l.lower() or "use_prefix_tree" in _l.lower():
            print(f"  L{i}: {_l.strip()}")

    # Check if fix is present: should check for None before using
    has_fix = (
        "prefix_tree_for_olp" in content
        and "is not None" in content[content.find("prefix_tree_for_olp") : content.find("prefix_tree_for_olp") + 200]
    )
    if has_fix:
        print("  FIX FOUND: None check present")
    else:
        # Check current code - what does it do?
        olp_start = content.find("prefix_tree_for_olp")
        snippet = content[olp_start : olp_start + 300]
        print(f"  Current code snippet:\n{snippet[:200]}")
        print("  FIX STATUS: NEED to check if None is handled")
else:
    print("  prefix_tree_for_olp not found in engine_workers.py")

print()
print("=" * 60)
print("Check 2: per-sample roll fix (magi.py)")
print("=" * 60)

with open("verl/utils/prefix_tree/magi.py") as f:
    magi_content = f.read()

# Check if the fix is present: look for _restore_and_roll_labels
has_magi_fix = "_restore_and_roll_labels" in magi_content

# Check if OLD flat roll is still there
has_old_roll = "torch.roll(pt_batch.flat_input_ids[:real_tokens]" in magi_content

print(f"  _restore_and_roll_labels present: {has_magi_fix}")
print(f"  OLD flat roll still present: {has_old_roll}")

if has_magi_fix:
    lines = magi_content.split("\n")
    for i, _l in enumerate(lines):
        if "_restore_and_roll_labels" in _l:
            print(f"  L{i}: {_l.strip()}")
    # Show the fix function
    idx = magi_content.find("def _restore_and_roll_labels")
    if idx > 0:
        print(f"\n  Fix function:\n{magi_content[idx : idx + 300]}")

if has_old_roll:
    lines = magi_content.split("\n")
    for i, _l in enumerate(lines):
        if "torch.roll(pt_batch.flat_input_ids[:real_tokens]" in _l:
            print(f"  OLD flat roll at L{i}: {_l.strip()}")

print()
print("SUMMARY:")
print(f"  Fix 1 (prefix_tree_for_olp): {'PRESENT' if has_fix else 'NEEDS VERIFICATION'}")
print(f"  Fix 2 (magi per-sample roll): {'PRESENT' if has_magi_fix else 'MISSING' if not has_magi_fix else 'UNKNOWN'}")
