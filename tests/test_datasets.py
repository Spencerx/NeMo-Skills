# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import subprocess
from pathlib import Path

# tuple of dataset name, available splits and prepared sft files
DATASETS = [
    ('aime25', ['test']),
    ('math-500', ['test']),
    ('aime24', ['test']),
    ('amc23', ['test']),
    ('omni-math', ['test']),
    ('algebra222', ['test']),
    ('arena-hard', ['test']),
    ('mt-bench', ['test']),
    ('asdiv', ['test']),
    ('gsm-plus', ['test', 'test_rounded']),
    ('gsm8k', ['train', 'test']),
    ('hle', ['math', 'text']),
    ('human-eval', ['test']),
    (
        'livecodebench',
        [
            'test_v5_2408_2502',
            'test_v5_2410_2502',
            'test_v5_2410_2504',
            'test_v6_2408_2502',
            'test_v6_2410_2502',
            'test_v6_2410_2504',
        ],
    ),
    ('ifeval', ['test']),
    ('math', ['train', 'test']),
    ('math-odyssey', ['test']),
    ('mawps', ['test']),
    ('mbpp', ['test']),
    ('mmlu', ['test', 'dev', 'val']),
    ('svamp', ['test']),
    ('answer-judge', ['test']),
    ('mmlu-pro', ['test']),
    ('mmlu-redux', ['test']),
    ('gpqa', ['diamond', 'main', 'extended']),
    ('minerva_math', ['test']),
    ('olympiadbench', ['test']),
    ('gaokao2023en', ['test']),
    ('college_math', ['test']),
    ('comp-math-24-25', ['test']),
]


def test_dataset_scripts():
    # test dataset groups
    dataset_groups = ["math", "code", "chat", "multichoice"]  # not testing long-context as it requires extra args
    prepared_datasets = set()
    for group in dataset_groups:
        # not using ns interface here as it takes quite a bit longer in the CI
        cmd = f"python -m nemo_skills.dataset.prepare --dataset_groups {group}"
        print(f"Running command (output is captured): {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
        )
        print("Finished")
        assert result.returncode == 0, f"Preparation of {group} dataset group failed"

        group_datasets = set(
            line.split("Preparing ")[1].strip() for line in result.stdout.split('\n') if "Preparing" in line
        )
        prepared_datasets.update(group_datasets)

        # Check if at least one dataset from the group was prepared
        assert len(group_datasets) > 0, f"No datasets were prepared for group {group}"

    all_datasets = set(dataset for dataset, _ in DATASETS)

    assert (
        prepared_datasets == all_datasets
    ), f"Not all datasets were covered. Missing: {all_datasets - prepared_datasets}"

    # checking that all expected files are created
    expected_files = []
    for dataset, splits in DATASETS:
        for split in splits:
            expected_files.append(f"{dataset}/{split}.jsonl")

    for file in expected_files:
        assert (Path(__file__).absolute().parents[1] / "nemo_skills" / "dataset" / file).exists()


def test_dataset_init_defaults():
    for dataset, _ in DATASETS:
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
        assert hasattr(dataset_module, 'PROMPT_CONFIG'), f"{dataset} is missing PROMPT_CONFIG attribute"
        assert hasattr(dataset_module, 'DATASET_GROUP'), f"{dataset} is missing DATASET_GROUP attribute"
        assert dataset_module.DATASET_GROUP in [
            "math",
            "code",
            "chat",
            "multichoice",
            "long-context",
        ], f"{dataset} has invalid DATASET_GROUP"
