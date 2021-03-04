# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mapping tasks to data loaders."""

import functools
from lra_benchmarks.image import input_pipeline

TASK_DATA_DICT = {
    'cifar10':
        input_pipeline.get_cifar10_datasets,
    'pathfinder32_easy':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=32,
            split='easy'),
    'pathfinder32_inter':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=32,
            split='intermediate'),
    'pathfinder32_hard':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=32,
            split='hard'),
    'pathfinder64_easy':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=64,
            split='easy'),
    'pathfinder64_inter':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=64,
            split='intermediate'),
    'pathfinder64_hard':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=64,
            split='hard'),
    'pathfinder128_easy':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=128,
            split='easy'),
    'pathfinder128_inter':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=128,
            split='intermediate'),
    'pathfinder128_hard':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=128,
            split='hard'),
    'pathfinder256_easy':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=256,
            split='easy'),
    'pathfinder256_inter':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=256,
            split='intermediate'),
    'pathfinder256_hard':
        functools.partial(
            input_pipeline.get_pathfinder_base_datasets,
            resolution=256,
            split='hard'),
}
