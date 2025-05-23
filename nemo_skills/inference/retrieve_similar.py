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

# partially adapted from https://github.com/lm-sys/llm-decontaminator/tree/main

import json
import logging
import os
import sys
from collections import namedtuple
from typing import Any

import hydra
import torch
from sentence_transformers import SentenceTransformer, util

from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


def top_k_similarity(from_emb, to_emb, top_k, chunk_size):
    # Process the cosine similarity computation in constant memory
    TopKResult = namedtuple('TopKResult', ['values', 'indices'])
    all_values = []
    all_indices = []
    n = to_emb.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Compute cosine similarities for the current chunk
        sim_chunk = util.cos_sim(to_emb[start:end], from_emb)  # shape: (chunk_size, M)
        topk_chunk = torch.topk(sim_chunk, k=top_k, dim=1)
        all_values.append(topk_chunk.values)
        all_indices.append(topk_chunk.indices)
    values = torch.cat(all_values, dim=0)
    indices = torch.cat(all_indices, dim=0)
    return TopKResult(values, indices)


def encode(model, data, batch_size):
    return model.encode(data, batch_size=batch_size, show_progress_bar=True)


def read_data(file_paths, retrieve_key) -> list:
    all_data = set()
    for file_path in unroll_files(file_paths):
        with open(file_path, 'rt', encoding='utf-8') as file:
            all_data.update(set([json.loads(line)[retrieve_key] for line in file]))
    return list(all_data)


@nested_dataclass
class RetrieveSimilarConfig:
    # will find top_k most similar examples in retrieve_from files for each example in compare_to files
    retrieve_from: Any
    compare_to: Any

    # where to save the final file with most similar examples
    # will have the same number of rows as the number of unique "retrieve_key" instances in compare_to files
    output_file: str

    # the model used to compute embedding, default is sentence transformer
    model: str = 'multi-qa-MiniLM-L6-cos-v1'

    # how many most-similar examples to retrieve
    top_k: int = 3
    retrieve_key: str = 'problem'

    # batch size for computing embeddings - increasing will make it faster but use more memory
    batch_size: int = 2048

    # chunk size for computing pairwise similarity - increasing will make it faster but use more memory
    chunk_size: int = 10000

    def __post_init__(self):
        if isinstance(self.retrieve_from, str):
            if ',' in self.retrieve_from:
                self.retrieve_from = self.retrieve_from.split(",")
            else:
                self.retrieve_from = self.retrieve_from.split(" ")

        if isinstance(self.compare_to, str):
            if ',' in self.compare_to:
                self.compare_to = self.compare_to.split(",")
            else:
                self.compare_to = self.compare_to.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_retrieve_similar_config", node=RetrieveSimilarConfig)


@hydra.main(version_base=None, config_name="base_retrieve_similar_config")
def retrieve_similar(cfg: RetrieveSimilarConfig):
    cfg = RetrieveSimilarConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)

    model = SentenceTransformer(cfg.model)

    retrieve_from_list = read_data(cfg.retrieve_from, cfg.retrieve_key)
    compare_to_list = read_data(cfg.compare_to, cfg.retrieve_key)

    assert all(retrieve_from_list), "retrieve_from_list contains none values"
    assert all(compare_to_list), "compare_to_list contains none values"

    retrieve_from_embeddings = encode(model, retrieve_from_list, batch_size=cfg.batch_size)
    compare_to_embeddings = encode(model, compare_to_list, batch_size=cfg.batch_size)
    top_k_indices = top_k_similarity(retrieve_from_embeddings, compare_to_embeddings, cfg.top_k, cfg.chunk_size)
    top_k_similar_items = []
    for i, compare_item in enumerate(compare_to_list):
        similar_items = [retrieve_from_list[index] for index in top_k_indices.indices[i]]
        similarity_scores = top_k_indices.values[i].tolist()
        top_k_similar_items.append(
            {
                cfg.retrieve_key: compare_item,
                'similar_items': similar_items,
                'similarity_scores': similarity_scores,
            }
        )

    with open(cfg.output_file, 'w', encoding='utf-8') as fout:
        for entry in top_k_similar_items:
            fout.write(json.dumps(entry) + '\n')


HELP_MESSAGE = get_help_message(RetrieveSimilarConfig)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        retrieve_similar()
