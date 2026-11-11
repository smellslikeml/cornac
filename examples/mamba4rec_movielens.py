# Copyright 2026 The Cornac Authors. All Rights Reserved.
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
# ============================================================================
"""Mamba4Rec: sequential recommendation with selective state space models.

Mamba4Rec keeps SASRec's training protocol but swaps the causal
self-attention encoder for a stack of Mamba (selective state space) blocks.
The selective scan is vendored in pure torch, so this example runs on CPU
with torch alone; installing the optional ``mamba-ssm`` package and setting
``device="cuda"`` switches the blocks to the fused CUDA kernel.

Each user's chronologically sorted history is one session. The last item is
held out for testing and the second-to-last for validation
(``NextItemEvaluation.leave_last_out``), the protocol used by SASRec,
BERT4Rec and the Mamba4Rec paper.

Settings here are trimmed for a quick CPU run: a tenth of the MovieLens
100K users, ``max_len`` 50 and 3 epochs. The vendored scan is an explicit
python loop over the sequence, so peak memory grows with
``batch_size * max_len * embedding_dim * expand * d_state`` -- on a GPU the
optional ``mamba-ssm`` kernel removes that cost. The paper's setting is
MovieLens 1M with ``max_len=200`` and 300 epochs.
"""

import torch

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import Mamba4Rec, SPop

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

data = movielens.load_feedback(fmt="UIRT")
# keep a slice of the users so the example finishes in minutes on CPU;
# drop this line to train on the full dataset
data = [tup for tup in data if int(tup[0]) % 10 == 0]
print(f"data loaded ({len(data)} interactions)")

next_item_eval = NextItemEvaluation.leave_last_out(
    data=data,
    exclude_unknowns=True,
    verbose=True,
)

models = [
    SPop(verbose=True),
    Mamba4Rec(
        embedding_dim=64,
        num_layers=1,
        d_state=32,
        d_conv=4,
        expand=2,
        dropout=0.2,
        loss="ce",
        max_len=50,
        n_sample=256,
        batch_size=128,
        n_epochs=3,
        learning_rate=0.001,
        device=DEVICE,
        verbose=True,
        seed=123,
    ),
]

metrics = [
    NDCG(k=10),
    NDCG(k=50),
    Recall(k=10),
    Recall(k=50),
    MRR(),
]

cornac.Experiment(
    eval_method=next_item_eval,
    models=models,
    metrics=metrics,
).run()
