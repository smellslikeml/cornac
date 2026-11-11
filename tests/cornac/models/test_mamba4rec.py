# Copyright 2026 The Cornac Authors. All Rights Reserved.
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

"""CPU checks for the Mamba4Rec selective scan and its fit/rank loop."""

import os
import subprocess
import sys

import numpy as np
import pytest

from cornac.eval_methods import NextItemEvaluation
from cornac.models import Mamba4Rec
from cornac.models.seq_utils import session_seq_iter

TINY_TRAIN = [
    ("s1", "a", 0),
    ("s1", "b", 1),
    ("s1", "c", 2),
    ("s2", "b", 0),
    ("s2", "c", 1),
    ("s2", "d", 2),
]
TINY_TEST = [("s1", "d", 3), ("s2", "a", 3)]


def tiny_evaluation(val_data=None):
    return NextItemEvaluation.from_splits(
        train_data=TINY_TRAIN,
        test_data=TINY_TEST,
        val_data=val_data,
        fmt="SIT",
        mode="last",
    )


def tiny_model(**kwargs):
    params = dict(
        embedding_dim=8,
        num_layers=1,
        d_state=4,
        d_conv=2,
        expand=2,
        dropout=0.0,
        max_len=4,
        n_sample=4,
        batch_size=4,
        n_epochs=2,
        device="cpu",
        verbose=False,
        seed=7,
    )
    params.update(kwargs)
    return Mamba4Rec(**params)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def fitted(torch):
    evaluation = tiny_evaluation()
    model = tiny_model().fit(evaluation.train_set)
    yield model, evaluation


def test_optional_dependencies_load_only_when_used():
    script = """
import sys
class BlockBackend:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'tensorflow'):
            raise ImportError('optional backend blocked')
sys.meta_path.insert(0, BlockBackend())
from cornac.models import Mamba4Rec
Mamba4Rec()
assert 'torch' not in sys.modules
assert 'tensorflow' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


def test_unsupported_loss_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        Mamba4Rec(loss="bpr-max")


def test_session_iter_left_pads_and_targets_next_item():
    """The shared iterator feeds Mamba4Rec left-padded prefixes of max_len."""
    evaluation = tiny_evaluation()
    train_set = evaluation.train_set
    batches = list(
        session_seq_iter(
            train_set,
            pad_index=train_set.num_items,
            batch_size=8,
            max_len=4,
            n_sample=0,
            shuffle=False,
        )
    )
    assert len(batches) == 1
    _, hist_iids, out_iids = batches[0]
    items = train_set.uir_tuple[1]
    assert hist_iids.shape == (4, 4)  # two sessions x two prefixes
    pad = train_set.num_items
    # every row is left-padded with the padding index and ends on a real item
    assert all(row[0] == pad for row in hist_iids)
    assert all(row[-1] != pad for row in hist_iids)
    # the target is the item that follows the prefix
    assert sorted(out_iids.tolist()) == sorted(
        [items[1], items[2], items[4], items[5]]
    )


def test_scan_output_shape_and_finiteness(torch):
    from cornac.models.mamba4rec.mamba4rec import MambaBlock

    torch.manual_seed(0)
    block = MambaBlock(d_model=8, d_state=4, d_conv=2, expand=2)
    assert block.backend is None  # torch-only install uses the vendored scan
    out = block(torch.randn(2, 6, 8))
    assert out.shape == (2, 6, 8)
    assert torch.isfinite(out).all()


def test_selective_scan_is_causal(torch):
    """Perturbing position t must leave outputs at positions < t unchanged."""
    from cornac.models.mamba4rec.mamba4rec import MambaBlock

    torch.manual_seed(0)
    block = MambaBlock(d_model=8, d_state=4, d_conv=2, expand=2)
    block.eval()
    x = torch.randn(2, 6, 8)
    t = 4
    perturbed = x.clone()
    perturbed[:, t] += 10.0
    before, after = block(x), block(perturbed)
    np.testing.assert_allclose(
        before[:, :t].detach().cpu().numpy(),
        after[:, :t].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert not np.allclose(
        before[:, t:].detach().cpu().numpy(), after[:, t:].detach().cpu().numpy()
    )


def test_cuda_fast_path_is_opt_in(torch, monkeypatch):
    """Without mamba_ssm the vendored scan is used even on a CUDA device."""
    from cornac.models.mamba4rec.mamba4rec import MambaBlock

    monkeypatch.setattr(
        MambaBlock, "_import_cuda_mamba", staticmethod(lambda: None)
    )
    block = MambaBlock(d_model=8, d_state=4, d_conv=2, expand=2, device="cuda")
    assert block.backend is None
    assert block.reset_ssm_parameters() is None  # no-op on the vendored path
    out = block(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 8)


def test_single_layer_omits_residual(torch):
    """Reference quirk: only stacked Mamba layers add the residual connection."""
    from cornac.models.mamba4rec.mamba4rec import MambaLayer

    single = MambaLayer(8, d_state=4, d_conv=2, dropout=0.0, n_layers=1)
    stacked = MambaLayer(8, d_state=4, d_conv=2, dropout=0.0, n_layers=2)
    assert single.residual is False
    assert stacked.residual is True


def test_fit_loss_is_finite_and_decreasing(torch, monkeypatch):
    from cornac.models.seq_utils import losses as seq_losses

    recorded = []
    original = seq_losses.get_loss_function

    def recording(name):
        fn = original(name)

        def wrapped(*args, **kwargs):
            value = fn(*args, **kwargs)
            recorded.append(float(value.detach().cpu()))
            return value

        return wrapped

    monkeypatch.setattr(seq_losses, "get_loss_function", recording)
    evaluation = tiny_evaluation()
    tiny_model(n_epochs=3).fit(evaluation.train_set)

    assert recorded and np.isfinite(recorded).all()
    first_epoch = np.mean(recorded[: len(recorded) // 3])
    last_epoch = np.mean(recorded[-(len(recorded) // 3) :])
    assert last_epoch < first_epoch


def test_score_is_finite_and_rankable(fitted):
    model, evaluation = fitted
    history = [0, 1]  # items 'a', 'b' of session s1
    scores = model.score(0, history)
    assert scores.shape == (model.total_items,)
    assert np.isfinite(scores).all()
    ranks = np.argsort(-scores)
    assert len(set(ranks.tolist())) == model.total_items


def test_empty_history_returns_uniform_scores(fitted):
    model, _ = fitted
    np.testing.assert_array_equal(model.score(0, []), np.ones(model.total_items))


def test_bpr_loss_variant_fits_and_scores(torch):
    evaluation = tiny_evaluation()
    model = tiny_model(loss="bpr").fit(evaluation.train_set)
    scores = model.score(0, [0, 1])
    assert scores.shape == (model.total_items,)
    assert np.isfinite(scores).all()


def test_stacked_layers_fit_and_score(torch):
    evaluation = tiny_evaluation()
    model = tiny_model(num_layers=2).fit(evaluation.train_set)
    scores = model.score(0, [1, 2])
    assert np.isfinite(scores).all()


def test_best_on_val_selection(torch):
    evaluation = tiny_evaluation(val_data=TINY_TEST)
    model = tiny_model(
        n_epochs=2, model_selection="best", val_eval_every=1, val_metric="recall"
    ).fit(evaluation.train_set, evaluation.val_set)
    assert np.isfinite(model.score(0, [0, 1])).all()
