"""Per-neuron activation statistics for one FFN layer, measured on real
inputs.

Weight-only criteria (`scoring.py`) can only see how *strongly* a neuron
is wired; they cannot see how *often* or how *hard* it actually fires.
Two neurons with identical incoming weights contribute very differently
to the layer output if one saturates on real text and the other is
almost always dead. `CalibrationStats` supplies that missing half, and
is what `selectors.ActivationAwareSelector` and the bias-compensation
path in `ffn_surgery.FFNSurgeon` consume.

Statistics are accumulated in a streaming fashion -- one small `(neurons,)`
reduction per batch -- so calibrating on a large corpus never materialises
the full `(tokens, neurons)` activation tensor the way
`activation_recording.ActivationRecorder` deliberately does. The two are
complementary, not redundant: the recorder keeps per-token detail because
`redundancy.JaccardTwinFinder` needs to compare firing *patterns*; the
calibrator throws per-token detail away because nothing downstream of it
needs more than a mean.
"""
from dataclasses import dataclass

import torch

from .model_adapter import BertLayerAdapter


@dataclass(frozen=True)
class CalibrationStats:
    """Per-neuron activation moments over the calibration corpus.

    `mean` is the signed expectation `E[a_i]` -- the quantity bias
    compensation needs, because what pruning removes from the layer
    output is `W_out[:, i] * a_i` and its expectation is
    `W_out[:, i] * E[a_i]`. `mean_abs` and `rms` are unsigned magnitude
    measures for importance scoring, where a neuron that swings hard in
    both directions matters even though its signed mean is near zero.
    """

    mean: torch.Tensor
    mean_abs: torch.Tensor
    rms: torch.Tensor
    tokens: int

    @property
    def num_neurons(self) -> int:
        return self.mean.shape[0]

    def to(self, *args, **kwargs) -> "CalibrationStats":
        """Move/cast the three stat vectors, as `torch.Tensor.to` would."""
        return CalibrationStats(
            mean=self.mean.to(*args, **kwargs),
            mean_abs=self.mean_abs.to(*args, **kwargs),
            rms=self.rms.to(*args, **kwargs),
            tokens=self.tokens,
        )

    def select(self, indices) -> "CalibrationStats":
        """Restrict the stats to a subset of neurons, keeping `tokens`.

        Needed when stats collected before surgery are reused against the
        resized layer afterwards.
        """
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.mean.device)
        return CalibrationStats(
            mean=self.mean[idx],
            mean_abs=self.mean_abs[idx],
            rms=self.rms[idx],
            tokens=self.tokens,
        )


class FFNCalibrator:
    """Runs batches through a model and accumulates `CalibrationStats`
    for one FFN layer.

    Takes an `FFNLayerAdapter` rather than reaching into a model layout
    directly, for the same reason every other consumer in this package
    does (see `model_adapter`): a new architecture is a new adapter, not
    an edit here.
    """

    def __init__(self, model, adapter=None):
        self.model = model
        self.adapter = adapter or BertLayerAdapter(model)

    def collect(self, batches, layer_idx: int, max_batches: int = None) -> CalibrationStats:
        """Accumulate stats over `batches`.

        `batches` is any iterable of kwarg dicts ready to splat into the
        model (e.g. what a HuggingFace `DataCollatorWithPadding` yields).
        If a batch carries an `attention_mask`, padding positions are
        excluded -- counting them would drag every mean toward whatever
        the model happens to emit on `[PAD]`, which is an artifact of
        batching rather than a property of the data.
        """
        module = self.adapter.get_activation_module(layer_idx)
        device = _model_device(self.model)

        # float64 CPU accumulators: the per-batch reduction stays in the
        # model's dtype on-device (fast), only the running total is
        # promoted, so a long corpus can't drift the way a float32 (or,
        # worse, float16) running sum would.
        totals = {"sum": None, "abs_sum": None, "sq_sum": None, "tokens": 0}
        state = {"mask": None}

        def hook(_module, _inputs, output):
            acts = output.reshape(-1, output.shape[-1])
            mask = state["mask"]
            if mask is not None:
                acts = acts[mask.reshape(-1).to(torch.bool)]
            if acts.shape[0] == 0:
                return
            acts = acts.detach().float()
            batch_sum = acts.sum(dim=0).double().cpu()
            batch_abs = acts.abs().sum(dim=0).double().cpu()
            batch_sq = acts.square().sum(dim=0).double().cpu()
            if totals["sum"] is None:
                totals["sum"], totals["abs_sum"], totals["sq_sum"] = batch_sum, batch_abs, batch_sq
            else:
                totals["sum"] += batch_sum
                totals["abs_sum"] += batch_abs
                totals["sq_sum"] += batch_sq
            totals["tokens"] += acts.shape[0]

        was_training = self.model.training
        self.model.eval()
        handle = module.register_forward_hook(hook)
        try:
            for i, batch in enumerate(batches):
                if max_batches is not None and i >= max_batches:
                    break
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                batch.pop("labels", None)
                state["mask"] = batch.get("attention_mask")
                with torch.no_grad():
                    self.model(**batch)
        finally:
            handle.remove()
            state["mask"] = None
            if was_training:
                self.model.train()

        if totals["tokens"] == 0:
            raise ValueError("Calibration saw no tokens -- `batches` was empty or fully masked.")

        n = totals["tokens"]
        return CalibrationStats(
            mean=(totals["sum"] / n).float(),
            mean_abs=(totals["abs_sum"] / n).float(),
            rms=(totals["sq_sum"] / n).sqrt().float(),
            tokens=n,
        )


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device
