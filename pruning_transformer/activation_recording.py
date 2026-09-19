"""Records which FFN neurons fire across a batch of real inputs.

Pure data collection -- deciding what to do with the recorded firing
pattern (e.g. finding redundant "twin" neurons) is a separate concern,
kept in `redundancy.py`. The original `CoActivationScanner` mixed both
responsibilities in one class; splitting them means the recording
mechanics don't change if the analysis criterion does, and vice versa.

Compare `calibration.FFNCalibrator`, which streams the same activations
but keeps only running moments. This class deliberately retains the full
per-token pattern, because `JaccardTwinFinder` compares *which tokens*
two neurons fired on -- information no summary statistic preserves. The
pattern is stored as `bool` for that reason: it is the one representation
that keeps every bit that matters at a quarter of float32's footprint.
"""
import torch

from .model_adapter import BertLayerAdapter


class ActivationRecorder:
    """Not reentrant -- don't call `record()` concurrently on one
    instance. Sequential reuse (same instance, different `layer_idx`) is
    fine.
    """

    def __init__(self, model, tokenizer, adapter=None, batch_size: int = 32, max_length: int = None):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter or BertLayerAdapter(model)
        self.batch_size = batch_size
        self.max_length = max_length

    def record(self, texts, layer_idx: int, batch_size: int = None, threshold: float = 0.0) -> torch.Tensor:
        """Return a `(valid_tokens, neurons)` bool tensor: did neuron `j`
        fire on token `i`?

        Texts are tokenized and run in batches rather than one at a time;
        at batch size 1 this was dominated by per-call Python and kernel
        launch overhead rather than by any actual work.
        """
        batch_size = batch_size or self.batch_size
        texts = list(texts)
        batches = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            batches.append(self.tokenizer(
                chunk, return_tensors="pt", truncation=True, padding=True,
                max_length=self.max_length,
            ))
        return self.record_batches(batches, layer_idx, threshold=threshold)

    def record_batches(self, batches, layer_idx: int, threshold: float = 0.0) -> torch.Tensor:
        """Same, for batches that are already tokenized (e.g. from a
        HuggingFace collator), so a caller can reuse the exact batching
        their evaluation uses.
        """
        module = self.adapter.get_activation_module(layer_idx)
        device = _model_device(self.model)
        recorded = []
        state = {"mask": None}

        def hook(_module, _inputs, output):
            fired = (output.detach() > threshold).reshape(-1, output.shape[-1])
            mask = state["mask"]
            if mask is not None:
                # Padding positions are an artifact of batching, not data.
                # Keeping them would make every neuron look co-active with
                # every other on whatever the model emits for [PAD] --
                # enough to manufacture twin pairs that don't exist.
                fired = fired[mask.reshape(-1).to(torch.bool)]
            recorded.append(fired.to(torch.bool).cpu())

        was_training = self.model.training
        self.model.eval()
        handle = module.register_forward_hook(hook)
        try:
            for batch in batches:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                batch.pop("labels", None)
                state["mask"] = batch.get("attention_mask")
                with torch.no_grad():
                    self.model(**batch)
        finally:
            handle.remove()
            if was_training:
                self.model.train()

        if not recorded:
            raise ValueError("Nothing was recorded -- `texts`/`batches` was empty.")
        return torch.cat(recorded, dim=0)


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device
