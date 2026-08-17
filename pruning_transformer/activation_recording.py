"""Records which FFN neurons fire (ReLU output > 0) across a batch of
real inputs.

Pure data collection -- deciding what to do with the recorded firing
pattern (e.g. finding redundant "twin" neurons) is a separate concern,
kept in `redundancy.py`. The original `CoActivationScanner` mixed both
responsibilities in one class; splitting them means the recording
mechanics don't change if the analysis criterion does, and vice versa.
"""
import torch


class ActivationRecorder:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self._activations = []
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        fired = (output > 0).float()
        self._activations.append(fired.reshape(-1, fired.shape[-1]).detach().cpu())

    def record(self, texts, layer_idx: int) -> torch.Tensor:
        target_module = self.model.encoder.layer[layer_idx].intermediate.dense
        self._hook_handle = target_module.register_forward_hook(self._hook_fn)
        try:
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.model(**inputs)
            return torch.cat(self._activations, dim=0)
        finally:
            self._hook_handle.remove()
            self._hook_handle = None
            self._activations = []
