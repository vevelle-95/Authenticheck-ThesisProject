"""RoBERTa token features followed by a mask-aware bidirectional GRU."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoConfig, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, name="roberta-base", dimension=256, frozen=False,
                 cache_dir=None, pretrained=True, backbone_config=None, backbone=None):
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        elif pretrained:
            self.backbone = AutoModel.from_pretrained(name, cache_dir=cache_dir)
        else:
            values = dict(backbone_config)
            model_type = values.pop("model_type")
            self.backbone = AutoModel.from_config(AutoConfig.for_model(model_type, **values))
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        self.gru = nn.GRU(self.backbone.config.hidden_size, dimension // 2,
                          bidirectional=True, batch_first=True)
        self.output_dim = dimension

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            sequence = self.backbone(input_ids=input_ids,
                                     attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(1).clamp_min(1).cpu()
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=input_ids.shape[1])
        return encoded
