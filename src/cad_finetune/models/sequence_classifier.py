from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from cad_finetune.heads.binary_linear import BinaryLinearHead


class BackboneForSequenceClassification(nn.Module):
    def __init__(
        self,
        backbone,
        num_labels: int = 2,
        dropout: float = 0.0,
        pooling: str = "last_token",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.num_labels = num_labels
        self.pooling = pooling

        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from the backbone config.")

        self.classifier = BinaryLinearHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None):
        if self.pooling == "mean" and attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            masked_hidden = hidden_states * expanded_mask
            denom = expanded_mask.sum(dim=1).clamp(min=1)
            return masked_hidden.sum(dim=1) / denom

        if attention_mask is not None:
            last_token_positions = attention_mask.sum(dim=1).clamp(min=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, last_token_positions]

        return hidden_states[:, -1]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        pooled_output = self._pool_hidden_states(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
