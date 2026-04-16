from __future__ import annotations

from transformers import DataCollatorWithPadding


def build_data_collator(tokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)
