import numpy as np

def augment_instruction(rlds_batch, use_zh_instruction=False, drop_high_level_prob=0.0, **kwargs):
    dataset_name = rlds_batch["dataset_name"]
    lang = rlds_batch["task"]["language_instruction"].decode().lower()
    # lang=f'{high_level_instruction}@{low_level_instruction}@{low_level_instruction_en}' randomly sampled
    lang_split = lang.split("@")
    if len(lang_split) == 1:
        return lang
    elif len(lang_split) == 2:
        high_level_instruction, low_level_instruction = lang_split
        if np.random.rand() < drop_high_level_prob:
            return f"[Low]: {low_level_instruction.strip()}"
        return f"[High]: {high_level_instruction.strip()}, [Low]: {low_level_instruction.strip()}"
    elif len(lang_split) == 3:
        high_level_instruction, low_level_instruction_zh, low_level_instruction = lang_split
        if use_zh_instruction:
            low_level_instruction = low_level_instruction_zh
        if np.random.rand() < drop_high_level_prob:
            return f"[Low]: {low_level_instruction.strip()}"
        return f"[High]: {high_level_instruction.strip()}, [Low]: {low_level_instruction.strip()}"
    else:
        raise ValueError(f"Invalid language instruction: {lang}")
