import torch
import utils
import numpy as np
import re
import json
import os


def print_header(text):
    """Print a formatted section header for clearer command-line logs."""
    print("\n" + "=" * 80)
    print(f"===== {text.center(68)} =====")
    print("=" * 80)


def seed_everything(seed):
    """Set all random seeds consistently and enable deterministic cudnn behavior."""
    utils.seed_everything(seed, cudnn_deterministic=True)


def preprocess_voxels(voxels, device):
    """
    Normalize voxel input shape and data type.
    """
    if isinstance(voxels, (list, tuple)):
        voxels = voxels[0]

    if not isinstance(voxels, torch.Tensor):
        voxels = torch.from_numpy(voxels)

    voxels = voxels.to(device, dtype=torch.float32)

    # Remove redundant batch dimensions if present
    if voxels.dim() == 3 and voxels.size(0) == 1:
        voxels = voxels.squeeze(0)
    elif voxels.dim() == 4 and voxels.size(0) == 1:
        voxels = voxels.squeeze(0)

    # If the input is [T, V], average over the time dimension;
    # if it is already [V], use it directly
    if voxels.dim() == 2:
        voxels = voxels.mean(dim=0)
    elif voxels.dim() != 1:
        raise RuntimeError(
            f"Unexpected voxel shape: {tuple(voxels.shape)}; expected [T,V] or [V]."
        )

    return voxels.unsqueeze(0)


def get_generation_kwargs(args, tokenizer):
    """
    Build generation arguments in a unified way from command-line settings.
    """
    gen_kwargs = dict(
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
    )

    if args.do_sample:
        gen_kwargs.update(
            dict(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )
    else:
        gen_kwargs.update(
            dict(
                do_sample=False,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                early_stopping=True,
            )
        )

    return gen_kwargs


def clean_caption(response):
    """
    Clean the generated caption text.
    """
    processed_caption = response.lower().strip()

    if '.' in processed_caption:
        first_sentence = processed_caption.split('.')[0]
        processed_caption = first_sentence + '.' if first_sentence else ''

    return processed_caption


def robust_json_parse(json_string: str):
    """
    Parse JSON from a string as robustly as possible.
    """
    match = re.search(r'```json\s*(\[.*?\])\s*```', json_string, re.DOTALL)
    if not match:
        match = re.search(r'(\[.*?\])', json_string, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def check_java_runtime():
    """
    Check whether a Java runtime is available in the current environment. This function is mainly used to determine whether SPICE can be enabled.
    """
    try:
        java_home = os.environ.get('JAVA_HOME')
        if java_home and os.path.exists(os.path.join(java_home, 'bin', 'java')):
            return True

        from shutil import which
        if which('java'):
            return True
    except Exception:
        return False

    return False