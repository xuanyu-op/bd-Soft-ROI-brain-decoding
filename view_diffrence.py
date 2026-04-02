import argparse
import json
import pickle
import random
from typing import Any, Dict, List, Optional


def load_reference_json(path: str) -> Dict[str, List[str]]:
    """Load the reference caption JSON and normalize it into {str(image_id): [ref1, ref2, ...]} format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if v is None:
            out[str(k)] = []
        elif isinstance(v, list):
            out[str(k)] = [str(x) for x in v if x is not None]
        else:
            out[str(k)] = [str(v)]
    return out


def load_generated_pkl(path: str) -> Dict[str, Any]:
    """Load the generated caption PKL file and convert all keys to strings."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"generated_pkl must be a dict, got {type(data)}")

    return {str(k): v for k, v in data.items()}


def normalize_candidate(obj: Any) -> str:
    """
    Normalize different candidate storage formats into a single string.

    Supported formats:
    - str
    - list[str]: by default, take the first element
    - dict: prioritize best / caption / pred / prediction / generated
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return str(obj[0]) if len(obj) > 0 else ""
    if isinstance(obj, dict):
        for k in ["best", "caption", "pred", "prediction", "generated"]:
            if k in obj and obj[k] is not None:
                return str(obj[k])
        return ""
    return str(obj)


def parse_ids(ids_str: str) -> Optional[List[str]]:
    """Parse the input image_id list, supporting comma-separated or space-separated formats."""
    if not ids_str:
        return None
    parts = [p.strip() for p in ids_str.replace(",", " ").split() if p.strip()]
    return parts if parts else None


def clean_text(s: str) -> str:
    """Remove tabs and line breaks from the text to preserve the one-line export format."""
    if s is None:
        return ""
    s = str(s).replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s


def main():
    """Main function: load inputs, match image_id, organize text, and export to a txt file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated_pkl",
        type=str,
        required=True,
        help="Path to the PKL file, formatted as {image_id: candidate_caption}, also compatible with list/dict forms.",
    )
    parser.add_argument(
        "--reference_json",
        type=str,
        required=True,
        help='Path to the JSON file, formatted as {"image_id": ["ref1","ref2",...]}',
    )
    parser.add_argument(
        "--out_txt",
        type=str,
        default="ref_cand_pairs.txt",
        help="Path to the output txt file.",
    )
    parser.add_argument(
        "--ref_mode",
        type=str,
        default="first",
        choices=["first", "join", "all"],
        help="Reference export mode: first|join|all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If greater than 0, export only the first limit image_ids.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle image_ids before truncation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="Optional: export only specific image_ids, separated by commas or spaces.",
    )
    args = parser.parse_args()

    # Load generated captions and reference captions
    gen = load_generated_pkl(args.generated_pkl)
    ref = load_reference_json(args.reference_json)

    # Parse whether the user specified a subset of image_ids
    wanted_ids = parse_ids(args.ids)

    # Keep only image_ids that appear in both generated results and reference results
    common_ids = [i for i in gen.keys() if i in ref]

    if wanted_ids is not None:
        wanted_set = set(wanted_ids)
        common_ids = [i for i in common_ids if i in wanted_set]

    if not common_ids:
        raise RuntimeError(
            "No matched image_id between generated_pkl and reference_json. "
            "Check that both use the same image_id keys (string/int)."
        )

    # Shuffle if requested; otherwise sort by numeric/alphabetical order
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(common_ids)
    else:
        def sort_key(x: str):
            return (0, int(x)) if x.isdigit() else (1, x)

        common_ids = sorted(common_ids, key=sort_key)

    # Keep only the first few samples if limit is set
    if args.limit and args.limit > 0:
        common_ids = common_ids[:args.limit]

    # Organize output lines according to ref_mode
    lines = []
    for image_id in common_ids:
        cand = clean_text(normalize_candidate(gen[image_id]))
        refs = [clean_text(x) for x in (ref.get(image_id, []) or [])]

        if args.ref_mode == "first":
            ref_text = refs[0] if refs else ""
            lines.append(f"{image_id}\t{ref_text}\t{cand}")
        elif args.ref_mode == "join":
            ref_text = " || ".join(refs) if refs else ""
            lines.append(f"{image_id}\t{ref_text}\t{cand}")
        else:  # all
            if not refs:
                lines.append(f"{image_id}\t\t{cand}")
            else:
                for r in refs:
                    lines.append(f"{image_id}\t{r}\t{cand}")

    # Write the organized results to the txt file
    with open(args.out_txt, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    # Print export statistics
    print(f"✅ imageids matched: {len(common_ids)}")
    print(f"✅ lines written: {len(lines)}")
    print(f"✅ output: {args.out_txt}")


if __name__ == "__main__":
    main()