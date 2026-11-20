"""Generate and validate an offline, category-agnostic attribute bank."""

import argparse
import json
from pathlib import Path
import re

FIELDS = ("attribute", "normal", "defect")
AXES = ("continuity", "texture", "edge integrity", "cleanliness", "surface profile", "finish")


def validate_bank(bank):
    rows = bank.get("pairs") if isinstance(bank, dict) else None
    if not isinstance(rows, list) or not 2 <= len(rows) <= 32:
        raise ValueError("Attribute bank must contain between 2 and 32 pairs.")
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(FIELDS):
            raise ValueError("Each pair must have attribute, normal and defect fields.")
        if any(not isinstance(value, str) or not value.strip() or len(value) > 180
               for value in row.values()):
            raise ValueError("Attribute fields must be nonempty short strings.")
        if row["normal"].strip().casefold() == row["defect"].strip().casefold():
            raise ValueError("Normal and defect descriptions must differ.")
    if len({row["attribute"].strip().casefold() for row in rows}) != len(rows):
        raise ValueError("Attribute names must be unique.")
    return rows


def parse_pair(reply):
    match = re.search(r"\{[^{}]*\}", reply, re.S)
    if match is None:
        raise ValueError("No JSON object in LLM response.")
    row = json.loads(match.group())
    if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in FIELDS):
        raise ValueError("LLM JSON is missing a required string field.")
    return {key: row[key].strip() for key in FIELDS}


def generate_bank(model_id, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).eval().to(device)
    rows = []
    for axis in AXES:
        request = (
            f"Write two short English phrases for matching a photo patch to a visual condition "
            f"in industrial inspection. Visual attribute: {axis}. Keep the same subject in "
            "both phrases and change exactly one visible property. Use generic surface or "
            "edge wording, no product names. At most 10 words per phrase. Return only JSON "
            'with keys "attribute", "normal", "defect".'
        )
        for attempt, prompt in enumerate((request, request + (
            ' Format example: {"attribute":"shape","normal":"a straight edge",'
            '"defect":"a bent edge"}. Use the requested attribute instead.'
        )), 1):
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(chat, return_tensors="pt").to(device)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=140, do_sample=False)
            reply = tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            try:
                row = parse_pair(reply)
                row["attribute"] = axis
                validate_bank({"pairs": [row, {"attribute": "__validation__", "normal": "a", "defect": "b"}]})
            except ValueError as exc:
                if attempt == 2:
                    raise RuntimeError(f"Could not generate attribute {axis!r}: {reply!r}") from exc
            else:
                rows.append(row)
                break
    bank = {"generator": model_id, "pairs": rows}
    validate_bank(bank)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = generate_bank(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved {len(bank['pairs'])} attribute pairs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
