# Clef-flash interface notes

Source: `Cloudflare/clef-flash` at revision `fb4417d31c1d0f9e7e77f1141b40aaede7491e00`
(model card `README.md` and `joint_schema_model.py`). Read on 2026-11-02. Weights not downloaded.

## Loading

```python
path = snapshot_download("Cloudflare/clef-flash", revision="fb4417d31c1d0f9e7e77f1141b40aaede7491e00")
sys.path.insert(0, path)
from joint_schema_model import collate_records, encode_record, load_release_model
model, processor = load_release_model(path, device="cuda")   # bf16 by default, returns model in .eval()
```

- Backbone: `Qwen3_5ForConditionalGeneration` (hidden size 4096, 32 layers, vision encoder included).
- Joint schema head: separate `joint_head.safetensors` (~243 MB), config in `joint_head_config.json`
  (width 1024, 2 routing layers, 4 decoder layers, 16 heads).
- Total download is ~19 GB. On ICE, point `HF_HOME` at scratch storage, not home.
- No `trust_remote_code`: the release code is imported from the snapshot directory via `sys.path`.

## Record format

From the model card's input-format table and `encode_record`:

| Field | Required | Notes |
|---|---|---|
| `state` | yes (`KeyError` if missing) | Any string or JSON value. JSON is rendered compact with sorted keys. |
| `images` | no | List of PIL images. Requires `processor=processor` in `encode_record`. |
| `media_kwargs` | no | Extra kwargs passed to the image processor (e.g. resolution limits). |
| `questions` | yes | Mapping of question ID → question. |
| `id` | no | Becomes `EncodedRecord.record_id` (defaults to `"unknown"`). |

Each `choice` question:

```python
{"type": "choice",
 "instructions": "...",                        # optional; question ID used if omitted
 "criteria": {option_id: description, ...}}    # description may be None
```

Classification record shape for this project (exact text is a Phase 3 / `p0` decision):

```python
{"state": <text>, "images": [pil_image],
 "questions": {"label": {"type": "choice", "instructions": <question>,
                         "criteria": {<option_id>: <description>, ...}}}}
```

## How the input is laid out

Token sequence built by `encode_record`:

```
<|im_start|>system
Read the complete state and schema. Decide every field jointly. ...<|im_end|>
<|im_start|>user
STATE:
<image tokens>\n<state text>

SCHEMA FIELDS:

FIELD 1
ID: label
TYPE: choice
INSTRUCTION: <instructions>
ALLOWED OPTIONS:
OPTION 1: {"description":"...","option_id":"..."}
OPTION 2: ...
END FIELD

<|im_end|>
<|im_start|>assistant
<think>

</think>

JOINT SCHEMA DECISIONS:
```

- The system prompt is fixed in the release code; we do not control it.
- Both the option ID and its description are rendered as JSON inside each option span, and the head
  reads the hidden states and token embeddings of that span. **The option ID text is seen by the model**,
  so it is part of the prompt, not just a key.

## Output

- `model(batch)` returns `list[list[Tensor]]`: per record, per question, a 1-D logits tensor with one
  entry per option. Softmax per question gives probabilities.
- **Option order is sorted by option ID** (`question_options` sorts `criteria` keys for `choice`),
  not insertion order. Always map logits to classes through `encoded.questions[i].option_ids`,
  never by position.

## Image tokens

- Processor: `Qwen3VLProcessor` with `Qwen2VLImageProcessor`, patch 16, merge 2, so one visual token per
  32×32 pixel block. Pixel bounds are `shortest_edge` 65,536 px (min area) and `longest_edge`
  16,777,216 px (max area).
- Flowers-102 (~500×600) and Food-101 (max side 512) should be roughly 250–300 visual tokens per image.
  No resizing needed, but we can cap resolution through `media_kwargs` if memory or speed requires.

## Training-relevant details (for later phases)

- The per-question logits tensor plugs directly into `F.cross_entropy` against the true option index
  (index into `option_ids`).
- `ClefModel.forward` already unwraps a `peft` model (`get_base_model`), so LoRA on
  `model.language_model` should work without patching the release code.
- `load_release_model` returns the model in `.eval()`; training must call `.train()`. Head dropout is 0.
- The head loops over records and options in Python, which is slow but differentiable.
- Batches are right-padded; the head uses `attention_mask.sum()` as each record's length.
- `encode_record` defaults to `max_length=16384` and raises if the schema alone exceeds it.
