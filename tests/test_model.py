import json

import pytest

from src import model
from src.data import DATASETS, DEFAULT_ROOT, load_class_names

HF_CACHE = DEFAULT_ROOT / "hf_cache"
PINNED_REVISION = "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"


@pytest.fixture(scope="module")
def release():
    """Release code and tokenizer from the local cache (no weights). Skips if not downloaded."""
    config = model.load_config("flowers102")
    try:
        path = model.release_path(config, cache_dir=HF_CACHE, weights=False, local_files_only=True)
    except Exception:
        pytest.skip("release code not in data/hf_cache")
    from transformers import AutoTokenizer

    return model.import_release(path), AutoTokenizer.from_pretrained(path)


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_configs(dataset_name):
    config = model.load_config(dataset_name)
    assert config["dataset"] == dataset_name
    assert config["model"]["revision"] == PINNED_REVISION
    assert list(config["prompts"]) == ["p0", "p1", "p2", "p3"]
    for prompt_id in config["prompts"]:
        prompt = model.get_prompt(config, prompt_id)
        assert prompt.instructions
        if prompt.option_description is not None:
            assert "{name}" in prompt.option_description


def test_get_prompt_requires_primary_or_id():
    config = model.load_config("flowers102")
    with pytest.raises(ValueError, match="no primary_prompt"):
        model.get_prompt({**config, "primary_prompt": None})
    with pytest.raises(ValueError, match="unknown prompt"):
        model.get_prompt(config, "p99")
    assert model.get_prompt({**config, "primary_prompt": "p2"}).prompt_id == "p2"


def test_build_record():
    prompt = model.Prompt("p", "state", "which?", "a {name}.")
    record = model.build_record("IMAGE", ["b", "a"], prompt, record_id="x")
    question = record["questions"][model.QUESTION_ID]
    assert record["images"] == ["IMAGE"] and record["state"] == "state" and record["id"] == "x"
    assert question == {"type": "choice", "instructions": "which?", "criteria": {"b": "a b.", "a": "a a."}}
    assert "images" not in model.build_record(None, ["a"], model.Prompt("p", "", "q"))


def test_class_order_maps_sorted_options_back_to_class_indices():
    class_names = ["tulip", "rose", "daisy"]
    option_ids = sorted(class_names)  # what the release code returns: daisy, rose, tulip
    order = model.class_order(option_ids, class_names)
    assert [option_ids[position] for position in order] == class_names
    with pytest.raises(ValueError):
        model.class_order(["daisy", "rose"], class_names)
    with pytest.raises(ValueError):
        model.class_order(["daisy", "rose", "rose"], class_names)


@pytest.mark.parametrize("dataset_name", DATASETS)
@pytest.mark.parametrize("prompt_id", ["p0", "p1", "p2", "p3"])
def test_encoded_options_round_trip(release, dataset_name, prompt_id):
    """Every class name reaches the model as an option and maps back to its own class index."""
    module, tokenizer = release
    class_names = load_class_names(dataset_name)
    prompt = model.get_prompt(model.load_config(dataset_name), prompt_id)
    encoded = module.encode_record(tokenizer, model.build_record(None, class_names, prompt))

    question = encoded.questions[0]
    order = model.class_order(question.option_ids, class_names)
    assert [question.option_ids[position] for position in order] == class_names
    assert len(encoded.input_ids) < 16384

    for class_index in (0, len(class_names) // 2, len(class_names) - 1):
        start, end = question.option_spans[order[class_index]]
        option = json.loads(tokenizer.decode(encoded.input_ids[start:end]))
        assert option["option_id"] == class_names[class_index]
        if prompt.option_description is not None:
            assert option["description"] == prompt.option_description.format(name=class_names[class_index])
