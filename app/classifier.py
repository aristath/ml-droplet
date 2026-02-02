from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

MODELS = {
    "XtremeDistil": {
        "id": "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
        "params": "12.8M",
        "size": "25 MB",
        "weight": 10,
    },
    "DeBERTa-v3-xsmall": {
        "id": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        "params": "22M",
        "size": "142 MB",
        "weight": 20,
    },
    "ModernBERT-base": {
        "id": "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
        "params": "149M",
        "size": "596 MB",
        "weight": 30,
    },
    "BGE-M3": {
        "id": "MoritzLaurer/bge-m3-zeroshot-v2.0",
        "params": "568M",
        "size": "1.14 GB",
        "weight": 40,
    },
}

_classifiers: dict = {}


def _load(name: str):
    model_id = MODELS[name]["id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForSequenceClassification.from_pretrained(
        model_id, subfolder="onnx"
    )
    _classifiers[name] = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
    )


def load_all():
    for name in MODELS:
        if name not in _classifiers:
            _load(name)


def classify(content: str, assertions: list[str], model: str) -> dict[str, float]:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    load_all()
    result = _classifiers[model](content, assertions, multi_label=True)
    return dict(zip(result["labels"], result["scores"]))
