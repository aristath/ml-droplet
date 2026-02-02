from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

MODEL_ID = "MoritzLaurer/bge-m3-zeroshot-v2.0"

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_ID, subfolder="onnx"
        )
        _classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
        )
    return _classifier


def classify(content: str, assertions: list[str]) -> dict[str, float]:
    result = get_classifier()(content, assertions, multi_label=True)
    return dict(zip(result["labels"], result["scores"]))
