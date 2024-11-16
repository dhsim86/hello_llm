import json

from transformers import AutoModelForSequenceClassification

if __name__ == '__main__':
    model_id = 'SamLowe/roberta-base-go_emotions'
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)

    print(f"model: {classification_model}")

    from transformers import pipeline
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    sentences = ["I am not having a great day"]

    model_outputs = classifier(sentences)
    print(json.dumps(model_outputs[0]))