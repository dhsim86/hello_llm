from transformers import AutoModelForSequenceClassification

if __name__ == '__main__':
    model_id = 'klue/roberta-base'
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)

    print(f"model: {classification_model}")
