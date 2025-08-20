# Entrypoint for MLflow training
import yaml
import mlflow
from transformers import BertTokenizer

from src.data_loader import load_and_preprocess_data, split_and_tokenize
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
#, predict_labels
import pandas as pd

if __name__ == "__main__":
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment("bert-sentiment")

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])

    with mlflow.start_run():
        train_df= load_and_preprocess_data("data/train.csv")
        train_enc, val_enc, train_labels, val_labels = split_and_tokenize(train_df, tokenizer, config["max_length"])
        print('split and tokenization done')
        #test_enc = tokenizer(test_df["clean_tweet"].tolist(), truncation=True, padding=True,
                             #max_length=config["max_length"], return_tensors='tf')

        model = build_model(config["bert_model"], config["learning_rate"])
        print('....model sucessfull.....')
        mlflow.log_params(config)
        print('...model training started........')
        history = train_model(model, train_enc, train_labels, val_enc, val_labels,
                              config["batch_size"], config["epochs"])
        print('...training sucessfull....')                      
        metrics = evaluate_model(model, val_enc, val_labels)

        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        mlflow.keras.log_model(model, "model")

        # Predict
        #predictions = predict_labels(model, test_enc)
        #submission = pd.read_csv("data/sample_submission.csv")
        #submission["label"] = predictions
        #submission.to_csv("models/sam_sub-BERT.csv", index=False)
