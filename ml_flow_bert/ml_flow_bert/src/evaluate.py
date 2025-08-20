# Evaluation logic
import tensorflow as tf

def evaluate_model(model, val_enc, val_labels):
    loss, acc = model.evaluate(
        {
            'input_ids': val_enc['input_ids'],
            'token_type_ids': val_enc['token_type_ids'],
            'attention_mask': val_enc['attention_mask']
        },
        tf.convert_to_tensor(val_labels),
        verbose=1
    )
    return {"val_loss": loss, "val_accuracy": acc}

