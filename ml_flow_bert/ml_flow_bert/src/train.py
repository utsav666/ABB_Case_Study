# Training logic with MLflow logging
def train_model(model, train_enc, train_labels, val_enc, val_labels, batch_size, epochs):
    import tensorflow as tf

    history = model.fit(
        x={
            'input_ids': train_enc['input_ids'],
            'token_type_ids': train_enc['token_type_ids'],
            'attention_mask': train_enc['attention_mask']
        },
        y=tf.convert_to_tensor(train_labels),
        validation_data=(
            {
                'input_ids': val_enc['input_ids'],
                'token_type_ids': val_enc['token_type_ids'],
                'attention_mask': val_enc['attention_mask']
            },
            tf.convert_to_tensor(val_labels)
        ),
        batch_size=batch_size,
        epochs=epochs
    )
    return history

