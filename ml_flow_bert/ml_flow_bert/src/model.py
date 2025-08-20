# Define Keras BERT model here
from transformers import TFBertForSequenceClassification
import tensorflow as tf

def build_model(bert_model_name, learning_rate):
    model = TFBertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2, from_pt=True)

    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model
