"""Scripts for evaluation of models."""
import datasets as hfds
import tensorflow as tf


def load_metric_for_glue_task(task: str):
    return hfds.load_metric("glue", task)


def evaluate_model(model, dataset: tf.data.Dataset, metric: hfds.Metric):
    for model_input, gold_references in dataset:
        model_predictions = model(model_input).logits
        model_predictions = tf.argmax(model_predictions, axis=-1)
        metric.add_batch(predictions=model_predictions, references=gold_references)
    return metric.compute()


def average_score(score):
    return sum(score.values()) / len(score.values())
