"""Utilities for HuggingFace."""
from typing import Tuple, Union

import tensorflow as tf
from transformers import TFBertPreTrainedModel
from transformers import TFRobertaPreTrainedModel


def get_body_and_head(
    model: Union[TFBertPreTrainedModel, TFRobertaPreTrainedModel]
) -> Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
    body, *head = model.layers
    if not head:
        head = None
    elif len(head) > 1:
        raise ValueError(
            f"Expected model to have a single 'head' layer. Instead found {len(head)}. TODO: Support this."
        )
    else:
        head = head[0]
    return body, head


def get_body(model):
    return get_body_and_head(model)[0]


def get_mergeable_variables(model):
    return get_body_and_head(model)[0].trainable_variables


def clone_model(model):
    cloned = model.__class__(model.config)
    cloned(model.dummy_inputs)
    cloned.set_weights(model.get_weights())
    return cloned
