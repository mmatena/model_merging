"""Code for loading data, focusing on GLUE."""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers.data.processors import glue as hf_glue


_glue_processors = hf_glue.glue_processors
_glue_output_modes = hf_glue.glue_output_modes


# Despite what the paper says, STS-B starts at 0, not 1.
_STSB_MIN = 0
_STSB_MAX = 5
# Corresponds to rounding to nearest 0.2 increment.
_STSB_NUM_BINS = 5 * (_STSB_MAX - _STSB_MIN)


def _to_tfds_task_name(task, split):
    if task == "sts-b":
        task = "stsb"
    elif task == "sst-2":
        task = "sst2"
    elif task == "mnli" and split != "train":
        task = "mnli_matched"
    elif task == "mnli-mm" and split != "train":
        task = "mnli_mismatched"
    return task


def _convert_dataset_to_features(
    dataset,
    tokenizer,
    max_length,
    task,
):
    """Note that this is only for single examples; won't work with batched inputs."""
    pad_token = tokenizer.pad_token_id
    # NOTE: Not sure if this is correct, but it matches up for BERT. RoBERTa does
    # not appear to use token types.
    pad_token_segment_id = tokenizer.pad_token_type_id

    processor = _glue_processors[task]()
    output_mode = _glue_output_modes[task]

    if task == "sts-b":
        # STS-B regression.
        stsb_bins = np.linspace(_STSB_MIN, _STSB_MAX, num=_STSB_NUM_BINS + 1)
        stsb_bins = stsb_bins[1:-1]
    else:
        label_list = processor.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}

    def py_map_fn(keys, *values):
        example = {tf.compat.as_str(k.numpy()): v for k, v in zip(keys, values)}
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        input_ids = tf.constant(input_ids, dtype=tf.int32)
        token_type_ids = tf.constant(token_type_ids, dtype=tf.int32)

        if output_mode == "classification":
            label = label_map[example.label]
            label = tf.constant(label, dtype=tf.int64)
        else:
            label = float(example.label)
            assert 0.0 <= label <= 5.0, f"Out of range STS-B label {label}."
            label = np.digitize(label, stsb_bins)
            label = tf.constant(label, dtype=tf.int64)
        return input_ids, token_type_ids, label

    def map_fn(example):
        input_ids, token_type_ids, label = tf.py_function(
            func=py_map_fn,
            inp=[list(example.keys()), *example.values()],
            Tout=[tf.int32, tf.int32, tf.int64],
        )
        return input_ids, token_type_ids, label

    def pad_fn(input_ids, token_type_ids, label):
        # Zero-pad up to the sequence length.
        padding_length = max_length - tf.shape(input_ids)[-1]

        input_ids = tf.concat(
            [input_ids, pad_token * tf.ones(padding_length, dtype=tf.int32)], axis=-1
        )
        token_type_ids = tf.concat(
            [
                token_type_ids,
                pad_token_segment_id * tf.ones(padding_length, dtype=tf.int32),
            ],
            axis=-1,
        )

        tf_example = {
            # Ensure the shape is known as this is often needed for downstream steps.
            "input_ids": tf.reshape(input_ids, [max_length]),
            "token_type_ids": tf.reshape(token_type_ids, [max_length]),
        }
        return tf_example, label

    dataset = dataset.map(map_fn)
    dataset = dataset.map(pad_fn)
    return dataset


def load_glue_dataset(task: str, split: str, tokenizer, max_length: int):
    tfds_task = _to_tfds_task_name(task, split)
    ds = tfds.load(f"glue/{tfds_task}", split=split)
    ds = _convert_dataset_to_features(
        ds,
        tokenizer,
        max_length,
        task,
    )
    return ds
