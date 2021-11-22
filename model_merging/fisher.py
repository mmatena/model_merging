"""Code for computing and storing Fishers."""
import tensorflow as tf
from model_merging import hf_util


def _batch_size(batch):
    return tf.shape(batch["input_ids"])[0]


@tf.function
def _compute_exact_fisher_for_batch(batch, model, variables, expectation_wrt_logits):
    assert expectation_wrt_logits, "TODO: Handle sampling from logits."
    num_labels = model.num_labels

    @tf.function
    def fisher_single_example(single_example_batch):
        """
        NOTE: I wrote this with Hugging Face classifiers in mind. There is
        probably a good way to do the same thing but with more customizability
        to support alternate forms of models.
        """
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(variables)

            logits = model(single_example_batch, training=False).logits
            # The batch dimension must be 1 to call the model, so we remove it
            # here.
            logits = tf.squeeze(logits, axis=0)

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)

            sq_grads = []
            for i in range(num_labels):
                log_prob = log_probs[i]
                with tape.stop_recording():
                    grad = tape.gradient(log_prob, variables)
                    sq_grad = [probs[i] * tf.square(g) for g in grad]
                    sq_grads.append(sq_grad)
            # Take the average across logits. The per-logit weight was added
            # earlier as each per-logit square gradient was weighted by the
            # probability of the class according to the output distribution.
            example_fisher = [tf.reduce_sum(g, axis=0) for g in zip(*sq_grads)]

        return example_fisher

    batch = {k: tf.expand_dims(v, axis=1) for k, v in batch.items()}

    fishers = tf.vectorized_map(fisher_single_example, batch)
    return [tf.reduce_sum(f, axis=0) for f in fishers]


def compute_fisher_for_model(
    model, dataset: tf.data.Dataset, expectation_wrt_logits=True
):
    variables = hf_util.get_mergeable_variables(model)

    fishers = [
        tf.Variable(tf.zeros(w.shape), trainable=False, name=f"fisher/{w.name}")
        for w in variables
    ]

    n_examples = 0
    for batch, _ in dataset:
        n_examples += _batch_size(batch)
        batch_fishers = _compute_exact_fisher_for_batch(
            batch, model, variables, expectation_wrt_logits=expectation_wrt_logits
        )
        for f, bf in zip(fishers, batch_fishers):
            f.assign_add(bf)

    for fisher in fishers:
        fisher.assign(fisher / float(n_examples))

    return fishers
