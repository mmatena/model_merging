# Model Merging

This repo contains a reference implementation for the model merging procedure from [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832) along
with a few scripts that allow some basic experimentation with HuggingFace models and the GLUE
dataset.

## Scripts
The scripts are all contained in the `scripts` folder.

### `compute_fisher.py`

This script computes the diagonal approximation to the Fisher matrix given a model and a GLUE task.
The Fisher matrix is saved to a file for use in merging.

Here are the parameters this script takes:

- `--model` Either the path to a saved HuggingFace model or the name of a pretrained HuggingFace
  model from the repository.
- `--glue_task` The name of the GLUE task to use when computing the Fisher.
- `--split` The split of the dataset to use when computing the Fisher.
- `--fisher_path` The path to the hdf5 where we will save the computed Fisher.
- `--n_examples` The number of examples to use when computing the Fisher.
- `--batch_size` The batch size.
- `--sequence_length` Sequence length to use.

### `merge_and_evaluate.py`

This script performs the merging and print the best result.

Here are the parameters this script takes:

- `--models` Comma-separated list of models to merge. Each model is either the path to a saved HuggingFace model or the name of a pretrained HuggingFace model from the repository.
- `--fishers` Optional comma-separated list of Fishers to use. If this flag is not provided, then the script will do an isometric merge. Otherwise the number of Fishers must match the number of models in the `--models` flag. The i-th Fisher in this list is the Fisher of the i-th model from the `--models` list. Each fisher should be the path to an hdf5 file created by the `compute_fisher.py` script.
- `--glue_task` The name of the GLUE task to evaluate on when merging the models.
- `--split` The split of the dataset to use for evaluating.
- `--n_examples` The number of examples to use when evaluating.
- `--batch_size` The batch size.
- `--sequence_length` Sequence length to use.
- `--n_coeffs` The total number of different merging coefficients to try.
- `--coeff_mode` Either `'grid'` or `'random'`. The grid mode corresponds to choosing coefficients uniformly on a grid. The script only allows it when merging exactly two model. Random corresponds to randomly generating coefficients.
- `--fisher_floor` Minimum value to use for each Fisher entry. Prevents numerical issues when the Fisher for a parameter is close to zero across all the models.
- `--favor_target_model` Whether to default to the first model's parameter value when all Fisher values are below the Fisher floor.
- `--normalize_fishers` Whether to normalize the Fishers so that each of them has an L2 norm of 1.

## Examples

### Isometric merge
This example command isometrically merges two RoBERTa models finetuned on RTE and MNLI and evaluates the merged models on RTE.

```bash
EVAL_TASK=rte
RTE_MODEL=textattack/roberta-base-RTE
MNLI_MODEL=textattack/roberta-base-MNLI

# Isometric merge.
python3 ./scripts/merge_and_evaluate.py  \
    --models=$RTE_MODEL,$MNLI_MODEL \
    --glue_task=$EVAL_TASK
```

### Fisher merge
These example commands first compute the Fishers of two RoBERTa models finetuned on RTE and MNLI.
Then the Fishers are used to Fisher merge the two models.

```bash
EVAL_TASK=rte
RTE_MODEL=textattack/roberta-base-RTE
MNLI_MODEL=textattack/roberta-base-MNLI
FISHER_DIR=/path/to/directory/containing/fishers

# Compute RTE Fisher.
python3 ./scripts/compute_fisher.py  \
    --model=$RTE_MODEL \
    --glue_task="rte" \
    --fisher_path="$FISHER_DIR/rte_fisher.h5"
    
# Compute MNLI Fisher.
python3 ./scripts/compute_fisher.py  \
    --model=$MNLI_MODEL \
    --glue_task="mnli" \
    --fisher_path="$FISHER_DIR/mnli_fisher.h5"

# Fisher merge
python3 ./scripts/merge_and_evaluate.py  \
    --models=$RTE_MODEL,$MNLI_MODEL \
    --fishers=$FISHER_DIR/rte_fisher.h5,$FISHER_DIR/mnli_fisher.h5 \
    --glue_task=$EVAL_TASK
```



