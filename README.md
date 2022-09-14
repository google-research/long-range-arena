## Long-Range Arena (LRA: pronounced ELRA).

Long-range arena is an effort toward systematic evaluation of efficient
transformer models. The project aims at establishing benchmark tasks/dtasets
using which we can evaluate transformer-based models in a systematic way, by
assessing their generalization power, computational efficiency, memory
foot-print, etc.

Long-range arena also implements different variants of Transformer models in
[JAX](https://github.com/google/jax), using
[Flax](https://github.com/google/flax).

This first initial release includes the benchmarks for the paper "Long Range
Arena: A benchmark for Efficient Transformers.

Currently we have released all the necessary code to get started and run our
benchmarks on vanilla Transformers.

## V2 release

*Update* We have released the xformer models used in our experiments.

<s> We are working on a 2nd update that will release more models and baselines for
  this benchmark suite. Stay tuned. </s>

Please see below for more examples on how to get started.



#### Our experiments

Current leaderboard results of all xformer results on our benchmark results. (as
of 8th November 2020)

Model           | ListOps   | Text      | Retrieval | Image     | Path      | Path-X | Avg
--------------- | --------- | --------- | --------- | --------- | --------- | ------ | ---
Local Att       | 15.82     | 52.98     | 53.39     | 41.46     | 66.63     | FAIL   | 46.06
Linear Trans.   | 16.13     | **65.90** | 53.09     | 42.34     | 75.30     | FAIL   | 50.55
Reformer        | **37.27** | 56.10     | 53.40     | 38.07     | 68.50     | FAIL   | 50.67
Sparse Trans.   | 17.07     | 63.58     | **59.59** | **44.24** | 71.71     | FAIL   | 51.24
Sinkhorn Trans. | 33.67     | 61.20     | 53.83     | 41.23     | 67.45     | FAIL   | 51.29
Linformer       | 35.70     | 53.94     | 52.27     | 38.56     | 76.34     | FAIL   | 51.36
Performer       | 18.01     | 65.40     | 53.82     | 42.77     | **77.05** | FAIL   | 51.41
Synthesizer     | 36.99     | 61.68     | 54.67     | 41.61     | 69.45     | FAIL   | 52.88
Longformer      | 35.63     | 62.85     | 56.89     | 42.22     | 69.71     | FAIL   | 53.46
Transformer     | 36.37     | 64.27     | 57.46     | 42.44     | 71.40     | FAIL   | 54.39
BigBird         | 36.05     | 64.02     | 59.29     | 40.83     | 74.87     | FAIL   | **55.01**

### Public External Entries

We list the entries of other papers and submissions that used our LRA benchmark.


Model           | ListOps   | Text      | Retrieval | Image     | Path      | Path-X | Avg
--------------- | --------- | --------- | --------- | --------- | --------- | ------ | ---
IGLOO           | 39.23	    |  82	      |  75.5	    |  47.0	    | 67.50	    | NA     |  62.25
TLB             | 37.05     |  81.88    |  76.91    |  57.51    | 79.06     | FAIL   |  66.48


IGLOO Submissions (by Vsevolod Sourkov) - https://github.com/redna11/lra-igloo \
TLB ([Temporal Latent Bottleneck](lra_benchmarks/models/transformer_tlb)) - [transformer_tlb](lra_benchmarks/models/transformer_tlb) 

## Citation

If you find out work useful, please cite our paper at:

```
@inproceedings{
tay2021long,
title={Long Range Arena : A Benchmark for Efficient Transformers },
author={Yi Tay and Mostafa Dehghani and Samira Abnar and Yikang Shen and Dara Bahri and Philip Pham and Jinfeng Rao and Liu Yang and Sebastian Ruder and Donald Metzler},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=qVyeW-grC2k}
}
```

**Note: Please also cite the original sources of these datasets! **

## Adding results to the leaderboard.

Please send the link of the paper (arxiv, or published) to the Yi Tay or Mostafa
Dehghani (emails in paper) to include your new results to the leaderboard. Just
like above, we will add results to the external submission part of the leaderboard.
This is so that we do not encourage hill-climbing on the leaderboard but rather
meaningful side by side comparisons. 

## A note on evaluation and comparisons

### Meaningful Comparisons

We intend for your benchmark to act as a tool and suite for inspecting model
behaviour. As such, if you're running a new setup and you have tuned hparams,
do consider running all the other models.

### Apples-to Apples setting

This setting is for folks who want to compare with our published results
*directly*.

The default hyperparameter setup (each benchmark should have a config file now).
You are not allowed to change hyperparameters such as embedding size, hidden
dimensions, number of layers of the new model.

The new model should be within at best 10% larger in terms of parameters
compared to the base Transformer model in the provided config file.

### Free-for-all Setting

You are allowed to run any model size and change any hyperparameter of the
model. However, in the end, you'll not be allowed to report results from *our*
leaderboard because they are no longer comparable. You can choose to rerun
models from our library in a comparable setting.

## Adding benchmarks or models to this suite

If you develop or could benefit from an extensive array of xformer baselines,
please feel free to let us know if you're interested in building new benchmarks.
We welcome contributions for new or older models that are not covered in the
existing suite.

## What if I find a better config for an existing model?

In this paper, we did not prioritize doing hparam sweeps. If you happen to find
an implementation related issue or a better hparam that allows a model to do
better on a certain task, do send a PR (or a new config file) and we will
run the model again internally and report new results for the existing model.

## I have a new Xyzformer, how do we add this to the benchmark.

The official results are *only* for code that have been verified and run in
our codebase. We report all external submissions as *external*. Either submit a PR,
an email showing us how to run your model in our codebase and we will update the
results accordingly. (Note due to bandwidth constraints this process will take
a substantial amount of time). 

# Example Usage

To run a task, run the train.py file in the corresponding task directory.
(please see how to obtain the data for certain tasks if applicable).

```
PYTHONPATH="$(pwd)":"$PYTHON_PATH" python lra_benchmarks/listops/train.py \
      --config=lra_benchmarks/listops/configs/transformer_base.py \
      --model_dir=/tmp/listops \
      --task_name=basic \
      --data_dir=$HOME/lra_data/listops/
```

## Dataset Setup

This section describes the methods to obtain the datasets and run the tasks in
LRA.

To download the datasets, please download it from
`gs://long-range-arena/lra_release`. If permissions fail, you may download the
entire gziped file at
https://storage.googleapis.com/long-range-arena/lra_release.gz.

### ListOps

This task can be found at `/listops`. The datasets used in our experiments can
be found at these google cloud buckets and are in TSV format.

If you would like to go to longer/shorter sequence lengths, we also support
generating your own split, run the following comment:

```
PYTHONPATH="$(pwd)":"$PYTHON_PATH" python lra_benchmarks/data/listops.py -- \
  --output_dir=$HOME/lra_data/listops/
```

### Text Classification

This task can be found at `/text_classification`. No action is required because
this task is already found in tensorflow datasets. The code should run as it is.

### Document Retrieval

Please download the dataset at (http://aan.how/download/). Please download the
train/test/dev splits from our google cloud bucket. Unfortunately, we were not
able to re-distribute this datasets and are only releasing the ids in the format
`label paper1_id paper2_id`. You may download the data from the original source
and extract the textual data.

## Pixel-level Image Classification

This task can be found at `/image`. No action is required because this task is
already found in tensorflow datasets. It should work out of the box.

## Pathfinder

Please see the `./data` directory, where the TFDS builder for the pathfinder
dataset can be found. We generated different datasets for pathfinder task, with
different levels of difficulty using the script provided
[here](https://github.com/drewlinsley/pathfinder). You can find information
about the parameters used for generatinng the data in the TFDS builder code in
`./data/pathfinder`. We are preparing the exact data splits for release at the
moment.

## Disclaimer

This is not an official Google product.
