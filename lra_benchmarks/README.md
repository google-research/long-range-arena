# Long Range Arena (LRA)
 
There have been a dizzying number of efficient Transformers proposed recently.
We present a framework and benchmark for systematically evaluating long-range 
Transformer  models. Not only are our tasks inherently difficult, they 
are also capability probing and tests for insight on a particular type 
of data structure, e.g., hierarchical structures, logical reasoning or 
spatial reasoning. 

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

This section describes the methods to obtain the datasets and run the tasks
in LRA. While some of the tasks already come nicely packaged in Tensorflow
datasets. we are still waiting on approval for releasing certain explicit
splits. 

### ListOps

This task can be found at `/listops`. We are still pending approval for
releasing the split used in our paper and will keep this updated.
To generate your own split, run the following comment:
```
PYTHONPATH="$(pwd)":"$PYTHON_PATH" python lra_benchmarks/data/listops.py -- \
  --output_dir=$HOME/lra_data/listops/
```

### Text Classification

This task can be found at `/text_classification`. No action is required
because this task is already found in tensorflow datasets.

### Document Retrieval

Please download the dataset at (http://aan.how/download/). We are still
pending approval for releasing the splits for these datasets and will
keep this updated.

## Pixel-level Image Classification

This task can be found at `/image`. No action is required
because this task is already found in tensorflow datasets.

## Pathfinder

TODO: dehghani


## Citation

@inproceedings{
tay2020long,
title={Long Range Arena : A Benchmark for Efficient Transformers },
author={Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri,
Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler},
booktitle={ArXiv Preprint},
year={2020},
url={},
note={under review}
}





