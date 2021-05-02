# PermGNN

About
-----
Code for PermGNN
This directory contains code necessary for running PermGNN experiments.

For more information, please check our paper:
> Roy, Indradyumna, Abir De, and Soumen Chakrabarti. "Adversarial Permutation Guided Node Representations for Link Prediction." arXiv preprint arXiv:2012.08974 (2020). [\[PDF\]](https://arxiv.org/pdf/2012.08974.pdf)

Requirements
------------

Recent versions of Pytorch, numpy, scipy, sklearn, networkx and matplotlib are required.
You can install all the required packages using  the following command:

	$ pip install -r requirements.txt

Run
---


To train PermGNN on the datasets- Twitter/Google/PB/cora/citeseer

```Bash
python -m minmax.linkpred_main --TASK="PermGNN"  --want_cuda=True --BATCH_SIZE=128 --TEST_FRAC=0.2 --VAL_FRAC=0.1  --DIR_PATH="."  --LEARNING_RATE_FUNC=0.0005 --LEARNING_RATE_PERM=0.0005 --MARGIN=0.001 --OPTIM="Adam" --SCORE="MAP" --DATASET_NAME="Twitter_3"
python -m minmax.linkpred_main --TASK="PermGNN"  --want_cuda=True --BATCH_SIZE=128 --TEST_FRAC=0.2 --VAL_FRAC=0.1  --DIR_PATH="."  --LEARNING_RATE_FUNC=0.001 --LEARNING_RATE_PERM=0.001 --MARGIN=0.01 --OPTIM="Adam" --SCORE="MAP" --DATASET_NAME="Gplus_1"
python -m minmax.linkpred_main --TASK="PermGNN"  --want_cuda=True --BATCH_SIZE=128 --TEST_FRAC=0.2 --VAL_FRAC=0.1  --DIR_PATH="."  --LEARNING_RATE_FUNC=0.001 --LEARNING_RATE_PERM=0.001 --MARGIN=0.01 --OPTIM="Adam" --SCORE="MAP" --DATASET_NAME="PB"
python -m minmax.linkpred_main --TASK="PermGNN"  --want_cuda=True --BATCH_SIZE=128 --TEST_FRAC=0.2 --VAL_FRAC=0.1  --DIR_PATH="."  --LEARNING_RATE_FUNC=0.001 --LEARNING_RATE_PERM=0.001 --MARGIN=0.1 --OPTIM="Adam" --SCORE="MAP" --DATASET_NAME="citeseer"
python -m minmax.linkpred_main --TASK="PermGNN"  --want_cuda=True --BATCH_SIZE=128 --TEST_FRAC=0.2 --VAL_FRAC=0.1  --DIR_PATH="."  --LEARNING_RATE_FUNC=0.001 --LEARNING_RATE_PERM=0.001 --MARGIN=0.01 --OPTIM="Adam" --SCORE="MAP" --DATASET_NAME="cora"
```

To run and compare performnce of our proposed hashing technique with random hyperplane hashing and exhaustive enumeration

```Bash
python -m minmax.hashing_main --DATASET_NAME="citeseer"
python -m minmax.hashing_main --DATASET_NAME="cora"
python -m minmax.hashing_main --DATASET_NAME="Twitter_3"
python -m minmax.hashing_main --DATASET_NAME="Gplus_1"
python -m minmax.hashing_main --DATASET_NAME="PB"
```

To reproduct the plots from paper for training loss variation wrt Ktau and Input/Output sensitivity

```Bash
python -m minmax.performance_variation_wrt_ktau --TEST_FRAC=0.2 --VAL_FRAC=0.1 --DATASET_NAME="cora"
python -m minmax.performance_variation_wrt_ktau --TEST_FRAC=0.2 --VAL_FRAC=0.1 --DATASET_NAME="citeseer"
```

Notes
-----
 - GPU usage is required
 - source code files are all in minmax folder.

Reference
---------

If you find the code useful, please cite our paper:

    @article{roy2020adversarial,
      title={Adversarial Permutation Guided Node Representations for Link Prediction},
      author={Roy, Indradyumna and De, Abir and Chakrabarti, Soumen},
      journal={arXiv preprint arXiv:2012.08974},
      year={2020}
    }

Indradyumna Roy, Indian Institute of Technology - Bombay 
indraroy15@cse.iitb.ac.in


