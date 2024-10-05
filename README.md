# Paper: Tabular data generation with tensor contraction layers and transformers

This repository has all the code needed to run the experiments performed in the paper: "Tabular data generation with tensor contraction
layers and transformers".

A guideline to reproduce the experiments presented in the paper is provided below.

We also release a loosely version of this repository [here](https://github.com/anibalsilva1/TensorConFormer), that allows you to run experiments on your own dataset.

# Create environment and install dependencies via conda:
To train the models in this work, start by creating a conda environment by executing the following command

```
conda env create -f requirements.yml
```
then activate the environment:
```
conda activate tensorconformer 
```

# Jax + Flax

To install jax, you can install it via pip:

```
pip install -U jax
```

If you have CUDA 12 installed, you can also

```
pip install -U "jax[cuda12]"
```

For more details, please follow the official installation guide (https://jax.readthedocs.io/en/latest/installation.html)

To install flax

```
pip install flax
```
# Synthethic Dataset Generation

To produce synthethic datasets evaluated in our experimental setup you can do so by executing

```
python main.py --model=[MODEL_NAME]
```

where `MODEL_NAME` is the name of a model. An example with more command options is provided as follows

```
python main.py \
    --model=TensorConFormer \
    --num_epochs=300 \
    --save_models=False \
    --save_best_models_only=False \
    --lr=1e-3 \
    --warmup=0 \
    --patience=25
```

**Note**: Executing this script will train the model over all datasets from OpenML CC18 suite.

# Evaluation

**Note**: In this repository, we only provide the evaluation results inside the `evaluation_results` directory. The synthetic datasets obtained from each model will be shared upon request.

To obtain evaluation results presented in the paper, you can do so by creating the environment

```
conda env create -f requirements_eval.yml
```

activating the environment

```
conda activate evaluation
```

and then running the following

```
python eval_metrics.py --metric_name=[METRIC]
```

where `METRIC` is the name of a metric in [`highdensity`, `ml_efficiency`, `quality`].

**Note**: This will run a given metric for all the models and all datasets from OpenML CC18.

## Radar plots

To produce radar plots, execute

```
python make_radar_plot.py 
```

## Bayes Sign Test

To produce the barplots for Bayes Sign Test, execute

```
python bayes_sign_test.py --models=[MODELS]
```

where `MODELS` stand for `main` or `ablation`.