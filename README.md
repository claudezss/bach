# bach
ML based music generator


## Setup

```shell
pip install uv
uv pip install -r pyproject.toml --all-extras
```

## Data

```shell
# load music data
python -m bach data load

# create rnn dataset and train
python -m bach data generate-rnn-dataset
python -m bach rnn train

# create transformer dataset and train
python -m bach data generate-transformer-dataset
python -m bach transformer train

# upload model artifact
python -m batch artifact upload-transformer-model ${path-to-model-artifact}
python -m batch artifact upload-rnn-model ${path-to-model-artifact}
```