# bach
ML based music generator

## Test Jupyter Notebook

[Notebook](./test.ipynb)

## Setup

```shell
pip install uv
uv pip install -r pyproject.toml --all-extras
```

## Data

```shell
# load music data
python -m bach data load
# all midi files are in ./data_cache/data

# create rnn dataset and train
python -m bach data generate-rnn-dataset
python -m bach rnn train

# create transformer dataset and train
python -m bach data generate-transformer-dataset
python -m bach transformer train

# upload model artifact
python -m bach artifact upload-transformer-model ${path-to-model-artifact}
python -m bach artifact upload-rnn-model ${path-to-model-artifact}

python -m bach infer generate-music-by-transformer ${path-to-input-midi-file} ${path-to-output-midi-file}
# output
# Downloading model from Hugging Face Hub (https://huggingface.co/claudezss/bach) to data_cache\model
# Running inference on cuda
# Generated MIDI file was saved at: input.mid
# Generated MIDI file was saved at: temperature-0.1.mid
# Generated MIDI file was saved at: temperature-0.5.mid
# Generated MIDI file was saved at: temperature-1.0.mid
```