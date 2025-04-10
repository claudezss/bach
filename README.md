# bach
ML based music generator

## Jupyter Notebook Tutorial

[Notebook](./test.ipynb)
This notebook contains tests and examples for using the Bach music generator. It demonstrates how to load data, and call
inference endpoint to generate music from input MIDI file

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
# Example
# python -m bach infer generate-music-by-transformer data_cache/data/albeniz-aragon_fantasia_op47_part_6.mid ./
#
# output
# Downloading model from Hugging Face Hub (https://huggingface.co/claudezss/bach) to data_cache\model
# Running inference on cuda
# Generated MIDI file was saved at: input.mid
# Generated MIDI file was saved at: temperature-0.1.mid
# Generated MIDI file was saved at: temperature-0.5.mid
# Generated MIDI file was saved at: temperature-1.0.mid
```