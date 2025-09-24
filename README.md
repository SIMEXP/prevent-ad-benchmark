# PrenventAD benchmark

Using BrainLM (and perhaps more) to explore phenotype prediction in PreventAD.

## Run this project

Please use [`uv`](https://docs.astral.sh/uv/) to install this project for the smoothest experience.

```
git clone git@github.com:SIMEXP/prevent-ad-benchmark.git
```

The brainLM submodule is for record keeping.
However, if you wish to pull it, run:

```
git submodule update --init --recursive
```

## Create virtual environment

On Rorqual, remember to add module:
```
module add cudacore/.12.6.2
module load httpproxy
```
With `uv`
```
uv venv
uv sync --extra build
uv sync --extra build --extra compile
```

You can either activate the environment with `source .venv/bin/activate` and use this environment in the conventional python way,
or prepend any command you want to run with `uv run` to activate the environment.


## Download models and example data

With `uv`, example:
```
uv run invoke prepare.models
uv run invoke prepare.data
```

Or with your virtual environment
```
source .venv/bin/activate
invoke prepare.models
invoke prepare.atlas
invoke prepare.data
invoke prepare.brainlm-workflow-timeseries
invoke prepare.prepare.gigaconnectome-workflow-timeseries
```
Check out `uv run invoke --list` for the commands and documentations.

## Other unrelated notes

### `uv`

[Quick tutorial with pip->uv table](https://www.datacamp.com/tutorial/python-uv)
