# Adapter layer wrapping the `brainharmonix` pip package (github.com/MedARC-AI/Brain-Harmony,
# pinned in pyproject.toml) so it can be driven from cli/finetune_brainharmonix.py and
# cli/extract_brainharmonix.py with PreventAD's Arrow datasets and checkpoint layout.
#
# Scope: self-supervised finetuning only (see loaders.py / models.py / fintuning_engines.py
# docstrings). Classification/regression are stubbed to raise NotImplementedError.
