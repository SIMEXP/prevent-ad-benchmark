"""Invoke task collections for PreventAD benchmark.

Available task namespaces:
- prepare: Data preparation tasks (preprocessing, atlas setup)
- models: Model training and feature extraction
- evaluation: Downstream evaluation and visualization
- notebooks: Jupyter notebook execution

Run `invoke --list` to see all available tasks.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import invoke

from . import brainharmonix, brainlm, baseline, prepare, reports


@invoke.task
def clean(c):
    """Remove all intermediate data files."""
    c.run("rm -rf ./data/interim/*")


# The main namespace MUST be named `namespace` or `ns`.
# See: http://docs.pyinvoke.org/en/1.2/concepts/namespaces.html
ns = invoke.Collection()

ns.add_collection(prepare)
ns.add_collection(brainlm)
ns.add_collection(brainharmonix)
ns.add_collection(baseline)
ns.add_collection(reports)

ns.add_task(clean)
