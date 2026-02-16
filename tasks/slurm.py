"""SLURM job array utilities for parallel GPU compute."""

import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLURM_CONFIG_PATH = PROJECT_ROOT / "slurm_config.yaml"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_slurm_config(task_name=None):
    """Load SLURM config, merging defaults with task-specific overrides.

    Parameters
    ----------
    task_name : str, optional
        Key in the ``overrides`` section of slurm_config.yaml.

    Returns
    -------
    dict
        Merged SLURM parameters.
    """
    with open(SLURM_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    params = dict(config.get("defaults", {}))
    if task_name and task_name in config.get("overrides", {}):
        params.update(config["overrides"][task_name])
    return params


SCRATCH_LOGS_DIR = Path("/scratch/hwang1/prevent-ad-benchmark/logs")


def _ensure_logs_dir():
    """Ensure logs/ is a symlink to scratch."""
    SCRATCH_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if LOGS_DIR.is_symlink() or LOGS_DIR.exists():
        return
    LOGS_DIR.symlink_to(SCRATCH_LOGS_DIR)


def submit_job_array(job_name, command_template, array_range, slurm_params, dry_run=False):
    """Generate and submit a SLURM job array script.

    Parameters
    ----------
    job_name : str
        SLURM job name (also used for the script filename).
    command_template : str
        Shell command(s) to execute. Use ``$SLURM_ARRAY_TASK_ID`` for the
        split index.
    array_range : str
        SLURM array range, e.g. ``"0-19"``.
    slurm_params : dict
        SLURM resource parameters (account, time, mem, etc.).
    dry_run : bool
        If True, print the script and sbatch command without submitting.

    Returns
    -------
    str or None
        The SLURM job ID if submitted, None if dry_run.
    """
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_logs_dir()

    account = slurm_params.get("account", "rrg-pbellec")
    time = slurm_params.get("time", "4:00:00")
    mem = slurm_params.get("mem", "32G")
    cpus = slurm_params.get("cpus_per_task", 4)
    gres = slurm_params.get("gres", "gpu:1")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --time={time}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --array={array_range}",
        f"#SBATCH --output=logs/{job_name}_%A_%a.out",
        f"#SBATCH --error=logs/{job_name}_%A_%a.err",
        "",
        f"cd {PROJECT_ROOT}",
        "source .venv/bin/activate",
        "",
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        "export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        "export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        "",
        command_template,
    ]
    script = "\n".join(lines) + "\n"

    script_path = SCRIPTS_DIR / f"{job_name}.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    sbatch_cmd = f"sbatch {script_path}"

    if dry_run:
        print(f"--- {script_path} ---")
        print(script)
        print(f"Would run: {sbatch_cmd}")
        return None

    print(f"Submitting: {sbatch_cmd}")
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    # sbatch output: "Submitted batch job 12345678"
    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted job array {job_name}: {job_id}")
    return job_id
