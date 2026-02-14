import invoke


@invoke.task()
def finetune(c):
    """Fine-tune BrainHarmonix harmonizer. """

    print("Fine-tuning BrainHarmonix harmonizer with self-supervised objective...")

    # zscore experiment
    cmd = "preventad-finetune-brainharmonix --dataset=data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow --task=self-supervised --output-dir=outputs/finetune/brainharmonix/zscore.self-supervised --task=self-supervised  --epoch=25 --lr=1e-3 --weight-decay=0.01"

    print(f"Running: {cmd}")
    c.run(cmd)

    # zscore extract
    cmd = "preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow --harmonizer-ckpt outputs/finetune/brainharmonix/zscore.self-supervised/harmonizer_checkpoint_final.pt --output-prefix zscore.brainharmonix.finetuned  --fmri-ckpt outputs/finetune/brainharmonix/zscore.self-supervised/fmri_encoder_checkpoint_final.pt  --t1-ckpt outputs/finetune/brainharmonix/zscore.self-supervised/t1_encoder_checkpoint_final.pt --is-finetuned"

    c.run(cmd)

    # nozscore experiment
    cmd = "preventad-finetune-brainharmonix --dataset=data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow --task=self-supervised --output-dir=outputs/finetune/brainharmonix/nozscore.self-supervised --task=self-supervised  --epoch=25 --lr=1e-4 --weight-decay=0.01"

    print(f"Running: {cmd}")
    c.run(cmd)

    # nozscore extract
    cmd = "preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow --harmonizer-ckpt outputs/finetune/brainharmonix/nozscore.self-supervised/harmonizer_checkpoint_final.pt --output-prefix nozscore.brainharmonix.finetuned  --fmri-ckpt outputs/finetune/brainharmonix/nozscore.self-supervised/fmri_encoder_checkpoint_final.pt  --t1-ckpt outputs/finetune/brainharmonix/nozscore.self-supervised/t1_encoder_checkpoint_final.pt --is-finetuned"
    c.run(cmd)


@invoke.task(
    help={
        "dataset": "Path to BrainHarmonix Arrow dataset",
        "output-dir": "Output directory for embeddings",
        "output-prefix": "Prefix for output files",
    }
)
def extract(
    c,
    dataset=None,
    output_dir="outputs/embeddings/brainharmonix",
    output_prefix="brainharmonix",
):
    """Extract embeddings from BrainHarmonix model.

    Extracts embeddings from:
    - fMRI encoder (Harmonix-F): 7200 tokens × 768 dim
    - T1 encoder (Harmonix-S): 1200 tokens × 768 dim
    - Harmonizer: 129 tokens × 768 dim (1 CLS + 128 latent)

    Example:
        inv evaluation.extract-brainharmonix --dataset=./data/processed/dataset.arrow
    """
    dataset = dataset or "data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow"

    cmd = f"preventad-extract-brainharmonix --dataset {dataset} --output-dir {output_dir} --output-prefix {output_prefix}"

    print(f"Running: {cmd}")
    c.run(cmd)