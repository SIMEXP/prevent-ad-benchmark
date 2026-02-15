import invoke


@invoke.task(
    help={
        "split-index": "Index of the train/test split (default: 0)",
    }
)
def finetune_and_extract(c, split_index=0):
    """Fine-tune BrainHarmonix harmonizer. """

    print("Fine-tuning BrainHarmonix harmonizer with self-supervised objective...")

    # zscore experiment
    cmd = f"preventad-finetune-brainharmonix --dataset=data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow --task=self-supervised --output-dir=outputs/finetune/brainharmonix/zscore.self-supervised --epoch=50 --lr=1e-4 --weight-decay=0.01 --split-index {split_index}"

    print(f"Running: {cmd}")
    c.run(cmd)

    # zscore extract
    cmd = f"preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow --harmonizer-ckpt outputs/finetune/brainharmonix/zscore.self-supervised/harmonizer_checkpoint_best.pt --output-prefix zscore.brainharmonix.finetuned --split-index {split_index}"

    c.run(cmd)


    # nozscore experiment
    cmd = f"preventad-finetune-brainharmonix --dataset=data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow --task=self-supervised --output-dir=outputs/finetune/brainharmonix/nozscore.self-supervised --epoch=25 --lr=1e-4 --weight-decay=0.01 --split-index {split_index}"

    print(f"Running: {cmd}")
    c.run(cmd)

    # nozscore extract
    cmd = f"preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow --harmonizer-ckpt outputs/finetune/brainharmonix/nozscore.self-supervised/harmonizer_checkpoint_best.pt --output-prefix nozscore.brainharmonix.finetuned --split-index {split_index}"
    c.run(cmd)


@invoke.task()
def extract(c):
    """Use BrainHarmonix harmonizer as feature extractor directly. """

    print("Use BrainHarmonix harmonizer as feature extractor directly.")

    for split_index in range(20):
        # zscore extract on default checkpoint
        cmd = f"preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow --output-prefix zscore.brainharmonix --split-index {split_index}"

        c.run(cmd)

        # nozscore extract on default checkpoint
        cmd = f"preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow --output-prefix nozscore.brainharmonix --split-index {split_index}"

        c.run(cmd)
