import invoke

@invoke.task
def finetune_brainlm(c):
    c.run(
        "python -m hfplayground.train "
        "./data/processed/dataset-preventad.brainlm.arrow "
        "./models/brainlm.finetuned/vitmae_111M.og "
        "--image-column-name Subtract_Mean_Divide_Global_STD_Normalized_Recording "
        "--model-params 111M"
    )
    c.run(
        "python -m hfplayground.extract "
        "./data/processed/dataset-preventad.brainlm.arrow "
        "./models/brainlm.finetuned/vitmae_111M.og "
        "./outputs/brainlm.vitmae_111M.finetuned.og "
        "--image-column-name Subtract_Mean_Divide_Global_STD_Normalized_Recording "

    )

    c.run(
        "python -m hfplayground.train "
        "./data/processed/dataset-preventad.brainlm.arrow "
        "./models/brainlm.finetuned/vitmae_650M.og "
        "--image-column-name Subtract_Mean_Divide_Global_STD_Normalized_Recording "
        "--model-params 650M"
    )
    c.run(
        "python -m hfplayground.extract "
        "./data/processed/dataset-preventad.brainlm.arrow "
        "./models/brainlm.finetuned/vitmae_650M.og "
        "./outputs/brainlm.vitmae_650M.finetuned.og "
        "--image-column-name Subtract_Mean_Divide_Global_STD_Normalized_Recording "
    )


@invoke.task
def finetune_brainlm_gigaconnectome(c):
    c.run(
        "python -m hfplayground.train "
        "./data/processed/dataset-preventad.gigaconnectome.arrow "
        "./models/brainlm.finetuned/vitmae_111M.gigaconnectome "
        "--image-column-name robustscaler_timeseries "
        "--model-params 111M"
    )

    c.run(
        "python -m hfplayground.extract "
        "./data/processed/dataset-preventad.gigaconnectome.arrow "
        "./models/brainlm.finetuned/vitmae_111M.gigaconnectome "
        "./outputs/brainlm.vitmae_111M.finetuned.gigaconnectome "
        "--image-column-name robustscaler_timeseries "

    )

    c.run(
        "python -m hfplayground.train "
        "./data/processed/dataset-preventad.gigaconnectome.arrow "
        "./models/brainlm.finetuned/vitmae_650M.gigaconnectome "
        "--image-column-name robustscaler_timeseries "
        "--model-params 650M"
    )
    c.run(
        "python -m hfplayground.extract "
        "./data/processed/dataset-preventad.gigaconnectome.arrow "
        "./models/brainlm.finetuned/vitmae_650M.gigaconnectome "
        "./outputs/brainlm.vitmae_650M.finetuned.gigaconnectome "
        "--image-column-name robustscaler_timeseries "
    )


@invoke.task
def direct_transfer(c):
    c.run(
        "python -m hfplayground.extract "
        "./data/processed/dataset-preventad.brainlm.arrow "
        "./models/brainlm/vitmae_111M "
        "./outputs/brainlm.vitmae_111M.direct_transfer.og "
        "--image-column-name Subtract_Mean_Divide_Global_STD_Normalized_Recording "

    )
    c.run(
        "python -m hfplayground.extract "
        "./data/processed/dataset-preventad.brainlm.arrow "
        "./models/brainlm/vitmae_650M "
        "./outputs/brainlm.vitmae_650M.direct_transfer.og "
        "--image-column-name Subtract_Mean_Divide_Global_STD_Normalized_Recording "
    )
    c.run(
        "python -m hfplayground.extract "
        "./data/processed/dataset-preventad.gigaconnectome.arrow "
        "./models/brainlm/vitmae_111M "
        "./outputs/brainlm.vitmae_111M.direct_transfer.gigaconnectome "
        "--image-column-name robustscaler_timeseries"

    )
    c.run(
        "python -m hfplayground.extract "
        "./data/processed/dataset-preventad.gigaconnectome.arrow "
        "./models/brainlm/vitmae_650M "
        "./outputs/brainlm.vitmae_650M.direct_transfer.gigaconnectome "
        "--image-column-name robustscaler_timeseries"
    )

