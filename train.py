import argparse
import glob
import os
import socket
import sys

import pytorch_lightning
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging,
    ModelSummary,
)
from pytorch_lightning.callbacks import TQDMProgressBar as ProgressBar
from pytorch_lightning.loggers import CometLogger

import utils.fileutils as fu
from lightning import FlowLightning


def parse_args():
    """
    Argument parser for training script.
    """
    parser = argparse.ArgumentParser(description="Train the Diffusion tagger.")

    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Path to config file."
    )
    parser.add_argument(
        "--gpus", default="", type=str, help="Comma separated list of GPUs to use."
    )
    parser.add_argument(
        "--ckpt_path", type=str, help="Restart training from a checkpoint."
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="No logging, checkpointing, test on a few jets.",
    )
    parser.add_argument("--batch_size", type=int, help="Overwrite config batch size.")
    parser.add_argument(
        "--reduce_dataset", type=float, help="Modify dataset size on the fly."
    )
    parser.add_argument("--num_workers", type=int, help="Overwrite config num workers.")
    parser.add_argument("--num_epochs", type=int, help="Overwrite config num epochs.")
    parser.add_argument(
        "--no_logging", action="store_true", help="Disable the logging framework."
    )

    args = parser.parse_args()
    return args


def update_config(args, config):
    """
    Update config with passed arguments
    """
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    return config


def setup_logger(args, config):
    """
    Handle logging for the DDP multi-gpu training.
    I think this is really a bug on comet's end that they don't
    automatically group the experiments...
    """

    # no logging
    if args.test_run or args.no_logging:
        return None

    # need to set up a new experiment
    comet_logger = setup_comet_logger(config, args, os.environ.get("COMET_EXP_ID"))
    if comet_logger.experiment.get_key():
        os.environ["COMET_EXP_ID"] = comet_logger.experiment.get_key()

    return comet_logger


def setup_comet_logger(config, args, exp_id=None):
    # initialise logger
    comet_logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        save_dir="logs",
        project_name=os.environ["COMET_PROJECT_NAME"],
        workspace=os.environ["COMET_WORKSPACE"],
        experiment_name=config["name"],
        experiment_key=exp_id,
    )

    # log config, hyperparameters and source files
    if os.environ.get("LOCAL_RANK") is None:
        comet_logger.experiment.log_parameter("batchsize", config["batchsize"])
        comet_logger.experiment.log_parameter("learningrate", config["learningrate"])
        comet_logger.experiment.log_parameter("num_epochs", config["num_epochs"])
        comet_logger.experiment.log_parameter("num_gpus", config["num_gpus"])
        comet_logger.experiment.log_parameter("num_workers", config["num_workers"])
        comet_logger.experiment.log_parameter("use_swa", config["use_swa"])

        comet_logger.experiment.log_parameter("torch_version", torch.__version__)
        comet_logger.experiment.log_parameter(
            "lightning_version", pytorch_lightning.__version__
        )
        comet_logger.experiment.log_parameter("cuda_version", torch.version.cuda)
        comet_logger.experiment.log_parameter("hostname", socket.gethostname())

        comet_logger.experiment.log_asset(args.config)
        all_files = glob.glob("./*.py") + glob.glob("models/*.py")
        for fpath in all_files:
            comet_logger.experiment.log_code(fpath)

    return comet_logger


def get_callbacks(config, args):
    """
    Initialise training callbacks
    """

    refresh_rate = 1 if args.test_run else 20
    callbacks = [ProgressBar(refresh_rate=refresh_rate), ModelSummary(max_depth=2)]

    # initialise checkpoint callback
    if not args.test_run:
        monitor_loss = "val_loss_avg"

        # filename template
        file_name = config["run_name"] + "-{epoch:02d}-{" + monitor_loss + ":.4f}"

        # callback
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_loss,
            dirpath=os.path.join("saved_models/", config["run_name"], "ckpts"),
            filename=file_name,
            save_top_k=-1,
            save_last=True,
        )
        callbacks += [checkpoint_callback]

    if config["use_swa"]:
        callbacks += [StochasticWeightAveraging(swa_lrs=float(config["learningrate"]))]

    return callbacks


def train(args, config, logger):
    """
    Fit the model.
    """

    # create a new model
    model = FlowLightning(config)

    # log number of parametesr
    if os.environ.get("LOCAL_RANK") is None and logger is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.experiment.log_parameter("trainable_params", trainable_params)

    # if we pass a checkpoint file, load the previous network
    if args.ckpt_path:
        print("Loading previously trained model from checkpoint file:", args.ckpt_path)
        model = FlowLightning.load_from_checkpoint(args.ckpt_path, config=config)

    # share workers between GPUs
    if config["num_gpus"]:
        config["num_workers"] = config["num_workers"] // config["num_gpus"]

    # get callbacks
    callbacks = get_callbacks(config, args)

    # create the lightening trainer
    print("Creating trainer...")
    trainer = Trainer(
        max_epochs=config["num_epochs"],
        accelerator=config["accelerator"],
        devices=config["num_gpus"],
        logger=logger,
        log_every_n_steps=20,
        fast_dev_run=args.test_run,
        callbacks=callbacks,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        gradient_clip_val=config.get("gradient_clip_val", 0),
    )

    # fit model model
    print("Fitting model...")
    if args.ckpt_path:
        trainer.fit(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model)

    return model, trainer


def print_job_info(args, config):
    """
    Print job information.
    """

    if os.environ.get("LOCAL_RANK") is not None:
        return

    print("-" * 100)
    print("torch", torch.__version__)
    print("lightning", pytorch_lightning.__version__)
    print("cuda", torch.version.cuda)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("Visible GPUs:", args.gpus, " - ", device_name)
    print("-" * 100, "\n")


def parse_gpus(config, gpus):
    # set available GPUs based on arguments
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    num_gpus = len(gpus.split(",")) if gpus != "" else None
    accelerator = "gpu" if num_gpus is not None else "cpu"
    config["accelerator"] = accelerator
    config["num_gpus"] = num_gpus

    return config


def cleanup(config, model):
    print("-" * 100)
    print("Cleaning up...")
    # keep main process only
    if model.global_rank != 0:
        sys.exit(0)

    # fu.remove_files_temp(config, tag="Training")


def main():
    """
    Training entry point.
    """
    # pytorch_lightning.seed_everything(42)

    # parse args
    args = parse_args()

    # read config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # overwrite config using args
    config = update_config(args, config)

    # parse the gpu argument and update config
    config = parse_gpus(config, args.gpus)

    # print job info once
    print_job_info(args, config)

    # setup logger
    logger = setup_logger(args, config)

    # copy files to output dir for reproducability
    if not args.test_run:
        config = fu.prep_out_dir(args, config)

    if args.test_run:
        config["reduce_ds_train"] = config.get("val_batchsize", config["batchsize"])
        config["reduce_ds_valid"] = config.get("val_batchsize", config["batchsize"])
    # run training
    model, trainer = train(args, config, logger)

    # cleanup
    if not args.test_run:
        cleanup(config, model)


if __name__ == "__main__":
    main()
