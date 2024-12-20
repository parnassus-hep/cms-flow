# Official repository for "Parnassus: An Automated Approach to Accurate, Precise, and Fast Detector Simulation and Reconstruction" paper

This repository contains the code for the paper "Parnassus: An Automated Approach to Accurate, Precise, and Fast Detector Simulation and Reconstruction", [arXiv:X](X).

The dataset used in the paper can be found [here]([https://zenodo.org](https://zenodo.org/records/11389651)).

## Requirements
The list of packages required to train/evaluate model is found at `requirements.txt` file. All studies were done with `Python 3.8.15`.

## Training

The training script is provided in the `train.py` file. The script can be run as follows:

```bash
python train.py -c <path_to_config_file> --gpus 0
```

## Evaluation

The evaluation script is provided in the `eval.py` file. The script can be run as follows:

```bash
python eval.py -c <path_to_config> -p <path_to_checkpoint> \
--test_path <path_to_test_file> -ne <number_of_events> -bs <batch_size> \
-n <num_steps> -npf [--prefix <prefix>]
```

The pre-trained model used in the paper can be found in the `trained_models` folder.
