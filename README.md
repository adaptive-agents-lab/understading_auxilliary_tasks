# Code for "When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning"

Authors: Claas Voelcker, Tyler Kastner, Igor Gilitschenski, Amir-massoud Farahmand

## Installation

Please ensure that you have a cuda12 capable GPU installed. Other GPUs can work, but we do not provide installation help.
All dependencies are best installed via pip using the provided `pyproject.toml` and we strongly recommend using [uv](https://docs.astral.sh/uv/).
With this tool you can simply execute `uv run mad_td/main.py` and all requirements will be installed in a virtual environment.

The codebase is a prior version of [mad-td](https://github.com/adaptive-agents-lab/MAD-TD), so in case of issues, it's always good to double check in that repo as well.

## Paper experiments

We provide raw results in the corresponding folder.

## Citation

If you use our paper or results, please cite us as 

```
@InProceedings{voelcker2024understanding,
  title={When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning},
  author={Voelcker, Claas and Kastner, Tyler and Gilitschenski, Igor and Farahmand, Amir-massoud},
  booktitle={Proceedings of the Reinforcement Learning Conference},
  year={2024}
}
```
