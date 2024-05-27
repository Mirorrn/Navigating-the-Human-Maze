# Navigating the Human Maze: Real-Time Robot Pathfinding with Generative Imitation Learning

This repository contains the source code for the research paper: [Navigating the Human Maze: Real-Time Robot Pathfinding with Generative Imitation Learning](https://human-maze-navigation.github.io). It builds upon our previous research: [Model-based Imitation Learning for Real-time Robot Navigation in Crowds](https://ieeexplore.ieee.org/document/10309382).

## Installation

1. Set up a Python environment on your local machine:

    ```bash
    $ python -m venv '/UserDefinedDirectoryPath/d3rlpyAgentsENV'
    $ source /UserDefinedDirectoryPath/d3rlpyAgentsENV/bin/activate
    ```

2. Install the `d3rlpy` open-source library from source:

    ```bash
    $ git clone https://github.com/takuseno/d3rlpy
    $ cd d3rlpy
    $ pip install -e .
    ```

3. Install PyTorch (this version uses PyTorch 2.0.1):

    ```bash
    $ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    ```

4. Finally, install the necessary additional modules:

    ```bash
    $ pip install -r requirements.txt
    ```

## Usage

To run the main policy script, use the following command:

```bash
$ python MPPI_policy.py
```
## Citation

If you find this work useful, please cite it as follows:

```code
@inproceedings{moder2024,
    title={Navigating the {H}uman {M}aze: {R}eal-{T}ime {R}obot {P}athfinding with {G}enerative {I}mitation {L}earning},
    author={Moder, Martin and Adhisaputra, Stephen and Pauli, Josef},
    year={2024},
    month={05},
    pages={},
    doi={10.13140/RG.2.2.27426.44485}
}


@inproceedings{moder2023model,
    title={Model-based Imitation Learning for Real-time Robot Navigation in Crowds},
    author={Moder, Martin and {\"O}zgan, Fatih and Pauli, Josef},
    booktitle={2023 32nd IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)},
    pages={513--519},
    year={2023},
    organization={IEEE}
}
```
