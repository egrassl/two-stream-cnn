# Two-Stream CNN with Keras

This repository implements the Two-Stream CNN approach for action video recognition and it was implemented for the Electrial Engineering masters' degree at [**Centro Universitário FEI**](https://portal.fei.edu.br/).

## Pre-requisites
To run this code, you will need:
* Python 3.7 interpreter (conda virtual environment recommended);
* NVIDIA Graphic card with CUDA installed (testes on versions 10 and 11).

## How to use
Is simple and easy to run this code. You just need to follow few steps that are listed bellow:

1. Clone/Download this repo to your PC;
2. If you do not have Python 3.7.0, download it from [Python's website](https://www.python.org/);
3. Set Python 3.7.0 as the project's interpreter;
4. Install dependencies through `conda install --file requirements.txt` command;

You are ready to go now :+1:!

I hope you enjoy and learn with the beauty of Machine Learning as much as I did :punch:!

## Content
This project contains the following functionalities:
- [x] TS CNN for spatial, temporal and spatio-temporal models;
- [x] Train script for each stream and spatio-temporal models through `train.py` script;
- [x] Data tool to extract data from videos into RGB motion flow images though `dt.py` script;
