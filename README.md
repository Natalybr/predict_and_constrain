# Predict and Constrain

This repository contains the TensorFlow source code to reproduce the experiments introduced in the paper __[Predict and Constrain: Modeling Cardinality in Deep Structured Prediction](http://proceedings.mlr.press/v80/brukhim18a/brukhim18a.pdf)__. Nataly Brukhim, Amir Globerson. ICML 2018.



### Requirements
To run this code you need to have tensorflow, numpy, liac-arff, and scikit-learn installed.
Install with 
```bash
pip install -r requirements.txt
```

### Replicating the experiments in the paper
__Bibtex__

To replicate the numbers for bibtex provided in the paper, run:
```python train.py```
By default, the model weights and logs are stored to `./bibtex_model`.
You can monitor the process using tensorboard with

`tensorboard --logdir ./bibtex_model/`

