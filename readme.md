We provide code of ***Learning Low-Rank Tensor Cores with Probabilistic ℓ0-Regularized
Rank Selection for Model Compression***.

The conda environment can be found in `enironment.yml`.

The implementation of the relaxed Bernoulli gates (built upon [runopti/stg: Python/R library for feature selection in neural nets. ("Feature selection using Stochastic Gates", ICML 2020) (github.com)](https://github.com/runopti/stg)) is in `./model/gate.py`. The implementations of tensorized layers and the learning models are in `./model/`.

`tensorly`([TensorLy: Tensor Learning in Python — TensorLy: Tensor Learning in Python](https://tensorly.org/dev/index.html)) is the lib we use for the implementations of tensorized layers.

## Usage

Run `main.py` to see how the model performs. The global variable `model_names` is a list containing tuples of `(model_name, λ)`，λ is the regularization coefficient. Other hyperparameters are set through global variables. Model names can be found at the bottom of `./model/lenet5.py`.  The results and training log files will be generated in `./log/`.


## Citation

```
@inproceedings{ijcai2024p418,
  title     = {Learning Low-Rank Tensor Cores with Probabilistic ℓ0-Regularized Rank Selection for Model Compression},
  author    = {Cao, Tianxiao and Sun, Lu and Nguyen, Canh Hao and Mamitsuka, Hiroshi},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {3780--3788},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/418},
  url       = {https://doi.org/10.24963/ijcai.2024/418},
}
```
