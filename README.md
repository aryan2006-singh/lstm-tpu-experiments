# LSTM on a TPU: why a small model barely uses the chip

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aryan2006-singh/lstm-tpu-experiments/blob/main/tpu_experiments.ipynb)

Three short experiments on the Tesla LSTM from [tesla-rnn-vs-lstm](https://github.com/aryan2006-singh/tesla-rnn-vs-lstm), run on a **TPU runtime** with the Keras JAX backend.

1. **Compile vs run time.** The first epoch includes XLA compiling the model, so it is timed separately from later epochs.
2. **Batch size sweep.** A TPU's matrix unit works on 128x128 tiles. The notebook trains a 32-unit and a 256-unit LSTM at batch sizes 32, 128, 512 and 1024 and reports steady milliseconds per epoch and samples per second.
3. **What XLA produced.** It prints the compiled program for one small function, counts the fused kernels, then captures a profile you can open in TensorBoard.

Each result is followed by a plain-English explanation. An LSTM runs its 60 time steps one after another, which limits how much work the chip can do in parallel.

## Running it

1. Open the notebook in Google Colab (button above) and choose a **TPU** runtime.
2. Upload `TSLA.csv` (a `Date` column and a `Close` column) when the notebook asks. The data file is not included in this repository.
3. Run all cells.

The notebook uses the JAX backend (`KERAS_BACKEND=jax`), which is set in its first code cell.
