# 🧠 CNN, RNN, ans LSTM Forward Propagation from Scratch

This repository contains a custom implementation of Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory (LSTM) networks from scratch, focusing on forward propagation. The implementation is designed to be flexible and modular, allowing for easy experimentation with different architectures and hyperparameters.

## 📁 Project Structure

```
root/
├── src/
│   ├── cnn.py
│   ├── rnn_lstm.py
│   ├── cnn.ipynb (Jupyter Notebook for CNN)
│   ├── rnn.ipynb (Jupyter Notebook for RNN)
│   └── lstm.ipynb (Jupyter Notebook for LSTM)
├── model/
│   ├── cnn.pkl (manually saved model)
│   ├── rnn.pkl (manually saved model)
│   ├── lstm.pkl (manually saved model)
│   └── .gitkeep
├── data/ (NusaX dataset for RNN and LSTM)
│   ├── train.csv
│   ├── test.csv 
│   └── validation.csv 
└── README.md
```

## 🚀 Key Features

- **Custom Implementations**: CNN, RNN, and LSTM networks implemented from scratch.
- **Forward Propagation**: Focus on the forward pass of the networks, allowing for understanding of how data flows through the model.
- **Modular Design**: Each network is implemented in its own module, making it easy to modify and extend.
- **Experimentation**: Includes a Jupyter Notebook for running experiments and visualizing results.

## 📊 Experiments

All experiments are included in `cnn.ipynb`, `rnn.ipynb`, and `lstm.ipynb` notebooks. Each notebook contains:
- Implementation of the respective network.
- Training and validation loops.
- Evaluation of the model on the CIFAR10 for CNN and Nusax Data for RNN and LSTM.
- Visualization of results including accuracy and loss plots.

## 🧪 How to Run

### 1. Requirements
Make sure you have Python 3 and the following libraries:

```bash
pip install -r requirements.txt
```

### 2. Running the Notebook
Open `src/cnn.ipynb`, `src/rnn.ipynb`, or `src/lstm.ipynb` in Jupyter Notebook or any compatible environment. You can run the cells to execute the forward propagation and visualize the results.


## 👨‍💻 Contributor

- 13521031 | Fahrian Afdholi