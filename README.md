# XOR Neural Network

A NumPy implementation of a 2-2-1 neural network that learns the XOR function.

## Project Structure

```text
xor-neural-network/
├── xor_neural_network.py
├── requirements.txt
├── README.md
└── outputs/
```

## Setup

1. Open a terminal in this project folder.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the project:

```powershell
python xor_neural_network.py
```

The script prints test probabilities and accuracy, saves the trained model as
`outputs/xor_model.npz`, and exports the loss curve and decision-boundary chart
as `outputs/fig_5_1_output_plots.png`.
