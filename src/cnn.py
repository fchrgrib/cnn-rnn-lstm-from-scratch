import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Union, Optional, Any
from numpy.lib.stride_tricks import as_strided


class CNNFromScratch:
    def __init__(self, model_path: str):
        """
        Initialize CNN model from a Keras model pickle file.

        Args:
            model_path: Path to the Keras model pickle file
        """
        self.model_path = model_path
        self.weights = self._load_model()
        self.layers = []

    def _load_model(self) -> Dict[str, np.ndarray]:
        """Load Model Pkl File"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")

        try:
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def add_layer(self, layer_type: str, **kwargs) -> 'CNNFromScratch':
        """
        Optimized layer addition with type checking
        layer_type: str [dense, conv2d, maxpooling2d]
        """
        layer_type = layer_type.lower()
        layer_map = {
            'dense': {
                'type': 'dense',
                'name': kwargs.get('name', f'dense_{len(self.layers)}'),
                'activation': kwargs.get('activation', 'linear'),
                'units': kwargs.get('units')
            },
            'conv2d': {
                'type': 'conv2d',
                'name': kwargs.get('name', f'conv2d_{len(self.layers)}'),
                'filters': kwargs.get('filters'),
                'kernel_size': kwargs.get('kernel_size', (3, 3)),
                'strides': kwargs.get('strides', (1, 1)),
                'padding': kwargs.get('padding', 'valid'),
                'activation': kwargs.get('activation', 'linear')
            },
            'maxpooling2d': {
                'type': 'maxpooling2d',
                'name': kwargs.get('name', f'maxpool_{len(self.layers)}'),
                'pool_size': kwargs.get('pool_size', (2, 2)),
                'strides': kwargs.get('strides')
            },
            'flatten': {
                'type': 'flatten',
                'name': kwargs.get('name', f'flatten_{len(self.layers)}')
            }
        }

        if layer_type not in layer_map:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.layers.append(layer_map[layer_type])
        return self

    def _activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """
        Calculate activation function.

        Args:
            x: Input array
            activation: Activation function name (e.g., 'relu', 'sigmoid', 'tanh', 'softmax', 'linear')
        """
        activations = {
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': np.tanh,
            'softmax': lambda x: np.exp(x - np.max(x, axis=-1, keepdims=True)) /
                                 np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)),
                                        axis=-1, keepdims=True),
            'linear': lambda x: x
        }

        return activations.get(activation.lower(), lambda x: x)(x)

    def _conv2d(self, input_data: np.ndarray, weights: np.ndarray,
                bias: np.ndarray, kernel_size: Tuple[int, int],
                strides: Tuple[int, int], padding: str) -> np.ndarray:
        """
        Conv 2D operation using strided view for optimized performance.
        :param input_data:
        :param weights:
        :param bias:
        :param kernel_size:
        :param strides:
        :param padding:
        :return:
        """
        batch_size, in_height, in_width, in_channels = input_data.shape
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = strides

        # Calculate output dimensions based on padding
        if padding.lower() == 'same':
            out_height = (in_height + stride_h - 1) // stride_h
            out_width = (in_width + stride_w - 1) // stride_w
            pad_h = max((out_height - 1) * stride_h + kernel_h - in_height, 0)
            pad_w = max((out_width - 1) * stride_w + kernel_w - in_width, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded_input = np.pad(input_data,
                                  ((0, 0), (pad_top, pad_bottom),
                                   (pad_left, pad_right), (0, 0)),
                                  mode='constant')
        else:  # 'valid' padding
            padded_input = input_data
            out_height = (in_height - kernel_h) // stride_h + 1
            out_width = (in_width - kernel_w) // stride_w + 1

        # Create strided view of the padded input
        shape = (batch_size, out_height, out_width, kernel_h, kernel_w, in_channels)
        strides = (padded_input.strides[0],
                   padded_input.strides[1] * stride_h,
                   padded_input.strides[2] * stride_w,
                   padded_input.strides[1],
                   padded_input.strides[2],
                   padded_input.strides[3])

        # Create a strided view of the padded input
        strided_input = as_strided(padded_input, shape=shape, strides=strides, writeable=False)

        """
        Perform convolution using einsum for optimized performance.
        The einsum operation computes the dot product between the strided input and the weights.
        The output shape will be (batch_size, out_height, out_width, filters).
        """
        output = np.einsum('bhwijc,ijcf->bhwf', strided_input, weights)
        output += bias  # Add bias

        return output

    def _maxpool2d(self, input_data: np.ndarray, pool_size: Tuple[int, int],
                   strides: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Max pooling operation using strided view for optimized performance.
        :param input_data:
        :param pool_size:
        :param strides:
        :return:
        """
        batch_size, in_height, in_width, channels = input_data.shape
        pool_h, pool_w = pool_size
        stride_h, stride_w = strides if strides is not None else pool_size

        out_height = (in_height - pool_h) // stride_h + 1
        out_width = (in_width - pool_w) // stride_w + 1

        # Create strided view
        shape = (batch_size, out_height, out_width, pool_h, pool_w, channels)
        strides = (input_data.strides[0],
                   input_data.strides[1] * stride_h,
                   input_data.strides[2] * stride_w,
                   input_data.strides[1],
                   input_data.strides[2],
                   input_data.strides[3])

        strided_input = as_strided(input_data, shape=shape, strides=strides, writeable=False)

        # Max pooling along the spatial dimensions
        return np.max(strided_input, axis=(3, 4))

    def _dense(self, input_data: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """
        Dense layer operation using matrix multiplication.
        :param input_data:
        :param weights:
        :param bias:
        :return:
        """
        return np.dot(input_data, weights) + bias

    def _flatten(self, input_data: np.ndarray) -> np.ndarray:
        """
        Flatten operation to convert input data to 2D.
        :param input_data:
        :return:
        """
        return input_data.reshape(input_data.shape[0], -1)

    def _forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the CNN layers.
        :param input_data:
        :return:
        """
        x = input_data

        for layer in self.layers:
            layer_type = layer['type']
            layer_name = layer['name']

            weights = self.weights.get(f"{layer_name}/kernel")
            bias = self.weights.get(f"{layer_name}/bias")

            if layer_type == 'dense':
                if weights is None or bias is None:
                    raise ValueError(f"Missing weights/bias for layer {layer_name}")
                x = self._dense(x, weights, bias)
                x = self._activation(x, layer['activation'])

            elif layer_type == 'conv2d':
                if weights is None or bias is None:
                    raise ValueError(f"Missing weights/bias for layer {layer_name}")
                x = self._conv2d(x, weights, bias,
                                 layer['kernel_size'],
                                 layer['strides'],
                                 layer['padding'])
                x = self._activation(x, layer['activation'])

            elif layer_type == 'maxpooling2d':
                x = self._maxpool2d(x, layer['pool_size'], layer['strides'])

            elif layer_type == 'flatten':
                x = self._flatten(x)

        return x

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predict method to handle both single sample and batch inputs.
        :param input_data:
        :return:
        """
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        elif len(input_data.shape) != 4:
            raise ValueError("Input must be 3D (single sample) or 4D (batch) array")

        return self._forward(input_data)
