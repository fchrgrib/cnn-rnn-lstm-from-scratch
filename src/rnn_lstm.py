import pickle
import numpy as np


class RNNLSTMFromScratch:
    def __init__(self, model_path):
        """
        Initialize RNNForwardProp with model path.

        :param model_path: Path to the pickled Keras model
        """

        self.model_path = model_path
        self.weights = {}
        self.model_config = {}
        self.load_model()

    def load_model(self):
        """Load weights and biases from Keras model pickle file"""

        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)

            # Get model architecture info
            self.model_config['layers'] = []
            for layer in model.layers:
                layer_info = {
                    'name': layer.name,
                    'type': type(layer).__name__,
                    'config': layer.get_config()
                }
                self.model_config['layers'].append(layer_info)

            # Extract weights from the model
            weights = model.get_weights()
            self.parse_weights(weights, model)

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def parse_weights(self, weights, model):
        """
        Parse weights from the model and assign them to the weights dictionary.

        :param weights: List of weight arrays from the model
        :param model: Keras model instance
        :return: None
        """

        weight_idx = 0

        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            layer_name = layer.name

            if layer_type == 'Embedding':
                # Embedding layer has one weight matrix
                if weight_idx < len(weights):
                    self.weights[f'{layer_name}_embeddings'] = weights[weight_idx]
                    weight_idx += 1

            elif layer_type == 'SimpleRNN':
                # SimpleRNN has 3 weight matrices in Keras
                if weight_idx + 2 < len(weights):
                    self.weights[f'{layer_name}_W_input'] = weights[weight_idx]
                    self.weights[f'{layer_name}_W_recurrent'] = weights[weight_idx + 1]
                    self.weights[f'{layer_name}_bias'] = weights[weight_idx + 2]
                    weight_idx += 3

            elif layer_type == 'LSTM':
                # LSTM has 3 weight matrices
                if weight_idx + 2 < len(weights):
                    self.weights[f'{layer_name}_W_input'] = weights[weight_idx]
                    self.weights[f'{layer_name}_W_recurrent'] = weights[weight_idx + 1]
                    self.weights[f'{layer_name}_bias'] = weights[weight_idx + 2]
                    weight_idx += 3

            elif layer_type == 'Dense':
                # Dense layer has weight matrix and bias
                if weight_idx + 1 < len(weights):
                    self.weights[f'{layer_name}_W'] = weights[weight_idx]
                    self.weights[f'{layer_name}_bias'] = weights[weight_idx + 1]
                    weight_idx += 2

    def _activation(self, x: np.array, activation: str):
        """
        Apply activation function to the input array.
        This method supports ReLU, Sigmoid, Tanh, Softmax, and Linear activations.

        :param x:
        :param activation:
        :return:
        """

        # Clip values to prevent overflow
        x = np.clip(x, -500, 500)

        if activation.lower() == 'relu':
            return np.maximum(0, x)
        elif activation.lower() == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation.lower() == 'tanh':
            return np.tanh(x)
        elif activation.lower() == 'softmax':
            return self._softmax(x)
        elif activation.lower() == 'linear':
            return x
        else:
            return x

    def _softmax(self, x):
        """
        Softmax is a function that converts logits to probabilities.
        It is numerically stable by subtracting the max value.
        This implementation supports both 1D and 2D inputs.

        :param x:
        :return:
        """

        # Handle both 1D and 2D inputs
        if x.ndim == 1:
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x)
        else:
            x_max = np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _get_dropout_rate(self, layer_name):
        """
        Dropout rate is a hyperparameter that controls the fraction of input units to drop.
        This method retrieves the dropout rate for a given layer.

        :param layer_name:
        :return:
        """

        for layer_info in self.model_config['layers']:
            if layer_info['name'] == layer_name and layer_info['type'] == 'Dropout':
                return layer_info['config'].get('rate', 0.0)
        return 0.0

    def _dropout(self, X, layer_name, training=False):
        """
        Dropout is a regularization technique to prevent overfitting.
        It randomly sets a fraction of input units to zero during training.

        :param X:
        :param layer_name:
        :param training:
        :return:
        """

        rate = self._get_dropout_rate(layer_name)

        if training:
            # During training, randomly set elements to zero and scale
            mask = np.random.binomial(1, 1 - rate, size=X.shape) / (1 - rate)
            return X * mask
        else:
            # During inference, no dropout is applied (Keras automatically handles scaling)
            return X

    def _create_mask(self, X, mask_value=0):
        """
        Mask is used to ignore certain values in the input data.
        This method creates a mask for the input data where the mask_value is considered as padding.
        Masking is useful for handling variable-length sequences in RNNs.
        The default value for masking is 0, which is common in NLP tasks.

        :param X:
        :param mask_value:
        :return:
        """

        return X != mask_value

    def embedding(self, X, layer_name):
        """
        Embedding layer converts integer indices to dense vectors.
        This method retrieves the embedding weights and applies them to the input indices.

        :param X:
        :param layer_name:
        :return:
        """

        embeddings = self.weights[f'{layer_name}_embeddings']

        # Convert to numpy if TensorFlow tensor
        if hasattr(X, 'numpy'):
            X = X.numpy()

        # Clip indices to valid range
        X = np.clip(X.astype(int), 0, embeddings.shape[0] - 1)

        # Apply embedding lookup
        embedded = embeddings[X]

        # Create mask for zero-padded positions
        mask = self._create_mask(X, mask_value=0)

        return embedded, mask

    def simple_rnn(self, X, layer_name, mask=None):
        """
        Simple RNN is a basic recurrent layer that processes sequences.
        This method performs forward propagation through a SimpleRNN layer.

        :param X:
        :param layer_name:
        :param mask: Mask to handle variable-length sequences
        :return:
        """

        W_input = self.weights[f'{layer_name}_W_input']
        W_recurrent = self.weights[f'{layer_name}_W_recurrent']
        bias = self.weights[f'{layer_name}_bias']

        batch_size, seq_len, input_dim = X.shape
        hidden_dim = W_recurrent.shape[0]

        # Initialize hidden state
        h = np.zeros((batch_size, hidden_dim))

        # Process each timestep
        for t in range(seq_len):
            # Get current input
            x_t = X[:, t, :]

            # Compute new hidden state
            # h_t = tanh(W_input * x_t + W_recurrent * h + bias)
            h_new = np.tanh(
                np.dot(x_t, W_input) +
                np.dot(h, W_recurrent) +
                bias
            )

            # Apply mask if provided
            if mask is not None:
                mask_t = mask[:, t].reshape(-1, 1)
                h = h_new * mask_t + h * (1 - mask_t)
            else:
                h = h_new

        return h

    def lstm(self, X, layer_name, mask=None):
        """
        LSTM (Long Short-Term Memory) is a type of RNN that can learn long-term dependencies.
        This method performs forward propagation through an LSTM layer.

        :param X:
        :param layer_name:
        :param mask: Mask to handle variable-length sequences
        :return:
        """

        W_input = self.weights[f'{layer_name}_W_input']
        W_recurrent = self.weights[f'{layer_name}_W_recurrent']
        bias = self.weights[f'{layer_name}_bias']

        batch_size, seq_len, input_size = X.shape
        hidden_size = W_recurrent.shape[0]

        # Initialize hidden and cell states
        h = np.zeros((batch_size, hidden_size))
        c = np.zeros((batch_size, hidden_size))

        # Process each timestep
        for t in range(seq_len):
            # Compute all gates at once
            # gates = W_input * x_t + W_recurrent * h + bias
            gates = np.dot(X[:, t, :], W_input) + np.dot(h, W_recurrent) + bias

            # Split gates (input, forget, candidate, output) - Keras order
            # i_gate = sigmoid(W_input * x_t + W_recurrent * h + bias[:hidden_size])
            i_gate = self._activation(gates[:, :hidden_size], 'sigmoid')

            # f_gate = sigmoid(W_input * x_t + W_recurrent * h + bias[hidden_size:2*hidden_size])
            f_gate = self._activation(gates[:, hidden_size:2 * hidden_size], 'sigmoid')

            # c_candidate = tanh(W_input * x_t + W_recurrent * h + bias[2*hidden_size:3*hidden_size])
            c_candidate = self._activation(gates[:, 2 * hidden_size:3 * hidden_size], 'tanh')

            # o_gate = sigmoid(W_input * x_t + W_recurrent * h + bias[3*hidden_size:])
            o_gate = self._activation(gates[:, 3 * hidden_size:], 'sigmoid')

            # c_new = f_gate * c + i_gate * c_candidate
            c_new = f_gate * c + i_gate * c_candidate

            # h_new = o_gate * tanh(c_new)
            h_new = o_gate * self._activation(c_new, 'tanh')

            # Apply mask
            if mask is not None:
                mask_t = mask[:, t].reshape(-1, 1)
                h = h_new * mask_t + h * (1 - mask_t)
                c = c_new * mask_t + c * (1 - mask_t)
            else:
                h = h_new
                c = c_new

        return h

    def dense(self, X, layer_name, activation='linear'):
        """
        Dense layer performs a linear transformation followed by an activation function.
        This method applies the weights and bias of a Dense layer to the input data.

        :param X:
        :param layer_name:
        :param activation:
        :return:
        """

        W = self.weights[f'{layer_name}_W']
        bias = self.weights[f'{layer_name}_bias']

        # Linear transformation
        output = np.dot(X, W) + bias

        # Apply activation
        output = self._activation(output, activation)
        return output

    def predict(self, X, training=False):
        """
        This method performs forward propagation through the RNN/LSTM model.
        It processes the input data through each layer in sequence, applying the appropriate operations.

        :param X: Input data (integer sequences for embedding)
        :param training: Whether to apply dropout
        :return: Output predictions
        """

        current_output = X
        mask = None

        # Process each layer in sequence
        for layer_info in self.model_config['layers']:
            layer_name = layer_info['name']
            layer_type = layer_info['type']
            layer_config = layer_info['config']

            if layer_type == 'Embedding':
                # Check if masking is enabled
                mask_zero = layer_config.get('mask_zero', False)
                if mask_zero:
                    current_output, mask = self.embedding(current_output, layer_name)
                else:
                    current_output = self.embedding(current_output, layer_name)
                    if isinstance(current_output, tuple):
                        current_output = current_output[0]  # If masking was returned anyway

            elif layer_type == 'SimpleRNN':
                current_output = self.simple_rnn(current_output, layer_name, mask)
                # After SimpleRNN, we no longer need the mask for subsequent layers
                mask = None

            elif layer_type == 'LSTM':
                current_output = self.lstm(current_output, layer_name, mask)
                # After LSTM, we no longer need the mask for subsequent layers
                mask = None

            elif layer_type == 'Dense':
                activation = layer_config.get('activation', 'linear')
                current_output = self.dense(current_output, layer_name, activation)

            elif layer_type == 'Dropout':
                current_output = self._dropout(current_output, layer_name, training)

        return current_output

    def predict_classes(self, X):
        """
        Predict class labels for the input data.

        :param X:
        :return:
        """

        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
