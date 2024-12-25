import numpy as np
import keras
from keras import ops
from keras import backend
from keras.src import initializers
from keras.src.layers import Layer, Dropout, LayerNormalization
import tensorflow as tf
from keras.layers import RNN

'''
This code has been modified by UNI: ap4617 
This code defines a Keras layer class `KANLinear` that incorporates Chebyshev polynomials and B-spline functions for 
flexible function approximation in neural networks. The `GridInitializer` class initializes a structured grid for spline computation, 
while the `KANLinear` class manages the core computations, including weight initialization, forward pass logic, and polynomial basis construction. 
The `build` method sets up trainable weights, including base weights, spline weights, and optional Chebyshev weights. The `call` method handles 
the forward pass, applying layer normalization, base activation, and computing outputs using either B-splines or Chebyshev polynomials. 
The implementation supports flexible configurations like using bias, dropout, and custom activations, 
making the layer adaptable for complex learning tasks.
'''

@keras.utils.register_keras_serializable(package="keras_efficient_kan", name="GridInitializer")
class GridInitializer(initializers.Initializer):
    def __init__(self, grid_range, grid_size, spline_order):
        self.grid_range = grid_range
        self.grid_size = grid_size
        self.spline_order = spline_order

    def __call__(self, shape, dtype=None):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        start = -self.spline_order * h + self.grid_range[0]
        stop = (self.grid_size + self.spline_order) * h + self.grid_range[0]
        num = self.grid_size + 2 * self.spline_order + 1
        
        grid = np.linspace(start, stop, num, dtype=np.float32)
        grid = np.tile(grid, (shape[1], 1))
        grid = np.expand_dims(grid, 0)
        return ops.convert_to_tensor(grid, dtype=dtype)

    def get_config(self):
        return {
            "grid_range": self.grid_range,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order
        }

@keras.utils.register_keras_serializable(package="keras_efficient_kan", name="KANLinear")
class KANLinear(Layer):
    def __init__(
        self,
        units,
        grid_size=3,
        spline_order=3,
        base_activation='relu',
        grid_range=[-1, 1],
        dropout=0.,
        use_bias=True,
        use_layernorm=True,
        alpha=0.5,
        beta=0.5,
        active_chebyshev=False,
        cheby_order=3,
        **kwargs
    ):
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation_name = base_activation
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout
        self.alpha = alpha
        self.beta = beta
        self.active_chebyshev = active_chebyshev
        self.cheby_order = cheby_order

       
        kwargs.pop("active_chebyshev", None)
        kwargs.pop("active_chebyshev", None)

        # super(KANLinear, self).__init__(**kwargs)
        super(KANLinear, self).__init__(**kwargs)

        self.dropout = Dropout(self.dropout_rate)
        if self.use_layernorm:
            self.layer_norm = LayerNormalization(axis=-1)
        else:
            self.layer_norm = None
        self.in_features = None

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        dtype = backend.floatx()

        # Grid for splines
        self.grid = self.add_weight(
            name="grid",
            shape=[1, self.in_features, self.grid_size + 2 * self.spline_order + 1],
            initializer=GridInitializer(self.grid_range, self.grid_size, self.spline_order),
            trainable=False,
            dtype=dtype
        )

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=[self.in_features, self.units],
            initializer='glorot_uniform',
            dtype=dtype
        )
        if self.use_bias:
            self.base_bias = self.add_weight(
                name="base_bias",
                shape=[self.units],
                initializer="zeros",
                dtype=dtype
            )

        # Depending on which basis we use

        if self.active_chebyshev:

            self.cheby_weight = self.add_weight(
                name="cheby_weight",
                shape=[(self.cheby_order + 1) * self.in_features, self.units],
                initializer='glorot_uniform',
                trainable=True,
                dtype=dtype
            )
        else:
            self.spline_weight = self.add_weight(
                name="spline_weight",
                shape=[self.in_features * (self.grid_size + self.spline_order), self.units],
                initializer='glorot_uniform',
                dtype=dtype
            )

        if self.use_layernorm:
            self.layer_norm.build(input_shape)
        
        self.built = True

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        x = ops.cast(x, self.dtype)
        x_2d = ops.reshape(x, [-1, self.in_features])

        if self.use_layernorm:
            x_2d = self.layer_norm(x_2d)

        base_activation = getattr(ops, self.base_activation_name)
        base_output = ops.matmul(base_activation(x_2d), self.base_weight)
        if self.use_bias:
            base_output = ops.add(base_output, self.base_bias)
        if self.active_chebyshev:
            cheby_output = ops.matmul(self.chebyshev_polynomials(x_2d), self.cheby_weight)
            output_2d = self.dropout(base_output, training=training) + self.dropout(cheby_output, training=training)
        else:
            spline_output = ops.matmul(self.b_splines(x_2d), self.spline_weight)
            output_2d = self.dropout(base_output, training=training) + self.dropout(spline_output, training=training)

        new_shape = tuple(input_shape[:-1]) + (self.units,)
        return ops.reshape(output_2d, new_shape)

    def b_splines(self, x):
        x_expanded = ops.expand_dims(x, -1)
        bases = ops.cast((x_expanded >= self.grid[..., :-1]) & (x_expanded < self.grid[..., 1:]), self.dtype)
        
        for k in range(1, self.spline_order + 1):
            left_denominator = self.grid[..., k:-1] - self.grid[..., :-(k + 1)]
            right_denominator = self.grid[..., k + 1:] - self.grid[..., 1:-k]

            left = (x_expanded - self.grid[..., :-(k + 1)]) / left_denominator
            right = (self.grid[..., k + 1:] - x_expanded) / right_denominator
            bases = left * bases[..., :-1] + right * bases[..., 1:]
    
        return ops.reshape(bases, [ops.shape(x)[0], -1])

   

    def chebyshev_polynomials(self, x):
        """
        Compute Chebyshev polynomials up to the specified order for the input tensor x.
        """
        # Clip input x to ensure values are in the range [-1, 1]
        x_clipped = tf.clip_by_value(x, -1.0, 1.0)

        # Determine batch size symbolically
        batch_size = tf.shape(x_clipped)[0]
        in_features = self.in_features
        cheby_order = self.cheby_order
        dtype = self.dtype

        # Initialize a 3D tensor to store Chebyshev polynomials
        cheby = tf.ones((batch_size, in_features, cheby_order + 1), dtype=dtype)

        # Prepare grid indices for batch and feature dimensions
        batch_indices = tf.range(batch_size)
        feature_indices = tf.range(in_features)
        batch_grid, feature_grid = tf.meshgrid(batch_indices, feature_indices, indexing='ij')

        # Compute U_1(x) = 2x if needed
        if cheby_order > 0:
            layer_index = tf.fill([batch_size, in_features], 1)
            update_indices = tf.stack([batch_grid, feature_grid, layer_index], axis=-1)
            u1_values = 2.0 * x_clipped
            cheby = tf.tensor_scatter_nd_update(cheby, update_indices, u1_values)

        # Define a function for computing higher-order polynomials
        def compute_next_chebyshev_polynomial(i, arr):
            layer_index = tf.fill([batch_size, in_features], i)
            update_indices = tf.stack([batch_grid, feature_grid, layer_index], axis=-1)
            u_prev = arr[..., i - 1]
            u_prev_prev = arr[..., i - 2]
            new_values = 2.0 * x_clipped * u_prev - u_prev_prev
            return i + 1, tf.tensor_scatter_nd_update(arr, update_indices, new_values)

        # Compute all Chebyshev polynomials up to the specified order
        i = tf.constant(2)
        cond = lambda i, _: tf.less_equal(i, cheby_order)
        _, cheby = tf.while_loop(cond, compute_next_chebyshev_polynomial, [i, cheby])

        # Reshape the final array to (batch_size, -1)
        return tf.reshape(cheby, (batch_size, -1))



    def get_config(self):
        config = super(KANLinear, self).get_config()
        config.update({
            'units': self.units,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'base_activation': self.base_activation_name,
            'grid_range': self.grid_range,
            'dropout': self.dropout_rate,
            'use_bias': self.use_bias,
            'use_layernorm': self.use_layernorm,
            'alpha': self.alpha,
            'beta': self.beta,
            'active_chebyshev': self.active_chebyshev,
            'cheby_order': self.cheby_order
        })
        return config
