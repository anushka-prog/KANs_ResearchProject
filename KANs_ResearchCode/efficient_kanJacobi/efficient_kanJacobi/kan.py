import numpy as np
import keras
from keras import ops
from keras import backend
from keras.src import initializers
from keras.src.layers import Layer, Dropout, LayerNormalization

'''
This code has been modified by: UNI- ap4617
This code defines two custom Keras layers, `GridInitializer` and `KANLinear`. The `GridInitializer` is a custom initializer 
designed to create a grid of values based on specified grid range, grid size, and spline order. It generates a 1D grid, 
which is then replicated to match the required shape for initialization. The `KANLinear` layer is a specialized linear 
layer that includes several advanced features, such as grid-based initialization, Jacobi polynomial and B-spline computations, 
and the option to apply layer normalization. The layer supports customizable activation functions, dropout regularization, 
and bias terms. It can optionally use Jacobi polynomials or B-splines for modeling complex relationships in the data, 
and it incorporates layer normalization for improved training stability. The layer is designed to handle both traditional 
and more complex sequential data processing tasks with flexible configuration options. It also includes serialization 
functionality to save and reload the model configuration.

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
        active_jacobi=False,
        jacobi_order=3,
        alpha=0.5,
        beta=0.5,
        **kwargs
    ):
        # super(KANLinear, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation_name = base_activation
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout
        self.active_jacobi = active_jacobi
        self.jacobi_order = jacobi_order
        self.alpha = alpha
        self.beta = beta
      


        # Remove custom arguments from kwargs
        kwargs.pop("active_jacobi", None)
        kwargs.pop("jacobi_order", None)
        kwargs.pop("alpha", None)
        kwargs.pop("beta", None)

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
        if self.active_jacobi:
            self.jacobi_weight = self.add_weight(
                name="jacobi_weight",
                shape=[self.jacobi_order * self.in_features, self.units],
                initializer="glorot_uniform",
                trainable = True,
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

        if self.active_jacobi:
            jacobi_output = ops.matmul(self.jacobi_polynomials(x_2d), self.jacobi_weight)
            output_2d = self.dropout(base_output, training=training) + self.dropout(jacobi_output, training=training)
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

    def jacobi_polynomials(self, x):
        import tensorflow as tf
        x_expanded = ops.expand_dims(x, -1)
        polynomials = [self._jacobi_polynomial(x_expanded, n) for n in range(self.jacobi_order)]
        return tf.reshape(tf.concat(polynomials, axis=-1), [tf.shape(x)[0], -1])

    def _jacobi_polynomial(self, x, n):
        if n == 0:
            return ops.ones_like(x)
        if n == 1:
            return 0.5 * ((2 * (self.alpha + 1)) + (self.alpha + self.beta + 2) * (x - 1))
        
        P_n_1 = self._jacobi_polynomial(x, n - 1)
        P_n_2 = self._jacobi_polynomial(x, n - 2)

        
        a1 = 2 * (n + self.alpha) * (n + self.beta) * (2 * n + self.alpha + self.beta)
        a2 = (2 * n + self.alpha + self.beta - 1) * (self.alpha ** 2 - self.beta ** 2)
        a3 = (2 * n + self.alpha + self.beta - 2) * (2 * n + self.alpha + self.beta)
        a4 = 2 * (n + self.alpha - 1) * (n + self.beta - 1) * (2 * n + self.alpha + self.beta)
        return ((a2 + a3 * x) * P_n_1 - a4 * P_n_2) / a1

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
            'active_jacobi': self.active_jacobi,
            'jacobi_order': self.jacobi_order,
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return config