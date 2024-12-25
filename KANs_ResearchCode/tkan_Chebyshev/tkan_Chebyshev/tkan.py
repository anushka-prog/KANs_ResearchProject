import time
import keras
from keras import ops
from keras import random
from keras import activations
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import InputSpec, Layer, RNN, Dense

'''
This code has been referenced from https://github.com/remigenet/TKAN and owned by: 
@article{genet2024tkan,
  title={Tkan: Temporal kolmogorov-arnold networks},
  author={Genet, Remi and Inzirillo, Hugo},
  journal={arXiv preprint arXiv:2405.07344},
  year={2024}
}
'''

'''
This code has been modified by: UNI- ap4617
This code defines a custom Keras layer called `TKANCell`, which is part of a Temporal Kolmogorov-Arnold Network (TKAN). 
The `TKANCell` layer is an advanced recurrent cell that integrates both traditional LSTM-like gating mechanisms and 
more sophisticated temporal operations, including Chebyshev polynomial expansions for improved modeling of sequential data. 
The layer uses sub-layers defined by a modular configuration, allowing for flexible sub-structures like Chebyshev polynomials or 
dense layers. Additionally, it supports dropout and recurrent dropout for regularization, and customizable initialization, 
activation, and regularization functions. The `TKAN` class builds on this `TKANCell` by wrapping it in an RNN layer, allowing 
for easy integration into larger models for sequential tasks. The model also includes functionality for serializing and deserializing 
the custom layers, ensuring compatibility with Keras' save/load mechanisms. Overall, the code is designed to enable efficient handling 
of complex temporal dependencies in sequential data.

'''

from efficient_kanChebyshev import KANLinear

def get_backend():
    import os
    return os.environ.get('KERAS_BACKEND', 'tensorflow')

@keras.utils.register_keras_serializable(package="tkan", name="TKANCell")
class TKANCell(Layer):
    """Cell class for the TKAN layer."""
    def __init__(
        self,
        units,
        sub_kan_configs=None,
        sub_kan_output_dim=None,
        sub_kan_input_dim=None,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        active_chebyshev=False,  
        cheby_order=3,       
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.sub_kan_configs = sub_kan_configs or [None]
        self.sub_kan_output_dim = sub_kan_output_dim
        self.sub_kan_input_dim = sub_kan_input_dim
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed if seed is not None else int(time.time())
        self.active_chebyshev = active_chebyshev  # Integrate Chebyshev support
        self.cheby_order = cheby_order
        self.state_size = [units, units] + [1 for _ in self.sub_kan_configs]
        self.output_size = units
        self.backend = get_backend()

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if self.sub_kan_input_dim is None:
            self.sub_kan_input_dim = input_dim
        if self.sub_kan_output_dim is None:
            self.sub_kan_output_dim = input_dim

        if self.active_chebyshev:
           
            self.cheby_weight = self.add_weight(
                name="cheby_weight",
                shape=[self.units],  
                initializer="glorot_uniform",
                trainable=True
            )

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return ops.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.get("ones")((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units,), *args, **kwargs),
                    ])
                bias_value = bias_initializer(None)
            else:
                bias_value = self.bias_initializer((self.units * 3,))
            self.bias = self.add_weight(
                shape=(self.units * 3,),
                name="bias",
                initializer=lambda shape, dtype=None: bias_value,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.tkan_sub_layers = []
        for config in self.sub_kan_configs:
            if config is None:
                layer = KANLinear(
                    self.sub_kan_output_dim,
                    use_layernorm=True,
                    active_chebyshev=self.active_chebyshev,   
                    cheby_order=self.cheby_order
                )
            elif isinstance(config, (int, float)):
                layer = KANLinear(
                    self.sub_kan_output_dim,
                    spline_order=config,
                    use_layernorm=True,
                    active_chebyshev=self.active_chebyshev,    
                    cheby_order=self.cheby_order
                )
            elif isinstance(config, dict):
                layer = KANLinear(
                    self.sub_kan_output_dim,
                    **config,
                    use_layernorm=True,
                    active_chebyshev=self.active_chebyshev,   
                    cheby_order=self.cheby_order
                )
            else:
                # If config is a string activation, fallback to Dense
                layer = Dense(self.sub_kan_output_dim, activation=config)
            layer.build((input_shape[0], self.sub_kan_input_dim))
            self.tkan_sub_layers.append(layer)

        self.sub_tkan_kernel = self.add_weight(
            shape=(len(self.tkan_sub_layers), self.sub_kan_output_dim * 2),
            name="sub_tkan_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        self.sub_tkan_recurrent_kernel_inputs = self.add_weight(
            shape=(len(self.tkan_sub_layers), input_dim, self.sub_kan_input_dim),
            name="sub_tkan_recurrent_kernel_inputs",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        self.sub_tkan_recurrent_kernel_states = self.add_weight(
            shape=(len(self.tkan_sub_layers), self.sub_kan_output_dim, self.sub_kan_input_dim),
            name="sub_tkan_recurrent_kernel_states",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        self.aggregated_weight = self.add_weight(
            shape=(len(self.tkan_sub_layers) * self.sub_kan_output_dim, self.units),
            name="aggregated_weight",
            initializer="glorot_uniform",
        )
        self.aggregated_bias = self.add_weight(
            shape=(self.units,),
            name="aggregated_bias",
            initializer="zeros",
        )

        self.built = True

    def _generate_dropout_mask(self, inputs):
        if 0 < self.dropout < 1:
            seed_generator = random.SeedGenerator(self.seed)
            return random.dropout(
                ops.ones_like(inputs),
                self.dropout,
                seed=seed_generator
            )
        return None

    def _generate_recurrent_dropout_mask(self, states):
        if 0 < self.recurrent_dropout < 1:
            seed_generator = random.SeedGenerator(self.seed + 1)
            return random.dropout(
                ops.ones_like(states),
                self.recurrent_dropout,
                seed=seed_generator
            )
        return None

    def call(self, inputs, states, training=None):
        if self.backend == 'tensorflow':
            return self._call_tensorflow(inputs, states, training)
        else:
            return self._call_generic(inputs, states, training)

    def _call_tensorflow(self, inputs, states, training=False):
        import tensorflow as tf
        h_tm1 = states[0] 
        c_tm1 = states[1]
        sub_states = states[2:]
    
        batch_size = tf.shape(inputs)[0]
        if training:
            self.seed = (self.seed + 1) % (2**32 - 1) 
            if self.dropout > 0.0:
                inputs = inputs * self._generate_dropout_mask(inputs)
            if self.recurrent_dropout > 0.0:
                h_tm1 = h_tm1 * self._generate_recurrent_dropout_mask(h_tm1)
    
        if self.use_bias:
            gates = tf.matmul(inputs, self.kernel) + tf.matmul(h_tm1, self.recurrent_kernel) + self.bias
        else:
            gates = tf.matmul(inputs, self.kernel) + tf.matmul(h_tm1, self.recurrent_kernel)

        # Split for LSTM-like structure
        i, f, c = tf.split(self.recurrent_activation(gates), 3, axis=1)
        c = f * c_tm1 + i * self.activation(c)

        sub_outputs = tf.TensorArray(dtype=tf.float32, size=len(self.tkan_sub_layers))
        new_sub_states = tf.TensorArray(dtype=tf.float32, size=len(self.tkan_sub_layers))

        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            sub_kernel_x = self.sub_tkan_recurrent_kernel_inputs[idx]
            sub_kernel_h = self.sub_tkan_recurrent_kernel_states[idx]
            agg_input = inputs @ sub_kernel_x + sub_state @ sub_kernel_h
            sub_output = sub_layer(agg_input)
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = tf.split(self.sub_tkan_kernel[idx], 2, axis=0)
            new_sub_state = sub_recurrent_kernel_h * sub_output + sub_state * sub_recurrent_kernel_x

            sub_outputs = sub_outputs.write(idx, sub_output)
            new_sub_states = new_sub_states.write(idx, new_sub_state)

        sub_outputs = sub_outputs.stack()
        aggregated_sub_output = tf.reshape(sub_outputs, (batch_size, -1))
        aggregated_input = tf.matmul(aggregated_sub_output, self.aggregated_weight) + self.aggregated_bias
        o = self.recurrent_activation(aggregated_input)
        h = o * self.activation(c)

        new_states_list = [h, c] + tf.unstack(new_sub_states.stack())
        return h, new_states_list

    def _call_generic(self, inputs, states, training=None):
        h_tm1 = states[0]
        c_tm1 = states[1]
        sub_states = states[2:]

        if training:
            self.seed = (self.seed + 1) % (2**32 - 1)
            dp_mask = self._generate_dropout_mask(inputs)
            rec_dp_mask = self._generate_recurrent_dropout_mask(h_tm1)
            if dp_mask is not None:
                inputs *= dp_mask
            if rec_dp_mask is not None:
                h_tm1 *= rec_dp_mask

        if self.use_bias:
            gates = ops.matmul(inputs, self.kernel) + ops.matmul(h_tm1, self.recurrent_kernel) + self.bias
        else:
            gates = ops.matmul(inputs, self.kernel) + ops.matmul(h_tm1, self.recurrent_kernel)
        
        i, f, c = ops.split(self.recurrent_activation(gates), 3, axis=-1)
        c = f * c_tm1 + i * self.activation(c)

        sub_outputs = []
        new_sub_states = []

        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            sub_kernel_x = self.sub_tkan_recurrent_kernel_inputs[idx]
            sub_kernel_h = self.sub_tkan_recurrent_kernel_states[idx]
            agg_input = ops.matmul(inputs, sub_kernel_x) + ops.matmul(sub_state, sub_kernel_h)
            sub_output = sub_layer(agg_input)
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = ops.split(self.sub_tkan_kernel[idx], 2, axis=0)
            new_sub_state = sub_recurrent_kernel_h * sub_output + sub_state * sub_recurrent_kernel_x

            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)

        aggregated_sub_output = ops.concatenate(sub_outputs, axis=-1)
        aggregated_input = ops.dot(aggregated_sub_output, self.aggregated_weight) + self.aggregated_bias
        o = self.recurrent_activation(aggregated_input)
        h = o * self.activation(c)

        return h, [h, c] + new_sub_states

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sub_kan_configs": self.sub_kan_configs,
            "sub_kan_output_dim": self.sub_kan_output_dim,
            "sub_kan_input_dim": self.sub_kan_input_dim,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "active_chebyshev": self.active_chebyshev,  # Include Chebyshev parameters
            "cheby_order": self.cheby_order,
        })
        return config

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or self.compute_dtype
        return [
            ops.zeros((batch_size, self.units), dtype=dtype),
            ops.zeros((batch_size, self.units), dtype=dtype)
        ] + [ops.zeros((batch_size, self.sub_kan_output_dim), dtype=dtype) for _ in range(len(self.tkan_sub_layers))]


@keras.utils.register_keras_serializable(package="tkan", name="TKAN")
class TKAN(RNN):
    def __init__(
        self,
        units,
        sub_kan_configs=None,
        sub_kan_output_dim=None,
        sub_kan_input_dim=None,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        seed=None,
        active_chebyshev=False,  
        cheby_order=3,        
        **kwargs,
    ):
        cell = TKANCell(
            units,
            sub_kan_configs=sub_kan_configs,
            sub_kan_output_dim=sub_kan_output_dim,
            sub_kan_input_dim=sub_kan_input_dim,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
            active_chebyshev=active_chebyshev,   # Pass Chebyshev parameters
            cheby_order=cheby_order
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def inner_loop(self, sequences, initial_state, mask, training=False):
        if isinstance(mask, (list, tuple)):
            mask = mask[0]
        return super().inner_loop(
            sequences, initial_state, mask=mask, training=training
        )

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences, mask=mask, training=training, initial_state=initial_state
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def sub_kan_configs(self):
        return self.cell.sub_kan_configs

    @property
    def sub_kan_output_dim(self):
        return self.cell.sub_kan_output_dim

    @property
    def sub_kan_input_dim(self):
        return self.cell.sub_kan_input_dim
    
    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sub_kan_configs": self.sub_kan_configs,
            "sub_kan_output_dim": self.sub_kan_output_dim,
            "sub_kan_input_dim": self.sub_kan_input_dim,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "active_chebyshev": self.cell.active_chebyshev,  
            "cheby_order": self.cell.cheby_order      
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)
