import tensorflow as tf
from tensorflow.contrib.rnn import LayerRNNCell, LSTMStateTuple

from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def normalization(tens,scope=None):
    # https://github.com/philipperemy/tensorflow-multi-dimensional-lstm/blob/master/md_lstm.py
    assert (len(tens.get_shape()) == 2)
    m, v = tf.nn.moments(tens, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'norm'):
        scale = tf.get_variable('scale',
                                shape=[tens.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tens.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tens - m) / tf.sqrt(v + 1e-5)

    return ln_initial * scale + shift


class STLSTMCell(LayerRNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  The default non-peephole implementation is based on:
    https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf
  Felix Gers, Jurgen Schmidhuber, and Fred Cummins.
  "Learning to forget: Continual prediction with LSTM." IET, 850-855, 1999.
  The peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  """

  def __init__(self, num_units,initializer=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True, do_norm=False,
               activation=None, reuse=None, name=None, dtype=None, **kwargs):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: Always True, accepted and returned states are 2-tuples of
        the `c_state` and `h_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().
      When restoring from CudnnLSTM-trained checkpoints, use
      `CudnnCompatibleLSTMCell` instead.
    """
    super(STLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self.do_norm = do_norm
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._initializer = initializers.get(initializer)

    self._state_is_tuple = state_is_tuple
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    self._state_size = (LSTMStateTuple(num_units, num_units))
    self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]  # Number of input channels
    h_depth = self._num_units

    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + 2 * h_depth, 5 * self._num_units],
        initializer=self._initializer)
    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[5 * self._num_units],
        initializer=initializer)
    self.built = True

  def call(self, inputs, state, scope=None):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, must be 2-D, `[batch, input_size]`.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `h_state`.
      do_norm: if True, perform batch normalization
    Returns:
      A tuple containing:
      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is: num_units
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    with tf.variable_scope(scope or type(self).__name__):
        (cs_prev, ct_prev, hj_prev, ht_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
          raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, fs = forget_gate_S, ft = forget_gate_T, o = output_gate, j = new_input
        lstm_matrix = math_ops.matmul(array_ops.concat([inputs, hj_prev, ht_prev], 1), self._kernel)
        if not self.do_norm:
            lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, fs, ft, o, j = array_ops.split(value=lstm_matrix, num_or_size_splits=5, axis=1)
        if self.do_norm:
            i = normalization(i, 'i/')
            fs = normalization(fs, 'fs/')
            ft = normalization(ft, 'ft/')
            o = normalization(o, 'o/')
            j = normalization(j, 'j/')
        # New state
        c = (tf.nn.sigmoid(i) * self._activation(j) +
             tf.nn.sigmoid(fs + self._forget_bias) * cs_prev +
             tf.nn.sigmoid(ft + self._forget_bias) * ct_prev)
        if self.do_norm:
            c = normalization(c,'c/')
        # New hidden state
        h = tf.nn.sigmoid(o) * self._activation(c)

        new_state = (LSTMStateTuple(c, h))
        return h, new_state



# def multi_dimensional_rnn_while_loop(lstm_size, input_data, sh, dims=None, scope_n="layer1"):
#     """Implements naive multi dimension recurrent neural networks
#     @param rnn_size: the hidden units
#     @param input_data: the data to process of shape [batch,h,w,channels]
#     @param sh: [height,width] of the windows
#     @param dims: dimensions to reverse the input data,eg.
#         dims=[False,True,True,False] => true means reverse dimension
#     @param scope_n : the scope
#     returns [batch,h/sh[0],w/sh[1],rnn_size] the output of the lstm
#     """
#
#     with tf.variable_scope("MultiDimensionalLSTMCell-" + scope_n):
#
#         # Create multidimensional cell with selected size
#         cell = STLSTMCell(lstm_size)
#
#         # Get the shape of the input (batch_size, x, y, channels)
#         shape = input_data.get_shape().as_list()
#         batch_size = shape[0]
#         X_dim = shape[1]
#         Y_dim = shape[2]
#         channels = shape[3]
#         # Window size
#         X_win = sh[0]
#         Y_win = sh[1]
#         # Get the runtime batch size
#         batch_size_runtime = tf.shape(input_data)[0]
#
#         # If the input cannot be exactly sampled by the window, we patch it with zeros
#         if X_dim % X_win != 0:
#             # Get offset size
#             offset = tf.zeros([batch_size_runtime, X_win - (X_dim % X_win), Y_dim, channels])
#             # Concatenate X dimension
#             input_data = tf.concat(axis=1, values=[input_data, offset])
#             # Get new shape
#             shape = input_data.get_shape().as_list()
#             # Update shape value
#             X_dim = shape[1]
#
#         # The same but for Y axis
#         if Y_dim % Y_win != 0:
#             # Get offset size
#             offset = tf.zeros([batch_size_runtime, X_dim, Y_win - (Y_dim % Y_win), channels])
#             # Concatenate Y dimension
#             input_data = tf.concat(axis=2, values=[input_data, offset])
#             # Get new shape
#             shape = input_data.get_shape().as_list()
#             # Update shape value
#             Y_dim = shape[2]
#
#         # Get the steps to perform in X and Y axis
#         h, w = int(X_dim / X_win), int(Y_dim / Y_win)
#
#         # Get the number of features (total number of imput values per step)
#         features = Y_win * X_win * channels
#
#         # Reshape input data to a tensor containing the step indexes and features inputs
#         # The batch size is inferred from the tensor size
#         x = tf.reshape(input_data, [batch_size_runtime, h, w, features])
#
#         # Reverse the selected dimensions
#         if dims is not None:
#             assert dims[0] is False and dims[3] is False
#             x = tf.reverse(x, dims)
#
#         # Reorder inputs to (h, w, batch_size, features)
#         x = tf.transpose(x, [1, 2, 0, 3])
#         # Reshape to a one dimensional tensor of (h*w*batch_size , features)
#         x = tf.reshape(x, [-1, features])
#         # Split tensor into h*w tensors of size (batch_size , features)
#         x = tf.split(axis=0, num_or_size_splits=h * w, value=x)
#
#         # Create an input tensor array (literally an array of tensors) to use inside the loop
#         inputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='input_ta')
#         # Unstack the input X in the tensor array
#         inputs_ta = inputs_ta.unstack(x)
#         # Create an input tensor array for the states
#         states_ta = tf.TensorArray(dtype=tf.float32, size=h * w + 1, name='state_ta', clear_after_read=False)
#         # And an other for the output
#         outputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='output_ta')
#
#         # initial cell hidden states
#         # Write to the last position of the array, the LSTMStateTuple filled with zeros
#         states_ta = states_ta.write(h * w, LSTMStateTuple(tf.zeros([batch_size_runtime, rnn_size], tf.float32),
#                                                           tf.zeros([batch_size_runtime, rnn_size], tf.float32)))
#
#         # Function to get the sample skipping one row
#         def get_up(t_, w_):
#             return t_ - tf.constant(w_)
#
#         # Function to get the previous sample
#         def get_last(t_, w_):
#             return t_ - tf.constant(1)
#
#         # Controls the initial index
#         time = tf.constant(0)
#         zero = tf.constant(0)
#
#         # Body of the while loop operation that applies the MD LSTM
#         def body(time_, outputs_ta_, states_ta_):
#
#             # If the current position is less or equal than the width, we are in the first row
#             # and we need to read the zero state we added in row (h*w).
#             # If not, get the sample located at a width distance.
#             state_up = tf.cond(tf.less_equal(time_, tf.constant(w)),
#                                lambda: states_ta_.read(h * w),
#                                lambda: states_ta_.read(get_up(time_, w)))
#
#             # If it is the first step we read the zero state if not we read the inmediate last
#             state_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w))),
#                                  lambda: states_ta_.read(get_last(time_, w)),
#                                  lambda: states_ta_.read(h * w))
#
#             # We build the input state in both dimensions
#             current_state = state_up[0], state_last[0], state_up[1], state_last[1]
#             # Now we calculate the output state and the cell output
#             out, state = cell(inputs_ta.read(time_), current_state)
#             # We write the output to the output tensor array
#             outputs_ta_ = outputs_ta_.write(time_, out)
#             # And save the output state to the state tensor array
#             states_ta_ = states_ta_.write(time_, state)
#
#             # Return outputs and incremented time step
#             return time_ + 1, outputs_ta_, states_ta_
#
#         # Loop output condition. The index, given by the time, should be less than the
#         # total number of steps defined within the image
#         def condition(time_, outputs_ta_, states_ta_):
#             return tf.less(time_, tf.constant(h * w))
#
#         # Run the looped operation
#         result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
#                                                       parallel_iterations=1)
#
#         # Extract the output tensors from the processesed tensor array
#         outputs = outputs_ta.stack()
#         states = states_ta.stack()
#
#         # Reshape outputs to match the shape of the input
#         y = tf.reshape(outputs, [h, w, batch_size_runtime, rnn_size])
#
#         # Reorder te dimensions to match the input
#         y = tf.transpose(y, [2, 0, 1, 3])
#         # Reverse if selected
#         if dims is not None:
#             y = tf.reverse(y, dims)
#
#         # Return the output and the inner states
#         return y, states