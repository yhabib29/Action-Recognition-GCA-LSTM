import tensorflow as tf
import collections
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


def normalization(tens, scope=None):
    # https://github.com/philipperemy/tensorflow-multi-dimensional-lstm/blob/master/md_lstm.py
    assert (len(tens.get_shape()) == 2)
    m, v = tf.nn.moments(tens, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'norm', reuse=tf.AUTO_REUSE):
        scale = tf.get_variable('scale',
                                shape=[tens.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tens.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tens - m) / tf.sqrt(v + 1e-5)

    return ln_initial * scale + shift


# _STLSTMStateTuple = collections.namedtuple("STLSTMStateTuple", ("cs","ct", "hs", "ht"))
# class STLSTMStateTuple(_STLSTMStateTuple):
#   """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
#
#   Stores two elements: `(cs,ct,hs,ht)`, in that order. Where `c` is the hidden state
#   and `h` is the output.
#
#   Only used when `state_is_tuple=True`.
#   """
#   __slots__ = ()
#
#   @property
#   def dtype(self):
#     (cs,ct,hs,ht) = self
#     if cs.dtype != hs.dtype or cs.dtype != ct.dtype or hs.dtype != ht.dtype:
#       raise TypeError("Inconsistent internal state: (%s,%s,%s,%s)" %
#                       (str(cs.dtype),str(ct.dtype),str(hs.dtype),str(ht.dtype)))
#     return cs.dtype


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

    def __init__(self, num_units, initializer=None, input_shape=None,
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
        # self._initializer = initializers.get(initializer)
        self._initializer = initializer

        self._state_is_tuple = state_is_tuple
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

        self._state_size = (LSTMStateTuple(num_units, num_units))
        self._output_size = num_units
        if input_shape is None:
            raise ValueError("Expected inputs_shape to be known")
        else:
            if not self.built:
                self.build(input_shape,name)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape, name=None):
        wname = _WEIGHTS_VARIABLE_NAME
        bname = _BIAS_VARIABLE_NAME
        if not name is None:
            wname += "_" + name
            bname += "_" + name
        if inputs_shape[-1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1].value  # Number of input channels
        h_depth = self._num_units
        wshape = [input_depth + 2 * h_depth, 5 * self._num_units]

        self._kernel = self.add_variable(wname,
            shape=wshape,
            initializer=self._initializer)
        if self.dtype is None:
            initializer = init_ops.zeros_initializer
        else:
            initializer = init_ops.zeros_initializer(dtype=self.dtype)
        self._bias = self.add_variable(bname,
            shape=[5 * self._num_units],
            initializer=initializer)
        self.built = True

    def __call__(self, inputs, state, scope=None):
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
            # input_size = inputs.get_shape()
            # print(inputs)
            # input_size = tf.shape(inputs)
            # if input_size is None:
            #     raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

            # if not self.built:
            #     self.build(input_size)
            (cs_prev, ct_prev, hs_prev, ht_prev) = state
            # print(state)
            # (c_prev, h_prev) = state
            # print(c_prev)
            # print(h_prev)
            # (cs_prev,ct_prev) = c_prev
            # (hs_prev, ht_prev) = h_prev

            # i = input_gate, fs = forget_gate_S, ft = forget_gate_T, o = output_gate, j = new_input
            lstm_matrix = math_ops.matmul(array_ops.concat([inputs, hs_prev, ht_prev], 1), self._kernel)
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
                c = normalization(c, 'c/')
            # New hidden state
            h = tf.nn.sigmoid(o) * self._activation(c)

            new_state = (LSTMStateTuple(c, h))
            return h, new_state


def stlstm_loop(lstm_size, input_data, do_norm=False):
    """https://github.com/philipperemy/tensorflow-multi-dimensional-lstm/blob/master/md_lstm.py
    Implements naive multi dimension recurrent neural networks
    @param lstm_size: the hidden units
    @param input_data: the data to process of shape [batch,frames,joints,channels]
    # @param sh: [height,width] of the windows
    # @param dims: dimensions to reverse the input data,eg.
    #     dims=[False,True,True,False] => true means reverse dimension
    @param scope_n : the scope
    returns [batch,h/sh[0],w/sh[1],lstm_size] the output of the lstm
    """

    with tf.variable_scope("ST-LSTM"):
        # Create multidimensional cell with selected size
        cell = STLSTMCell(lstm_size[0],
                          input_shape=input_data.get_shape(),
                          initializer=tf.truncated_normal_initializer,
                          name="layer1",
                          do_norm=do_norm)

        cell2 = STLSTMCell(lstm_size[1],
                           input_shape=tf.TensorShape([lstm_size[0]]),
                           initializer=tf.truncated_normal_initializer,
                           name="layer2",
                           do_norm=do_norm)

        # Get the shape of the input (batch_size, x, y, channels)
        # shape = input_data.get_shape().as_list()
        shape = tf.shape(input_data)
        batch_size = shape[0]
        T_dim = shape[1]
        S_dim = shape[2]
        channels = shape[3]
        #         # Window size
        #         X_win = sh[0]
        #         Y_win = sh[1]
        # Get the runtime batch size
        # ! TODO: Check difference with batch_size
        # batch_size_runtime = tf.shape(input_data)[0]
        #
        # Get the number of features (total number of input values per step)
        # features = S_dim * channels

        # The batch size is inferred from the tensor size
        x = tf.reshape(input_data, [batch_size, T_dim, S_dim, channels])

        # Reorder inputs to (t, s, batch_size, features) - t=T_dim, s=S_dim
        x = tf.transpose(x, [1, 2, 0, 3])
        # Reshape to a one dimensional tensor of (t*s*batch_size , features)
        x = tf.reshape(x, [-1, batch_size, channels])
        # Split tensor into t*s tensors of size (batch_size , features)
        # x = tf.split(axis=0, num_or_size_splits=T_dim*S_dim, value=x)

        # Create an input tensor array (literally an array of tensors) to use inside the loop
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=T_dim * S_dim, name='input_array', dynamic_size=True,
                                   infer_shape=False)
        inputs_ta = inputs_ta.unstack(x)
        states_ta = tf.TensorArray(dtype=tf.float32, size=T_dim * S_dim + 1, name='state_array_1', clear_after_read=False)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=T_dim * S_dim, name='output_array_1')
        states_ta2 = tf.TensorArray(dtype=tf.float32, size=T_dim * S_dim + 1, name='state_array2', clear_after_read=False)
        outputs_ta2 = tf.TensorArray(dtype=tf.float32, size=T_dim * S_dim, name='output_array2')

        # initial cell hidden states
        # Write to the last position of the array, the LSTMStateTuple filled with zeros
        states_ta = states_ta.write(T_dim * S_dim, LSTMStateTuple(tf.zeros([batch_size, lstm_size[0]], tf.float32),
                                                      tf.zeros([batch_size, lstm_size[0]], tf.float32)))
        states_ta2 = states_ta2.write(T_dim * S_dim, LSTMStateTuple(tf.zeros([batch_size, lstm_size[1]], tf.float32),
                                                                  tf.zeros([batch_size, lstm_size[1]], tf.float32)))

        # Function to get the previous joints id (cs_prev,hs_prev)
        def get_prevS(t_, w_=1):
            # ! TODO: Handle the first element !
            # return S_dim + tf.mod(t_, S_dim) - tf.constant(w_)
            return t_ - tf.constant(w_)

        # Function to get the previous time id (ct_prev,ht_prev)
        def get_prevT(t_, w_=S_dim):
            # return tf.mod(t_, w_)  # - tf.constant(w_)
            return t_ - w_

        # Controls the initial index
        index = tf.constant(0)
        zero = tf.constant(0)

        # Body of the while loop operation that applies the MD LSTM
        def body(id_, outputs_ta_, states_ta_, outputs_ta2_, states_ta2_):
            # If the current position is less or equal than the width, we are in the first row
            # so we read the zero state we added before.
            # If not, get the sample located at a width distance.
            prevstate_T = tf.cond(tf.less_equal(id_, S_dim),
                                  lambda: states_ta_.read(T_dim * S_dim),  # first row = zero state
                                  lambda: states_ta_.read(get_prevT(id_, S_dim)))  # other rows = previous time id (t-1)

            # If it is the first step we read the zero state if not we read the inmediate last
            prevstate_S = tf.cond(tf.less(zero, tf.mod(id_, S_dim)),
                                  lambda: states_ta_.read(get_prevS(id_)),  # get previous joint state id (j-1)
                                  lambda: states_ta_.read(T_dim * S_dim))  # first joint - get zero state !TODO: use the good joint

            # We build the input state in both dimensions
            current_state = prevstate_S[0], prevstate_T[0], prevstate_S[1], prevstate_T[1]
            # Now we calculate the hidden state and the new cell state
            out, state = cell(inputs_ta.read(id_), current_state)
            # We write the output to the output tensor array
            outputs_ta_ = outputs_ta_.write(id_, out)
            # And save the output state to the state tensor array
            states_ta_ = states_ta_.write(id_, state)

            #----------------
            # LSTM 2nd layer
            # Process prevStates for the 2nd LSTM layer
            prevstate_T2 = tf.cond(tf.less_equal(id_, S_dim),
                                  lambda: states_ta2_.read(T_dim * S_dim),  # first row = zero state
                                  lambda: states_ta2_.read(get_prevT(id_, S_dim)))  # other rows = previous time id (t-1)

            # If it is the first step we read the zero state if not we read the inmediate last
            prevstate_S2 = tf.cond(tf.less(zero, tf.mod(id_, S_dim)),
                                  lambda: states_ta2_.read(get_prevS(id_)),  # get previous joint state id (j-1)
                                  lambda: states_ta2_.read(T_dim * S_dim))  # first joint - get zero state !TODO: use the good joint
            # Process cureent state and then the new state
            current_state2 = prevstate_S2[0], prevstate_T2[0], prevstate_S2[1], prevstate_T2[1]
            out2, state2 = cell2(out, current_state2)
            outputs_ta2_ = outputs_ta2_.write(id_, out2)
            states_ta2_ = states_ta2_.write(id_, state2)

            # Return outputs and incremented time step
            return id_ + 1, outputs_ta_, states_ta_, outputs_ta2_, states_ta2_

        # # Body of the while loop operation that applies the MD LSTM
        # def body2(id_, outputs_ta2_, states_ta2_):
        #     prevstate_T2 = tf.cond(tf.less_equal(id_, S_dim),
        #                           lambda: states_ta2_.read(T_dim * S_dim),  # first row = zero state
        #                           lambda: states_ta2_.read(get_prevT(id_, S_dim)))  # other rows = previous time id (t-1)
        #
        #     # If it is the first step we read the zero state if not we read the inmediate last
        #     prevstate_S2 = tf.cond(tf.less(zero, tf.mod(id_, S_dim)),
        #                           lambda: states_ta2_.read(get_prevS(id_)),  # get previous joint state id (j-1)
        #                           lambda: states_ta2_.read(T_dim * S_dim))  # first joint - get zero state !TODO: use the good joint
        #     # Process cureent state and then the new state
        #     current_state2 = prevstate_S2[0], prevstate_T2[0], prevstate_S2[1], prevstate_T2[1]
        #     out2, state2 = cell2(outputs_ta.read(id_), current_state2)
        #     outputs_ta2_ = outputs_ta2_.write(id_, out2)
        #     states_ta2_ = states_ta2_.write(id_, state2)
        #
        #     # Return outputs and incremented time step
        #     return id_ + 1, outputs_ta_, states_ta_, outputs_ta2_, states_ta2_

        # Loop output condition. The index, given by the time, should be less than the
        # total number of steps defined within the image
        def condition(id_, outputs_ta_, states_ta_, outputs_ta2_, states_ta2_):
            return tf.less(id_, T_dim * S_dim)

        # Run the looped operation
        result, _, _, outputs_ta2, states_ta2 = tf.while_loop(condition, body,
                                                      [index, outputs_ta, states_ta, outputs_ta2, states_ta2],
                                                      parallel_iterations=1)

        # Extract the output tensors from the processesed tensor array
        outputs = outputs_ta2.stack()
        states = states_ta2.stack()

        # Reshape outputs to match the shape of the input
        # y = tf.reshape(outputs, [T_dim, S_dim, batch_size, lstm_size[0]])   # For outputs_ta
        # states = tf.reshape(states, [T_dim,S_dim,batch_size,2,lstm_size[0]])
        y = tf.reshape(outputs, [T_dim, S_dim, batch_size, lstm_size[1]])

        # Reorder te dimensions to match the input
        y = tf.transpose(y, [2, 0, 1, 3])
        #         # Reverse if selected
        #         if dims is not None:
        #             y = tf.reverse(y, dims)
        #
        # Return the output and the inner states
        return y, states