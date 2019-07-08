import tensorflow as tf
import collections
from tensorflow.contrib.rnn import LayerRNNCell, LSTMStateTuple
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import _CudnnRNN

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
# JOINTS_ORDER = [1, 2, 3, 2, 4, 5, 6, 5, 4, 2, 7, 8, 9, 8, 7, 2, 1,
# 10, 11, 12, 13, 12, 11, 10, 14, 15, 16, 15, 14, 10,1]
# Joint indices - 1 (start from 0)
JOINTS_ORDER = [0, 1, 2, 1, 3, 4, 5, 4, 3, 1, 6, 7, 8, 7, 6, 1, 0, 9, 10, 11, 12, 11, 10, 9, 13, 14, 15, 14, 13, 9, 0]
OUT_DIM1 = len(JOINTS_ORDER)


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
    """ST-LSTM cell adapted from Basic LSTM cell.
    """

    def __init__(self, num_units, initializer=None, input_shape=None,
                 forget_bias=1.0, trainable=True, do_norm=False, useDropout=False,
                 activation=None, reuse=None, name=None, dtype=None, **kwargs):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
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
        super(STLSTMCell, self).__init__(_reuse=reuse, name=name, trainable=trainable,
                                         dtype=dtype, **kwargs)
        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn("%s: Note that this cell is not optimized for performance. "
                         "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                         "performance on GPU.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        self.do_norm = do_norm
        self.useDropout = useDropout
        self._num_units = num_units
        self._forget_bias = forget_bias
        # self._initializer = initializers.get(initializer)
        self._initializer = initializer

        # self._state_is_tuple = state_is_tuple
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
                self.build(input_shape)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        if inputs_shape[-1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1].value  # Number of input channels
        h_depth = self._num_units
        wshape = [5 * self._num_units, input_depth + 2 * h_depth]

        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME,
                                         shape=wshape,
                                         initializer=self._initializer)
        if self.dtype is None:
            initializer = init_ops.zeros_initializer
        else:
            initializer = init_ops.zeros_initializer(dtype=self.dtype)
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME,
                                       shape=[5 * self._num_units],
                                       initializer=initializer)
        self.built = True

    def __call__(self, inputs, state, informativeness=None, scope=None):
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
            (cs_prev, ct_prev, hs_prev, ht_prev) = state

            # Dropout
            if self.useDropout:
                inputs = tf.nn.dropout(inputs, ratio=0.33)

            # i = input_gate, fs = forget_gate_S, ft = forget_gate_T, o = output_gate, j = new_input
            lstm_matrix = math_ops.matmul(array_ops.concat([inputs, hs_prev, ht_prev], 1),
                                          self._kernel, transpose_b=True)
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
            if informativeness is None:
                c = (tf.nn.sigmoid(i) * self._activation(j) +
                     tf.nn.sigmoid(fs + self._forget_bias) * cs_prev +
                     tf.nn.sigmoid(ft + self._forget_bias) * ct_prev)
            else:
                r = tf.nn.sigmoid(informativeness)
                c = (r * tf.nn.sigmoid(i) * self._activation(j) +
                     (1 - r) * tf.nn.sigmoid(fs + self._forget_bias) * cs_prev +
                     (1 - r) * tf.nn.sigmoid(ft + self._forget_bias) * ct_prev)
            if self.do_norm:
                c = normalization(c, 'c/')
            # New hidden state
            h = tf.nn.sigmoid(o) * self._activation(c)

            new_state = (LSTMStateTuple(c, h))
            return h, new_state


class GCACell():  # LayerRNNCell
    # Static weights
    static_weights = False

    def initialize_static_weights(self):
        with tf.variable_scope(type(self).__name__, reuse=tf.AUTO_REUSE):
            # TODO: Hyperparameters hyperX to set
            # Initialize We1
            hyperX = (self.input_size + self.output_size) // 2
            wshape1 = [self.input_size, hyperX]  # [d1,X]
            wshape2 = [hyperX, 2 * self.output_size]  # [X,d2]
            # Initialize We2
            self.We1 = tf.get_variable("We1", wshape1, tf.float32,
                                       tf.truncated_normal_initializer(),
                                       trainable=True)
            self.We2 = tf.get_variable("We2", wshape2, tf.float32,
                                       tf.truncated_normal_initializer(),
                                       trainable=True)
            self.static_weights = True

    def __init__(self, lstm_sizes, iteration, initializer=None, activation=None,
                 reuse=None, name="GCA", dtype=tf.float32, **kwargs):

        # super(GCACell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        if name == "GCA":
            self.name = name + str(iteration)
        else:
            self.name = name
        self.dtype = dtype
        self.iteration = iteration
        if len(lstm_sizes) < 2:
            raise ValueError("Expected lstm_sizes to be known, saw shape: %s"
                             % str(lstm_sizes))
        self.input_size = lstm_sizes[0]  # Number of input channels
        self.output_size = lstm_sizes[1]  # Number of OUTPUT channels
        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)
        # self.do_norm = do_norm
        # self._forget_bias = forget_bias
        # self._initializer = initializers.get(initializer)
        self._initializer = initializer
        self.built = False
        # Global Context
        self.context = tf.zeros([1, self.input_size])
        self.prevcontext = tf.zeros([1, self.input_size])

        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        # self._state_size =
        # self._output_size = num_units
        if not self.static_weights:
            self.initialize_static_weights()
        if not self.built:
            self.build(iteration)

    def set_prevcontext(self, prevctx):
        self.prevcontext = tf.reshape(prevctx, [1, self.input_size])

    def build(self, iteration=None):
        # Initializer WF_n
        wname = _WEIGHTS_VARIABLE_NAME
        if not iteration is None:
            wname += "_F" + str(iteration)
        wshape = [self.input_size, self.output_size + self.input_size]

        self._kernel = tf.get_variable(wname, wshape, tf.float32, self._initializer,
                                       trainable=True)
        self.built = True

    def __call__(self, hidden_output, scope=None):
        # if self.prevcontext is None:
        #     self.prevcontext = prevctx
        with tf.variable_scope(scope or type(self).__name__):
            hidden_output = tf.reshape(hidden_output, [1, self.input_size])
            e = math_ops.mat_mul(array_ops.concat([hidden_output, self.prevcontext], 1),
                                 self.We2, transpose_b=True)
            e = math_ops.mat_mul(self._activation(e), self.We1, transpose_b=True)
            return e

    def update_context(self, last_output, useDropout_=False):
        ctx = math_ops.mat_mul(array_ops.concat([last_output, self.prevcontext], 1),
                               self._kernel, transpose_b=True)
        self.context = tf.nn.relu(ctx)
        if useDropout_:
            self.context = tf.nn.dropout(self.context, rate=0.5)
        return self.context


def stlstm_loop(lstm_size, input_data, nb_classes, usePrevGCA=False, previousGCA=None,
                iters=2, do_norm=False, useDropout=False):
    """https://github.com/philipperemy/tensorflow-multi-dimensional-lstm/blob/master/md_lstm.py
    Implements multi dimension LSTM
    @param lstm_size: the hidden units
    @param input_data: the data to process of shape [batch,frames,joints,channels]
    @param scope_n : the scope
    returns (y,states) - y=[batch,frames,joints,lstm_size[1]] the output of the lstm
    """

    with tf.variable_scope("ST-LSTM", reuse=tf.AUTO_REUSE):
        # Get the shape of the input (batch_size, x, y, channels)
        # shape = input_data.get_shape().as_list()
        shape = tf.shape(input_data)
        batch_size = shape[0]
        T_dim = shape[1]
        S_dim = shape[2]
        channels = shape[3]
        # Get the number of features (total number of input values per step)
        # features = S_dim * channels

        # Results list
        results = []
        # Create ST-LTSM cells
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
        # Create the GCA cells (one per iteration)
        # gca_cells = [GCACell(lstm_size, ite) for ite in range(1, iters + 1)]
        gca_cells = [[GCACell(lstm_size, ite) for ite in range(1, iters + 1)] for b_ in batch_size]

        # The batch size is inferred from the tensor size
        x = tf.reshape(input_data, [batch_size, T_dim, S_dim, channels])

        # Reorder inputs to (t, s, batch_size, features) - t=T_dim, s=S_dim
        x = tf.transpose(x, [1, 2, 0, 3])
        # Reshape to a one dimensional tensor of (t*s*batch_size , features)
        x = tf.reshape(x, [-1, batch_size, channels])
        # Split tensor into t*s tensors of size (batch_size , features)
        # x = tf.split(axis=0, num_or_size_splits=T_dim*S_dim, value=x)

        # Create an input tensor array (literally an array of tensors) to use inside the loop
        # inputs_ta = tf.TensorArray(dtype=tf.float32, size=T_dim * S_dim, name='input_array', dynamic_size=True,
        #                            infer_shape=False)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=T_dim * S_dim, name='input_array',
                                   dynamic_size=True, infer_shape=False)
        inputs_ta = inputs_ta.unstack(x)

        # Create a TensorArray for the order of the joints
        jointsorder_ta = tf.TensorArray(tf.int32, OUT_DIM1, clear_after_read=False)
        jointsorder_ta = jointsorder_ta.unstack(tf.constant(JOINTS_ORDER))

        # Function to get the previous joints id (cs_prev,hs_prev)
        def get_prevS(t_, w_=1):
            # return S_dim + tf.mod(t_, S_dim) - tf.constant(w_)
            # return t_ - tf.constant(w_)
            return jointsorder_ta.read(tf.mod(t_, OUT_DIM1)-1)

        # Function to get the previous time id (ct_prev,ht_prev)
        def get_prevT(t_, w_=S_dim):
            # return tf.mod(t_, w_)  # - tf.constant(w_)
            return t_ - w_

        def init_context(output_layer1):
            return tf.reduce_mean(output_layer1, axis=0)

        def process_information(id_, e_ta_):
            # TODO: Calculer la mesure d'information pour chaque GCA du batch
            # e_ta_data = np.arra
            # for b in batch_size_:
            gca_ = gca_cells[b][it - 1]
            e_ta_ = e_ta_.write(id_, gca_(outputs_ta.read(id_)))
            return id_ + 1, e_ta_

        # Controls the initial index
        zero = tf.constant(0)
        e_sum = tf.constant(0)

        # Body of the while loop operation that applies the MD LSTM
        def body1(id_, outputs_ta_, states_ta_):
            # If the current position is less or equal than the width, we are in the first row
            # so we read the zero state we added before.
            # If not, get the sample located at a width distance.
            prevstate_T = tf.cond(tf.less_equal(id_, OUT_DIM1),
                                  lambda: states_ta_.read(T_dim * OUT_DIM1),  # first row = zero state
                                  lambda: states_ta_.read(get_prevT(id_, OUT_DIM1)))  # other rows = previous time id (t-1)

            # If it is the first step we read the zero state if not we read the inmediate last
            prevstate_S = tf.cond(tf.less(zero, tf.mod(id_, OUT_DIM1)),
                                  lambda: states_ta_.read(get_prevS(id_)), # get previous joint state id (j-1) from JOINT ORDER
                                  lambda: states_ta_.read(T_dim * OUT_DIM1))  # first joint, get zero state

            # We build the input state in both dimensions
            current_state = prevstate_S[0], prevstate_T[0], prevstate_S[1], prevstate_T[1]
            # Now we calculate the hidden state and the new cell state
            out, state = cell(inputs_ta.read(id_), current_state)
            # We write the output to the output tensor array
            outputs_ta_ = outputs_ta_.write(id_, out)
            # And save the output state to the state tensor array
            states_ta_ = states_ta_.write(id_, state)

            # Return outputs and incremented time step
            return id_ + 1, outputs_ta_, states_ta_  # , outputs_ta2_, states_ta2_

        # Body of the while loop operation that applies the MD LSTM
        def body2(id_, outputs_ta2_, states_ta2_, e_ta_):
            # Informativeness
            r = e_ta_.read(id_) / e_sum
            prevstate_T2 = tf.cond(tf.less_equal(id_, OUT_DIM1),
                                   lambda: states_ta2_.read(T_dim * OUT_DIM1),  # first row = zero state
                                   lambda: states_ta2_.read(
                                       get_prevT(id_, OUT_DIM1)))  # other rows = previous time id (t-1)

            # If it is the first step we read the zero state if not we read the inmediate last
            prevstate_S2 = tf.cond(tf.less(zero, tf.mod(id_, OUT_DIM1)),
                                   lambda: states_ta2_.read(get_prevS(id_)),  # get previous joint state id (j-1)
                                   lambda: states_ta2_.read(
                                       T_dim * OUT_DIM1))  # first joint - get zero state !
            # Process cureent state and then the new state
            current_state2 = prevstate_S2[0], prevstate_T2[0], prevstate_S2[1], prevstate_T2[1]
            out2, state2 = cell2(outputs_ta.read(id_), current_state2, r)
            outputs_ta2_ = outputs_ta2_.write(id_, out2)
            states_ta2_ = states_ta2_.write(id_, state2)

            # Return outputs and incremented time step
            return id_ + 1, outputs_ta2_, states_ta2_, e_ta_

        # Loop output condition. The index, given by the time, should be less than the
        # total number of steps defined within the image
        def condition1(id_, outputs_ta_, states_ta_):  # , outputs_ta2_, states_ta2_
            return tf.less(id_, T_dim * S_dim)  # T_dim * S_dim

        def condition(id_, e_ta_):  # , outputs_ta2_, states_ta2_
            return tf.less(id_, T_dim * S_dim)

        def condition2(id_, outputs_ta2_, states_ta2_, e_ta_):  # , outputs_ta2_, states_ta2_
            return tf.less(id_, T_dim * S_dim)

        # Init ST-LSTM1 states and output arrays
        states_ta = tf.TensorArray(dtype=tf.float32, size=T_dim * OUT_DIM1 + 1, name='state_array_1',
                                   clear_after_read=False)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=T_dim * OUT_DIM1, name='output_array_1',
                                    clear_after_read=False)
        # initial cell hidden states: last position of the array = LSTMStateTuple filled with zeros
        states_ta = states_ta.write(T_dim * OUT_DIM1,
                                    LSTMStateTuple(tf.zeros([batch_size, lstm_size[0]], tf.float32),
                                                   tf.zeros([batch_size, lstm_size[0]], tf.float32)))

        # Loop 1: First ST-LSTM layer
        index = tf.constant(0)
        _, outputs_ta, states_ta = tf.while_loop(condition1, body1, [index, outputs_ta, states_ta],
                                                 parallel_iterations=1)
        for it in range(1, iters + 1):
            states_ta2 = tf.TensorArray(dtype=tf.float32, size=T_dim * OUT_DIM1 + 1, name='state_array2',
                                        clear_after_read=False)
            outputs_ta2 = tf.TensorArray(dtype=tf.float32, size=T_dim * OUT_DIM1, name='output_array2')

            # initial cell hidden states: last position of the array = LSTMStateTuple filled with zeros
            states_ta2 = states_ta2.write(T_dim * OUT_DIM1,
                                          LSTMStateTuple(tf.zeros([batch_size, lstm_size[1]], tf.float32),
                                                         tf.zeros([batch_size, lstm_size[1]], tf.float32)))

            # Informativeness tensors
            # e_ta = tf.TensorArray(tf.float32, T_dim * OUT_DIM1, name='e_it{}'.format(it), clear_after_read=False)
            e_ta = tf.TensorArray(tf.float32, T_dim * OUT_DIM1, name='e_array', clear_after_read=False)

            # # Loop 1: First ST-LSTM layer
            # index = tf.constant(0)
            # _, outputs_ta, states_ta = tf.while_loop(condition1, body1, [index, outputs_ta, states_ta],
            #                                          parallel_iterations=1)

            # Initialize context 0
            if it ==1:
                for b in batch_size:
                    if usePrevGCA:
                        initial_context = tf.cond(tf.less(tf.constant(0,dtype=tf.int64),
                                                          tf.count_nonzero(previousGCA[b])),
                                                  lambda: previousGCA[b],
                                                  lambda: init_context(outputs_ta.stack()[:,b,:]))
                        gca_cells[0].set_prevcontext(initial_context)
                    else:
                        gca_cells[0].set_prevcontext(init_context(outputs_ta.stack()[:,b,:]))

            # Process e
            # TODO: Handle batch of data
            index = tf.constant(0)
            _, e_ta = tf.while_loop(condition, process_information, [index, e_ta],
                                    parallel_iterations=1)
            e_sum = tf.reduce_sum(e_ta.stack(), axis=0)

            # Loop 2: Second ST-LSTM layer
            index = tf.constant(0)
            _, outputs_ta2, states_ta2, _ = tf.while_loop(condition2, body2,
                                                          [index, outputs_ta2, states_ta2, e_ta],
                                                          parallel_iterations=1)

            # Update context
            ctx = gca_cells[it - 1].update_context(outputs_ta2.read(S_dim * T_dim - 1), useDropout)
            # it += 1
            if it < iters:
                gca_cells[it].prevcontext = ctx

            # Compute Softmax from context
            Wc = tf.get_variable("Wc", [nb_classes, lstm_size[0]], tf.float32,
                                 tf.truncated_normal_initializer(), trainable=True)
            y = math_ops.mat_mul(Wc, gca_cells[it-1].context, transpose_b=True)
            y = tf.nn.softmax(tf.transpose(y))
            results.append(y)

        # Extract the output tensors from the processesed tensor array
        # outputs = outputs_ta2.stack()
        # states = states_ta2.stack()

        # Reshape outputs to match the shape of the input
        # y = tf.reshape(outputs, [T_dim, S_dim, batch_size, lstm_size[0]])   # For outputs_ta
        # states = tf.reshape(states, [T_dim,S_dim,batch_size,2,lstm_size[0]])
        # y = tf.reshape(outputs, [T_dim, S_dim, batch_size, lstm_size[1]])

        # Reorder te dimensions to match the input
        # y = tf.transpose(y, [2, 0, 1, 3])

        # Global Context
        gca = gca_cells[-1].context

        # Return the output and the inner states
        return results, gca


def stlstm_loss(prediction, ground_truth, nb_classes):
    label = tf.one_hot(ground_truth, nb_classes)
    label = tf.reshape(label, [-1, nb_classes])
    loss = tf.losses.log_loss(label, prediction)
    return loss




# class CudNNSTLSTMCell(CudnnLSTM):
#     """ST-LSTM cell adapted from Basic LSTM cell.
#     """
#
#     def __init__(self, num_units, initializer=None, input_shape=None,
#                  forget_bias=1.0, trainable=True, do_norm=False,
#                  activation=None, reuse=None, name=None, dtype=None, **kwargs):
#         """Initialize the parameters for an LSTM cell.
#         Args:
#           num_units: int, The number of units in the LSTM cell.
#           initializer: (optional) The initializer to use for the weight and
#             projection matrices.
#           forget_bias: Biases of the forget gate are initialized by default to 1
#             in order to reduce the scale of forgetting at the beginning of
#             the training. Must set it manually to `0.0` when restoring from
#             CudnnLSTM trained checkpoints.
#           activation: Activation function of the inner states.  Default: `tanh`. It
#             could also be string that is within Keras activation function names.
#           reuse: (optional) Python boolean describing whether to reuse variables
#             in an existing scope.  If not `True`, and the existing scope already has
#             the given variables, an error is raised.
#           name: String, the name of the layer. Layers with the same name will
#             share weights, but to avoid mistakes we require reuse=True in such
#             cases.
#           dtype: Default dtype of the layer (default of `None` means use the type
#             of the first input). Required when `build` is called before `call`.
#           **kwargs: Dict, keyword named properties for common layer attributes, like
#             `trainable` etc when constructing the cell from configs of get_config().
#           When restoring from CudnnLSTM-trained checkpoints, use
#           `CudnnCompatibleLSTMCell` instead.
#         """
#         super(CudNNSTLSTMCell, self).__init__(_reuse=reuse, name=name, trainable=trainable,
#                                          dtype=dtype, **kwargs)
#
#         # Inputs must be 2-dimensional.
#         self.input_spec = base_layer.InputSpec(ndim=2)
#         self.do_norm = do_norm
#         self.num_units = num_units
#         self._forget_bias = forget_bias
#         # self._initializer = initializers.get(initializer)
#         self._initializer = initializer
#
#         # self._state_is_tuple = state_is_tuple
#         if activation:
#             self._activation = activations.get(activation)
#         else:
#             self._activation = math_ops.tanh
#
#         self._state_size = (LSTMStateTuple(num_units, num_units))
#         self._output_size = num_units
#         if input_shape is None:
#             raise ValueError("Expected inputs_shape to be known")
#         else:
#             if not self.built:
#                 self.build(input_shape)
#
#     def state_shape(self, batch_size):
#         """Shape of Cudnn LSTM states.
#
#         Shape is a 2-element tuple. Each is
#         [num_layers * num_dirs, batch_size, num_units]
#         Args:
#           batch_size: an int
#         Returns:
#           a tuple of python arrays.
#         """
#         return ([self.num_layers * self.num_dirs, batch_size, self.num_units],
#                 [self.num_layers * self.num_dirs, batch_size, self.num_units])
#
#     @property
#     def output_size(self):
#         return self._output_size
#
#     def build(self, inputs_shape):
#         if inputs_shape[-1].value is None:
#             raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
#                              % str(inputs_shape))
#
#         input_depth = inputs_shape[-1].value  # Number of input channels
#         h_depth = self._num_units
#         wshape = [5 * self._num_units, input_depth + 2 * h_depth]
#
#         self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME,
#                                          shape=wshape,
#                                          initializer=self._initializer)
#         if self.dtype is None:
#             initializer = init_ops.zeros_initializer
#         else:
#             initializer = init_ops.zeros_initializer(dtype=self.dtype)
#         self._bias = self.add_variable(_BIAS_VARIABLE_NAME,
#                                        shape=[5 * self._num_units],
#                                        initializer=initializer)
#         self.built = True
#
#     def __call__(self, inputs, state, informativeness=None, scope=None):
#         """Run one step of LSTM.
#         Args:
#           inputs: input Tensor, must be 2-D, `[batch, input_size]`.
#           state: if `state_is_tuple` is False, this must be a state Tensor,
#             `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
#             tuple of state Tensors, both `2-D`, with column sizes `c_state` and
#             `h_state`.
#           do_norm: if True, perform batch normalization
#         Returns:
#           A tuple containing:
#           - A `2-D, [batch, output_dim]`, Tensor representing the output of the
#             LSTM after reading `inputs` when previous state was `state`.
#             Here output_dim is: num_units
#           - Tensor(s) representing the new state of LSTM after reading `inputs` when
#             the previous state was `state`.  Same type and shape(s) as `state`.
#         Raises:
#           ValueError: If input size cannot be inferred from inputs via
#             static shape inference.
#         """
#         with tf.variable_scope(scope or type(self).__name__):
#             (cs_prev, ct_prev, hs_prev, ht_prev) = state
#
#             # i = input_gate, fs = forget_gate_S, ft = forget_gate_T, o = output_gate, j = new_input
#             lstm_matrix = math_ops.matmul(array_ops.concat([inputs, hs_prev, ht_prev], 1),
#                                           self._kernel, transpose_b=True)
#             if not self.do_norm:
#                 lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)
#
#             i, fs, ft, o, j = array_ops.split(value=lstm_matrix, num_or_size_splits=5, axis=1)
#             if self.do_norm:
#                 i = normalization(i, 'i/')
#                 fs = normalization(fs, 'fs/')
#                 ft = normalization(ft, 'ft/')
#                 o = normalization(o, 'o/')
#                 j = normalization(j, 'j/')
#             # New state
#             if informativeness is None:
#                 c = (tf.nn.sigmoid(i) * self._activation(j) +
#                      tf.nn.sigmoid(fs + self._forget_bias) * cs_prev +
#                      tf.nn.sigmoid(ft + self._forget_bias) * ct_prev)
#             else:
#                 r = tf.nn.sigmoid(informativeness)
#                 c = (r * tf.nn.sigmoid(i) * self._activation(j) +
#                      (1 - r) * tf.nn.sigmoid(fs + self._forget_bias) * cs_prev +
#                      (1 - r) * tf.nn.sigmoid(ft + self._forget_bias) * ct_prev)
#             if self.do_norm:
#                 c = normalization(c, 'c/')
#             # New hidden state
#             h = tf.nn.sigmoid(o) * self._activation(c)
#
#             new_state = (LSTMStateTuple(c, h))
#             return h, new_state