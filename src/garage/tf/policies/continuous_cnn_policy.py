"""ContinuousCNNPolicy with model."""
import akro
import tensorflow as tf

from garage._functions import cnn_verify_obs_space
from garage.tf.models import CNNModel
from garage.tf.models import MLPModel
from garage.tf.models import Sequential
from garage.tf.policies import Policy


class ContinuousCNNPolicy(Policy):
    """ContinuousCNNPolicy.

    A policy that contains a CNN and a MLP to predict an action in continuous
    action spaces.

    It only works with akro.Box action spaces.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        filter_dims (tuple[int]): Dimension of the filters. For example,
            (3, 5) means there are two convolutional layers. The filter for
            first layer is of dimension (3 x 3) and the second one is of
            dimension (5 x 5).
        num_filters (tuple[int]): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        strides (tuple[int]): The stride of the sliding window. For
            example, (1, 2) means there are two convolutional layers. The
            stride of the filter for first layer is 1 and that of the second
            layer is 2.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        name (str): Policy name, also the variable scope of the policy.
        hidden_sizes (tuple[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this policy consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 filter_dims,
                 num_filters,
                 strides,
                 padding='SAME',
                 name='ContinuousCNNPolicy',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=tf.nn.softmax,
                 output_w_init=tf.initializers.glorot_uniform(),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        if not isinstance(env_spec.action_space, akro.Box):
            raise ValueError(
                'ContinuousCNNPolicy only works with akro.Box action '
                'spaces.')

        cnn_verify_obs_space(env_spec)
        super().__init__(name, env_spec)

        self.obs_dim = env_spec.observation_space.shape
        self.action_dim = env_spec.action_space.flat_dim
        self._action_space = env_spec.action_space
        self._obs_space = env_spec.observation_space

        self._env_spec = env_spec
        self._filter_dims = filter_dims
        self._num_filters = num_filters
        self._strides = strides
        self._padding = padding
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        self.model = Sequential(
            CNNModel(filter_dims=filter_dims,
                     num_filters=num_filters,
                     strides=strides,
                     padding=padding,
                     hidden_nonlinearity=hidden_nonlinearity,
                     name='CNNModel'),
            MLPModel(output_dim=self.action_dim,
                     hidden_sizes=hidden_sizes,
                     hidden_nonlinearity=hidden_nonlinearity,
                     hidden_w_init=hidden_w_init,
                     hidden_b_init=hidden_b_init,
                     output_nonlinearity=output_nonlinearity,
                     output_w_init=output_w_init,
                     output_b_init=output_b_init,
                     layer_normalization=layer_normalization,
                     name='MLPModel'))

        self._initialize()

    def _initialize(self):
        if isinstance(self._obs_space, akro.Image):
            state_input = tf.compat.v1.placeholder(tf.uint8,
                                                   shape=(None, ) +
                                                   self.obs_dim)
            state_input = tf.cast(state_input, tf.float32)
            state_input /= 255.0
        else:
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, ) +
                                                   self.obs_dim)

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(state_input)

        self._f_prob = tf.compat.v1.get_default_session().make_callable(
            self.model.outputs, feed_list=[self.model.input])

    @property
    def vectorized(self):
        """bool: True if primitive supports vectorized operations."""
        return True

    def get_action_sym(self, obs_var, name=None):
        r"""Symbolic graph of the action.

        Args:
            obs_var (tf.Tensor): Tensor input of shape
                :math:`(N, O*)` for symbolic graph.
            name (str): Name for symbolic graph.

        Returns:
             tf.Tensor: symbolic tensor with shape :math:`(N \bullet [A], )`.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            if isinstance(self._obs_space, akro.Image):
                obs_var = tf.cast(obs_var, tf.float32) / 255.0
            return self.model.build(obs_var, name=name)

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment of shape
                :math:`(O*, )`.

        Returns:
            numpy.ndarray: Predicted action of shape :math:`(A*, )`.
            dict: Empty dict since this policy does not model a distribution.

        """
        if len(observation.shape) < len(self.obs_dim):
            observation = self._obs_space.unflatten(observation)
        action = self._f_prob([observation])
        action = self._action_space.unflatten(action)
        return action, dict()

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment of
                shape :math:`(N, O*)`.

        Returns:
            numpy.ndarray: Predicted actions of shape :math:`(N, A*)`.
            dict: Empty dict since this policy does not model a distribution.

        """
        if len(observations[0].shape) < len(self.obs_dim):
            observations = self._obs_space.unflatten_n(observations)
        actions = self._f_prob(observations)
        actions = self._action_space.unflatten_n(actions)

        return actions, dict()

    def clone(self, name):
        """Return a clone of the policy.

        It only copies the configuration of the policy,
        not the parameters.

        Args:
            name (str): Name of the newly created policy. It has to be
                different from source policy if cloned under the same
                computational graph.

        Returns:
            garage.tf.policies.ContinuousCNNPolicy: Clone of this object

        """
        return self.__class__(env_spec=self._env_spec,
                              filter_dims=self._filter_dims,
                              num_filters=self._num_filters,
                              strides=self._strides,
                              padding=self._padding,
                              name=name,
                              hidden_sizes=self._hidden_sizes,
                              hidden_nonlinearity=self._hidden_nonlinearity,
                              hidden_w_init=self._hidden_w_init,
                              hidden_b_init=self._hidden_b_init,
                              output_nonlinearity=self._output_nonlinearity,
                              output_w_init=self._output_w_init,
                              output_b_init=self._output_b_init,
                              layer_normalization=self._layer_normalization)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
