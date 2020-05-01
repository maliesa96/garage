"""Tests for garage.tf.policies.ContinuousCNNPolicy"""
import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.models import CNNModel
from garage.tf.policies import ContinuousCNNPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxPixelEnv
from tests.fixtures.envs.dummy import DummyDictEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.models import SimpleCNNModel
from tests.fixtures.models import SimpleMLPModel


class TestContinuousCNNPolicy(TfGraphTestCase):
    """Test class for ContinuousCNNPolicy"""

    @pytest.mark.parametrize(
        'obs_dim, action_dim, filter_dims, num_filters, '
        'strides, padding, hidden_sizes', [
            ((1, 1), (1, ), (32, ), (3, ), (1, ), 'VALID', (4, )),
            ((2, 2), (2, ), (32, ), (3, ), (1, ), 'VALID', (4, )),
            ((2, 2), (2, ), (32, 64), (3, 3), (1, 1), 'VALID', (4, 4)),
            ((2, 2, 2), (2, ), (32, 64), (3, 3), (2, 2), 'VALID', (4, 4)),
            ((1, 1, 1), (1, ), (32, ), (3, ), (1, ), 'SAME', (4, )),
            ((2, 2), (2, ), (32, ), (3, ), (1, ), 'SAME', (4, )),
            ((2, 2), (2, ), (32, 64), (3, 3), (1, 1), 'SAME', (4, 4)),
            ((2, 2), (2, ), (32, 64), (3, 3), (2, 2), 'SAME', (4, 4)),
        ])
    def test_get_action(self, obs_dim, action_dim, filter_dims, num_filters,
                        strides, padding, hidden_sizes):
        """Test get_action and get_actions."""

        env = TfEnv(DummyBoxPixelEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'continuous_cnn_policy.MLPModel'),
                        new=SimpleMLPModel):
            with mock.patch(('garage.tf.policies.'
                             'continuous_cnn_policy.CNNModel'),
                            new=SimpleCNNModel):
                policy = ContinuousCNNPolicy(env_spec=env.spec,
                                             filter_dims=filter_dims,
                                             num_filters=num_filters,
                                             strides=strides,
                                             padding=padding,
                                             hidden_sizes=hidden_sizes)

        env.reset()
        obs, _, _, _ = env.step(1)

        action, dist = policy.get_action(obs)
        expected_action = np.full(action_dim, 0.5)

        assert len(dist) == 0
        assert env.action_space.contains(action)
        assert np.array_equal(action, expected_action)

        actions, dists = policy.get_actions([obs, obs, obs])
        assert len(dists) == 0

        for action in actions:
            assert env.action_space.contains(action)
            assert np.array_equal(action, expected_action)

    @pytest.mark.parametrize(
        'obs_dim, action_dim, filter_dims, num_filters, '
        'strides, padding, hidden_sizes', [
            ((1, 1), (1, ), (32, ), (3, ), (1, ), 'VALID', (4, )),
            ((2, 2), (2, ), (32, ), (3, ), (1, ), 'VALID', (4, )),
            ((2, 2), (2, ), (32, 64), (3, 3), (1, 1), 'VALID', (4, 4)),
            ((2, 2, 2), (2, ), (32, 64), (3, 3), (2, 2), 'VALID', (4, 4)),
            ((1, 1, 1), (1, ), (32, ), (3, ), (1, ), 'SAME', (4, )),
            ((2, 2), (2, ), (32, ), (3, ), (1, ), 'SAME', (4, )),
            ((2, 2), (2, ), (32, 64), (3, 3), (1, 1), 'SAME', (4, 4)),
            ((2, 2), (2, ), (32, 64), (3, 3), (2, 2), 'SAME', (4, 4)),
        ])
    def test_get_action_sym(self, obs_dim, action_dim, filter_dims,
                            num_filters, strides, padding, hidden_sizes):
        env = TfEnv(DummyBoxPixelEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'continuous_cnn_policy.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.policies.'
                             'continuous_cnn_policy.MLPModel'),
                            new=SimpleMLPModel):
                policy = ContinuousCNNPolicy(env_spec=env.spec,
                                             filter_dims=filter_dims,
                                             num_filters=num_filters,
                                             strides=strides,
                                             padding=padding,
                                             hidden_sizes=hidden_sizes)

        env.reset()
        obs, _, _, _ = env.step(1)

        state_input = tf.compat.v1.placeholder(tf.uint8,
                                               shape=(None, ) + obs_dim)
        action_sym = policy.get_action_sym(state_input, name='action_sym')

        expected_action = np.full(action_dim, 0.5)

        action = self.sess.run(action_sym, feed_dict={state_input: [obs]})
        action = policy._action_space.unflatten(action)

        assert np.array_equal(action, expected_action)
        assert env.action_space.contains(action)

    def test_obs_is_image(self):
        """akro.Image observations should be normalized."""
        env = TfEnv(DummyBoxPixelEnv(), is_image=True)
        with mock.patch(('garage.tf.policies.'
                         'continuous_cnn_policy.CNNModel._build'),
                        autospec=True,
                        side_effect=CNNModel._build) as build:
            policy = ContinuousCNNPolicy(env_spec=env.spec,
                                         num_filters=(32, ),
                                         filter_dims=(3, ),
                                         strides=(1, ),
                                         padding='VALID',
                                         hidden_sizes=(3, ))

            normalized_obs = build.call_args_list[0][0][1]

            input_ph = tf.compat.v1.get_default_graph().get_tensor_by_name(
                'Placeholder:0')

            fake_obs = [np.full(env.spec.observation_space.shape, 255)]
            assert (self.sess.run(normalized_obs,
                                  feed_dict={input_ph: fake_obs}) == 1.).all()

            obs_dim = env.spec.observation_space.shape
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, ) + obs_dim)

            policy.get_action_sym(state_input, name='another')
            normalized_obs = build.call_args_list[1][0][1]

            input_ph = tf.compat.v1.get_default_graph().get_tensor_by_name(
                'Placeholder_1:0')

            fake_obs = [np.full(env.spec.observation_space.shape, 255)]
            assert (self.sess.run(normalized_obs,
                                  feed_dict={state_input:
                                             fake_obs}) == 1.).all()

    # yapf: disable
    @pytest.mark.parametrize(
        'obs_dim,', [
            (1, 1),
            (2, 2),
            (2, 2),
            (2, 2, 2),
            (1, 1, 1),
        ])
    # yapf: enable
    def test_flat_obs(self, obs_dim):
        """Flattened observations should be unflattened by the policy."""

        env = TfEnv(DummyBoxPixelEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.policies.'
                         'continuous_cnn_policy.MLPModel'),
                        new=SimpleMLPModel):
            with mock.patch(('garage.tf.policies.'
                             'continuous_cnn_policy.CNNModel'),
                            new=SimpleCNNModel):
                policy = ContinuousCNNPolicy(env_spec=env.spec,
                                             num_filters=(32, ),
                                             filter_dims=(3, ),
                                             strides=(1, ),
                                             padding='VALID',
                                             hidden_sizes=(3, ))

        expected_action = np.full(policy.action_dim, 0.5)
        policy._f_prob = mock.MagicMock(return_value=expected_action)

        env.reset()
        action_dim = env.action_space.flat_dim
        obs, _, _, _ = env.step(1)

        flattened = env.observation_space.flatten(obs)
        action, dist = policy.get_action(flattened)

        # obs should be unflattened before being passed to f_prob()
        # in get_action() and get_actions()

        unflattened_obs = policy._f_prob.call_args_list[0][0][0]
        assert (obs == unflattened_obs).all()

        expected_actions = np.full((3, policy.action_dim), 0.5)
        policy._f_prob = mock.MagicMock(return_value=expected_actions)

        actions, dists = policy.get_actions([flattened, flattened, flattened])
        for idx, _ in enumerate(actions):
            unflattened_obs = policy._f_prob.call_args_list[0][0][0][idx]
            assert (obs == unflattened_obs).all()

    def test_invalid_obs_type(self):
        """ContinuousCNNPolicy only accepts akro.Box observation spaces."""
        env = TfEnv(DummyDictEnv())
        with pytest.raises(ValueError):
            ContinuousCNNPolicy(env_spec=env.spec,
                                num_filters=(32, ),
                                filter_dims=(3, ),
                                strides=(1, ),
                                padding='VALID',
                                hidden_sizes=(3, ))

    def test_invalid_action_type(self):
        """ContinuousCNNPolicy only accepts akro.Box action spaces."""
        env = TfEnv(DummyDiscreteEnv())
        with pytest.raises(ValueError):
            ContinuousCNNPolicy(env_spec=env.spec,
                                num_filters=(32, ),
                                filter_dims=(3, ),
                                strides=(1, ),
                                padding='VALID',
                                hidden_sizes=(3, ))

    # yapf: disable
    @pytest.mark.parametrize('obs_dim', [
                             (1, ),
                             (2, 2, 2, 2),
                             (3, 3, 3, 3, 3),
                            ])
    # yapf: enable
    def test_invalid_obs_shape(self, obs_dim):
        """Only 2D and 3D observations should be accepted by CNNs."""
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim))
        with pytest.raises(ValueError):
            ContinuousCNNPolicy(env_spec=env.spec,
                                num_filters=(32, ),
                                filter_dims=(3, ),
                                strides=(1, ),
                                padding='VALID',
                                hidden_sizes=(3, ))

    def test_obs_not_image(self):
        """Non-akro.Image observations should not be normalized."""
        env = TfEnv(DummyBoxPixelEnv(), is_image=False)
        with mock.patch(('garage.tf.policies.'
                         'continuous_cnn_policy.CNNModel._build'),
                        autospec=True,
                        side_effect=CNNModel._build) as build:
            policy = ContinuousCNNPolicy(env_spec=env.spec,
                                         num_filters=(32, ),
                                         filter_dims=(3, ),
                                         strides=(1, ),
                                         padding='VALID',
                                         hidden_sizes=(3, ))

            normalized_obs = build.call_args_list[0][0][1]

            input_ph = tf.compat.v1.get_default_graph().get_tensor_by_name(
                'Placeholder:0')

            fake_obs = [np.full(env.spec.observation_space.shape, 255)]
            assert (self.sess.run(normalized_obs,
                                  feed_dict={input_ph:
                                             fake_obs}) == 255.).all()

            obs_dim = env.spec.observation_space.shape
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, ) + obs_dim)

            policy.get_action_sym(state_input, name='another')
            normalized_obs = build.call_args_list[1][0][1]

            input_ph = tf.compat.v1.get_default_graph().get_tensor_by_name(
                'Placeholder_1:0')

            fake_obs = [np.full(env.spec.observation_space.shape, 255)]
            assert (self.sess.run(normalized_obs,
                                  feed_dict={state_input:
                                             fake_obs}) == 255.).all()

    @pytest.mark.parametrize(
        'obs_dim, action_dim, filter_dims, num_filters, '
        'strides, padding, hidden_sizes', [
            ((1, 1), (1, ), (32, ), (3, ), (1, ), 'VALID', (4, )),
            ((2, 2), (2, ), (32, ), (3, ), (1, ), 'VALID', (4, )),
            ((2, 2), (2, ), (32, 64), (3, 3), (1, 1), 'VALID', (4, 4)),
            ((2, 2, 2), (2, ), (32, 64), (3, 3), (2, 2), 'VALID', (4, 4)),
            ((1, 1, 1), (1, ), (32, ), (3, ), (1, ), 'SAME', (4, )),
            ((2, 2), (2, ), (32, ), (3, ), (1, ), 'SAME', (4, )),
            ((2, 2), (2, ), (32, 64), (3, 3), (1, 1), 'SAME', (4, 4)),
            ((2, 2), (2, ), (32, 64), (3, 3), (2, 2), 'SAME', (4, 4)),
        ])
    def test_is_pickleable(self, obs_dim, action_dim, filter_dims, num_filters,
                           strides, padding, hidden_sizes):
        """Test if ContinuousCNNPolicy is pickleable."""
        env = TfEnv(DummyBoxPixelEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'continuous_cnn_policy.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.policies.'
                             'continuous_cnn_policy.MLPModel'),
                            new=SimpleMLPModel):
                policy = ContinuousCNNPolicy(env_spec=env.spec,
                                             filter_dims=filter_dims,
                                             num_filters=num_filters,
                                             strides=strides,
                                             padding=padding,
                                             hidden_sizes=hidden_sizes)
        env.reset()
        obs, _, _, _ = env.step(1)

        with tf.compat.v1.variable_scope(
                'ContinuousCNNPolicy/Sequential/MLPModel', reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())
        output1 = self.sess.run(policy.model.outputs,
                                feed_dict={policy.model.input: [obs]})

        p = pickle.dumps(policy)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(policy_pickled.model.outputs,
                               feed_dict={policy_pickled.model.input: [obs]})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize(
        'obs_dim, action_dim, filter_dims, num_filters, '
        'strides, padding, hidden_sizes', [
            ((1, 1), (1, ), (32, ), (3, ), (1, ), 'VALID', (1, )),
            ((2, 2), (2, ), (32, ), (3, ), (1, ), 'VALID', (2, )),
            ((2, 2), (2, ), (32, 64), (3, 3), (1, 1), 'VALID', (4, 4)),
            ((2, 2, 2), (2, ), (32, 64), (3, 3), (2, 2), 'VALID', (4, 4)),
            ((1, 1, 1), (1, ), (32, ), (3, ), (1, ), 'SAME', (4, )),
        ])
    def test_clone(self, obs_dim, action_dim, filter_dims, num_filters,
                   strides, padding, hidden_sizes):
        env = TfEnv(DummyBoxPixelEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'continuous_cnn_policy.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.policies.'
                             'continuous_cnn_policy.MLPModel'),
                            new=SimpleMLPModel):
                policy = ContinuousCNNPolicy(env_spec=env.spec,
                                             filter_dims=filter_dims,
                                             num_filters=num_filters,
                                             strides=strides,
                                             padding=padding,
                                             hidden_sizes=hidden_sizes)

                policy_clone = policy.clone('ContinuousCNNPolicyClone')
        assert policy.env_spec == policy_clone.env_spec
