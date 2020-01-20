"""Model-Agnostic Meta-Learning (MAML) algorithm implementation for RL."""
import collections

from dowel import tabular
import numpy as np
import torch

from garage.misc import tensor_utils
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.samplers import BatchSampler
from garage.torch.algos import _Default, make_optimizer
from garage.torch.optimizers import ConjugateGradientOptimizer, DiffSGD
from garage.torch.utils import update_module_params


class MAML:
    """Model-Agnostic Meta-Learning (MAML).

    Args:
        env (garage.envs.GarageEnv): A gym environment.
        policy (garage.torch.policies.Policy): Policy.
        baseline (garage.np.baselines.Baseline): The baseline.
        meta_optimizer (Union[torch.optim.Optimizer, tuple]):
            Type of optimizer.
            This can be an optimizer type such as `torch.optim.Adam` or a tuple
            of type and dictionary, where dictionary contains arguments to
            initialize the optimizer e.g. `(torch.optim.Adam, {'lr' = 1e-3})`.
        meta_batch_size (int): Number of tasks sampled per batch.
        inner_lr (double): Adaptation learning rate.
        num_grad_updates (int): Number of adaptation gradient steps.
        inner_algo (garage.torch.algos.VPG): The inner algorithm used for
            computing loss.

    """

    def __init__(self,
                 inner_algo,
                 env,
                 policy,
                 baseline,
                 meta_optimizer,
                 meta_batch_size=40,
                 inner_lr=0.1,
                 num_grad_updates=1):
        if policy.vectorized:
            self.sampler_cls = OnPolicyVectorizedSampler
        else:
            self.sampler_cls = BatchSampler

        self.max_path_length = inner_algo.max_path_length
        self._policy = policy
        self._env = env
        self._baseline = baseline
        self._num_grad_updates = num_grad_updates
        self._meta_batch_size = meta_batch_size
        self._inner_algo = inner_algo
        self._inner_optimizer = DiffSGD(self._policy, lr=inner_lr)
        self._meta_optimizer = make_optimizer(meta_optimizer,
                                              policy,
                                              lr=_Default(1e-3),
                                              eps=_Default(1e-5))
        self._episode_reward_mean = [
            collections.deque(maxlen=100) for _ in range(num_grad_updates + 1)
        ]

    def train(self, runner):
        """Obtain samples and start training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            all_samples, all_params = self._obtain_samples(runner)
            tabular.record('TotalEnvSteps', runner.total_env_steps)
            last_return = self.train_once(runner, all_samples, all_params)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, all_samples, all_params):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            all_samples (list[list[MAMLTrajectoryBatch]]): A two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).

        Returns:
            float: Average return.

        """
        old_theta = dict(self._policy.named_parameters())

        kl_before = self._compute_kl_constraint(itr,
                                                all_samples,
                                                all_params,
                                                set_grad=False)

        meta_objective = self._compute_meta_loss(itr, all_samples, all_params)

        self._meta_optimizer.zero_grad()
        meta_objective.backward()

        self._meta_optimize(itr, all_samples, all_params)

        # Log
        loss_after = self._compute_meta_loss(itr,
                                             all_samples,
                                             all_params,
                                             set_grad=False)
        kl_after = self._compute_kl_constraint(itr,
                                               all_samples,
                                               all_params,
                                               set_grad=False)

        with torch.no_grad():
            policy_entropy = self._compute_policy_entropy(
                [task_samples[0] for task_samples in all_samples])
            average_return = self.evaluate_performance(
                itr, all_samples, meta_objective.item(), loss_after.item(),
                kl_before.item(), kl_after.item(),
                policy_entropy.mean().item())

        update_module_params(self._old_policy, old_theta)

        return average_return

    def _obtain_samples(self, runner):
        """Obtain samples for each task before and after the fast-adaptation.

        Args:
            runner (LocalRunner): A local runner instance to obtain samples.

        Returns:
            tuple: Tuple of (all_samples, all_params).
                all_samples (list[MAMLTrajectoryBatch]): A list of size
                    [meta_batch_size * (num_grad_updates + 1)]
                all_params (list[dict]): A list of named parameter
                    dictionaries.

        """
        tasks = self._env.sample_tasks(self._meta_batch_size)
        all_samples = [[] for _ in range(len(tasks))]
        all_params = []
        theta = dict(self._policy.named_parameters())

        for i, task in enumerate(tasks):
            self._set_task(runner, task)

            for j in range(self._num_grad_updates + 1):
                paths = runner.obtain_samples(runner.step_itr)
                batch_samples = self._process_samples(runner.step_itr, paths)
                all_samples[i].append(batch_samples)

                # The last iteration does only sampling but no adapting
                if j != self._num_grad_updates:
                    self._adapt(runner.step_itr, batch_samples, set_grad=False)

            all_params.append(dict(self._policy.named_parameters()))
            # Restore to pre-updated policy
            update_module_params(self._policy, theta)

        return all_samples, all_params

    def _adapt(self, itr, batch_samples, set_grad=True):
        """Performs one MAML inner step to update the policy.

        Args:
            itr (int): Iteration.
            batch_samples (MAMLTrajectoryBatch): Samples data for one
                task and one gradient step.
            set_grad (bool): if False, update policy parameters in-place.
                Else, allow taking gradient of functions of updated parameters
                with respect to pre-updated parameters.

        """
        loss = self._compute_loss(itr, batch_samples)

        # Update policy parameters with one SGD step
        self._inner_optimizer.zero_grad()
        loss.backward(create_graph=set_grad)

        with torch.set_grad_enabled(set_grad):
            self._inner_optimizer.step()

    def _meta_optimize(self, itr, all_samples, all_params):
        if isinstance(self._meta_optimizer, ConjugateGradientOptimizer):
            self._meta_optimizer.step(
                f_loss=lambda: self._compute_meta_loss(
                    itr, all_samples, all_params, set_grad=False),
                f_constraint=lambda: self._compute_kl_constraint(
                    itr, all_samples, all_params))
        else:
            self._meta_optimizer.step(lambda: self._compute_meta_loss(
                itr, all_samples, all_params, set_grad=False))

    def _compute_meta_loss(self, itr, all_samples, all_params, set_grad=True):
        """Compute loss to meta-optimize.

        Args:
            itr (int): Iteration number.
            all_samples (list[list[MAMLTrajectoryBatch]]): A two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).
            set_grad (bool): Whether to enable gradient calculation or not.

        Returns:
            torch.Tensor: Calculated mean value of loss.

        """
        theta = dict(self._policy.named_parameters())
        old_theta = dict(self._old_policy.named_parameters())

        losses = []
        for task_samples, task_params in zip(all_samples, all_params):
            for i in range(self._num_grad_updates):
                self._adapt(itr, task_samples[i], set_grad=set_grad)

            update_module_params(self._old_policy, task_params)
            with torch.set_grad_enabled(set_grad):
                loss = self._compute_loss(itr, task_samples[-1])
            losses.append(loss)

            update_module_params(self._policy, theta)
            update_module_params(self._old_policy, old_theta)

        return torch.stack(losses).mean()

    def _compute_kl_constraint(self,
                               itr,
                               all_samples,
                               all_params,
                               set_grad=True):
        """Compute KL divergence.

        For each task, compute the KL divergence between the old policy
        distribution and current policy distribution.

        Args:
            itr (int): Iteration number.
            all_samples (list[list[MAMLTrajectoryBatch]]): Two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).
            set_grad (bool): Whether to enable gradient calculation or not.

        Returns:
            torch.Tensor: Calculated mean value of KL divergence.

        """
        theta = dict(self._policy.named_parameters())
        old_theta = dict(self._old_policy.named_parameters())

        kls = []
        for task_samples, task_params in zip(all_samples, all_params):
            for i in range(self._num_grad_updates):
                self._adapt(itr, task_samples[i], set_grad=set_grad)

            update_module_params(self._old_policy, task_params)
            with torch.set_grad_enabled(set_grad):
                # pylint: disable=protected-access
                kl = self._inner_algo._compute_kl_constraint(
                    task_samples[-1].observations)
            kls.append(kl)

            update_module_params(self._policy, theta)
            update_module_params(self._old_policy, old_theta)

        return torch.stack(kls).mean()

    def _compute_loss(self, itr, batch_samples):
        """Compute loss for one task and one gradient step.

        Args:
            itr (int): Iteration number.
            batch_samples (MAMLTrajectoryBatch): Samples data for one
                task and one gradient step.

        Returns:
            torch.Tensor: Computed loss value.

        """
        # pylint: disable=protected-access
        return self._inner_algo._compute_loss(itr, batch_samples.observations,
                                              batch_samples.actions,
                                              batch_samples.rewards,
                                              batch_samples.lengths,
                                              batch_samples.baselines)

    def _compute_policy_entropy(self, task_samples):
        """Compute policy entropy.

        Args:
            task_samples (list[MAMLTrajectoryBatch]): Samples data for
                one task.

        Returns:
            torch.Tensor: Computed entropy value.

        """
        # pylint: disable=protected-access
        entropies = [
            self._inner_algo._compute_policy_entropy(samples.observations)
            for samples in task_samples
        ]
        return torch.stack(entropies).mean()

    def _set_task(self, runner, task):
        # pylint: disable=protected-access, no-self-use
        for env in runner._sampler._vec_env.envs:
            env.set_task(task)

    @property
    def policy(self):
        """Current policy of the inner algorithm.

        Returns:
            garage.torch.policies.Policy: Current policy of the inner
                algorithm.

        """
        return self._policy

    @property
    def _old_policy(self):
        """Old policy of the inner algorithm.

        Returns:
            garage.torch.policies.Policy: Old policy of the inner algorithm.

        """
        # pylint: disable=protected-access
        return self._inner_algo._old_policy

    def _process_samples(self, itr, paths):
        """Process sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            MAMLTrajectoryBatch: Processed samples data.

        """
        for path in paths:
            path['returns'] = tensor_utils.discount_cumsum(
                path['rewards'], self._inner_algo.discount)

        self._baseline.fit(paths)
        obs, actions, rewards, valids, baselines \
            = self._inner_algo.process_samples(itr, paths)
        return MAMLTrajectoryBatch(obs, actions, rewards, valids, baselines)

    def evaluate_performance(self, itr, all_samples, loss_before, loss_after,
                             kl_before, kl, policy_entropy):
        """Evaluate performance of this batch.

        Args:
            itr (int): Iteration number.
            all_samples (list[list[MAMLTrajectoryBatch]]): Two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            loss_before (float): Loss before optimization step.
            loss_after (float): Loss after optimization step.
            kl_before (float): KL divergence before optimization step.
            kl (float): KL divergence after optimization step.
            policy_entropy (float): Policy entropy.

        Returns:
            float: The average return in last epoch cycle.

        """
        tabular.record('Iteration', itr)

        for i in range(self._num_grad_updates + 1):
            all_rewards = [
                path_rewards for task_samples in all_samples
                for path_rewards in task_samples[i].rewards.numpy()
            ]

            discounted_returns = [
                tensor_utils.discount_cumsum(path_rewards,
                                             self._inner_algo.discount)[0]
                for path_rewards in all_rewards
            ]
            undiscounted_returns = np.sum(all_rewards, axis=-1)
            average_return = np.mean(undiscounted_returns)
            self._episode_reward_mean[i].extend(undiscounted_returns)

            with tabular.prefix('Update_{0}/'.format(i)):
                tabular.record('AverageDiscountedReturn',
                               np.mean(discounted_returns))
                tabular.record('AverageReturn', average_return)
                tabular.record('Extras/EpisodeRewardMean',
                               np.mean(self._episode_reward_mean[i]))
                tabular.record('StdReturn', np.std(undiscounted_returns))
                tabular.record('MaxReturn', np.max(undiscounted_returns))
                tabular.record('MinReturn', np.min(undiscounted_returns))
                tabular.record('NumTrajs', len(all_rewards))

        with tabular.prefix(self._policy.name + '/'):
            tabular.record('LossBefore', loss_before)
            tabular.record('LossAfter', loss_after)
            tabular.record('dLoss', loss_before - loss_after)
            tabular.record('KLBefore', kl_before)
            tabular.record('KLAfter', kl)
            tabular.record('Entropy', policy_entropy)

        return average_return


class MAMLTrajectoryBatch(
        collections.namedtuple(
            'MAMLTrajectoryBatch',
            ['observations', 'actions', 'rewards', 'lengths', 'baselines'])):
    r"""A tuple representing a batch of whole trajectories in MAML.

    A :class:`MAMLTrajectoryBatch` represents a batch of whole trajectories
    produced from one environment.

    +-----------------------+-------------------------------------------------+
    | Symbol                | Description                                     |
    +=======================+=================================================+
    | :math:`N`             | Trajectory index dimension                      |
    +-----------------------+-------------------------------------------------+
    | :math:`[T]`           | Variable-length time dimension of each          |
    |                       | trajectory                                      |
    +-----------------------+-------------------------------------------------+
    | :math:`S^*`           | Single-step shape of a time-series tensor       |
    +-----------------------+-------------------------------------------------+
    | :math:`N \bullet [T]` | A dimension computed by flattening a            |
    |                       | variable-length time dimension :math:`[T]` into |
    |                       | a single batch dimension with length            |
    |                       | :math:`sum_{i \in N} [T]_i`                     |
    +-----------------------+-------------------------------------------------+

    Attributes:
        observations (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet [T], O^*)` containing the (possibly
            multi-dimensional) observations for all time steps in this batch.
            These must conform to :obj:`env_spec.observation_space`.
        actions (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet [T], A^*)` containing the (possibly
            multi-dimensional) actions for all time steps in this batch. These
            must conform to :obj:`env_spec.action_space`.
        rewards (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet [T])` containing the rewards for all time
            steps in this batch.
        lengths (numpy.ndarray): An integer numpy array of shape :math:`(N, )`
            containing the length of each trajectory in this batch. This may be
            used to reconstruct the individual trajectories.
        baselines (numpy.ndarray): An numpy array of shape
            :math:`(N \bullet [T], )` containing the value function estimation
            at all time steps in this batch.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
