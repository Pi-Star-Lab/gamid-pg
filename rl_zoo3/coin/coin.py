import warnings
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union

from gym import spaces
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_parameters_by_name,
    get_system_info,
    polyak_update,
    safe_mean,
    should_collect_more_steps,
)
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.dqn.policies import (
    CnnPolicy,
    DQNPolicy,
    MlpPolicy,
    MultiInputPolicy,
)

SelfCOIN = TypeVar("SelfCOIN", bound="COIN")


class COINReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    next_actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class COINReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.next_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        next_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        self.next_actions[self.pos] = np.array(next_action).copy()

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> COINReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self.next_actions[batch_inds, env_indices, :],
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
        )
        return COINReplayBufferSamples(*tuple(map(self.to_torch, data)))


class COIN(OffPolicyAlgorithm):
    """
    Continual Optimistic Initialization (COIN)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param max_grad_norm: The maximum value for the gradient clipping
    :param bonus: Bonus value
    :param bonus_update_interval: Update the bonus every ``bonus_update_interval`` steps.
    :param regret_bound: The regret bound.
    :param prior_q_net_path: The path to the prior q-net (baseline policy).
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.0001,
        buffer_size: int = 1000000,
        learning_starts: int = 100000,
        batch_size: int = 32,
        tau: float = 1,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[COINReplayBuffer]] = COINReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        max_grad_norm: float = 10,
        bonus: float = 1,
        bonus_update_interval: int = 10000000,
        regret_bound: float = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )
        # Bonus
        self.bonus = bonus
        self.bonus_update_interval = bonus_update_interval
        # Cumulative bonus
        self.cumulative_bonus = self.bonus
        # Return of the prior policy
        self.prior_return_buffer = deque(maxlen=10)
        self.prior_return = 0
        # Baseline regret
        self.regret = 0
        self.regret_bound = regret_bound
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        self.q_coin, self.q_coin_target = None, None
        self.q_pi, self.q_pi_target = None, None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        elif self.replay_buffer_class == HerReplayBuffer:
            assert (
                self.env is not None
            ), "You must pass an environment when using `HerReplayBuffer`"

            # If using offline sampling, we need a classic replay buffer too
            if self.replay_buffer_kwargs.get("online_sampling", True):
                replay_buffer = None
            else:
                replay_buffer = DictReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    device=self.device,
                    optimize_memory_usage=self.optimize_memory_usage,
                )

            self.replay_buffer = HerReplayBuffer(
                self.env,
                self.buffer_size,
                device=self.device,
                replay_buffer=replay_buffer,
                **self.replay_buffer_kwargs,
            )

        if self.replay_buffer is None:
            self.replay_buffer = COINReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        self.q_pi_policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.q_pi_policy = self.q_pi_policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_coin, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.q_coin_target, ["running_"]
        )

        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(
                self.target_update_interval // self.n_envs, 1
            )

    def _setup_prior(self):
        self.prior_policy = deepcopy(self.policy)
        self.q_prior = self.prior_policy.q_net

    def _create_aliases(self) -> None:
        self.q_coin = self.policy.q_net
        self.q_coin_target = self.policy.q_net_target
        self.q_pi = self.q_pi_policy.q_net
        self.q_pi_target = self.q_pi_policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the bonus and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(
                self.q_coin.parameters(), self.q_coin_target.parameters(), self.tau
            )
            polyak_update(
                self.q_pi.parameters(), self.q_pi_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        if self._n_calls % self.bonus_update_interval == 0:
            self.cumulative_bonus += self.bonus

    def _weighted_mse_loss(self, predictions, target, weight):
        return th.mean(weight * (predictions - target) ** 2)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.q_pi_policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        self._update_learning_rate(self.q_pi_policy.optimizer)

        losses_coin, losses_pi = [], []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # Q-coin
            # Get current Q-coin estimates
            current_q_coin = self.q_coin(replay_data.observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_qa_coin = th.gather(
                current_q_coin, dim=1, index=replay_data.actions.long()
            )

            # Q-pi
            # Get current Q-pi estimates
            current_q_pi = self.q_pi(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_qa_pi = th.gather(
                current_q_pi, dim=1, index=replay_data.actions.long()
            )

            # with th.no_grad():
            #     # Q-coin
            #     # Compute the next Q-coin values using the target network
            #     next_q_coin = self.q_coin_target(replay_data.next_observations)
            #     # Follow greedy policy: use the one with the highest value
            #     next_q_coin, _ = next_q_coin.max(dim=1)
            #     # Avoid potential broadcast issue
            #     next_q_coin = next_q_coin.reshape(-1, 1)
            #     # Add bonus to reward: r-b
            #     coin_rewards = replay_data.rewards - self.cumulative_bonus
            #     # 1-step TD target
            #     target_qa_coin = (
            #         coin_rewards + (1 - replay_data.dones) * self.gamma * next_q_coin
            #     )

            #     # Q-pi
            #     # Compute the next Q-pi values using the target network
            #     next_q_pi = self.q_pi_target(replay_data.next_observations)
            #     # Follow pi: use the one from the policy
            #     next_qa_pi = th.gather(
            #         next_q_pi, dim=1, index=replay_data.next_actions.long()
            #     )
            #     # next_q_pi_values, _ = next_q_pi_values.max(dim=1)
            #     # Avoid potential broadcast issue
            #     # 1-step TD target
            #     target_qa_pi = (
            #         replay_data.rewards
            #         + (1 - replay_data.dones) * self.gamma * next_qa_pi
            #     )

            #     # Target Q-value estimates
            #     target_q_pi = self.q_pi_target(replay_data.observations)

            #     # Q-values of other actions must remain unchanged
            #     target_q_pi = target_q_pi.index_put_(
            #         tuple(
            #             th.column_stack(
            #                 (
            #                     th.arange(replay_data.actions.shape[0]).to(self.device),
            #                     replay_data.actions.squeeze(-1),
            #                 )
            #             )
            #             .long()
            #             .t()
            #             .to(self.device)
            #         ),
            #         target_qa_pi.squeeze(-1),
            #     )

            #     # COIN regret bound update stuff
            #     prior_q_values = self.q_prior(replay_data.observations)
            #     prior_q_values_max, prior_actions = prior_q_values.max(dim=1)
            #     is_prior_action = (
            #         replay_data.actions.squeeze(-1) == prior_actions
            #     ).float()

            #     # Q-value of the prior greedy action
            #     # prior_qa_values = th.gather(
            #     #     prior_q_values, dim=1, index=replay_data.actions.long()
            #     # )

            #     # Regret gap to close
            #     delta = prior_q_values_max - self.cumulative_bonus / (1 - self.gamma)
            #     # delta = prior_q_values_max - (
            #     #     self.num_timesteps / self._total_timesteps
            #     # ) * (self.cumulative_bonus / (1 - self.gamma))
            #     eta = (
            #         prior_q_values_max - current_qa_pi.squeeze(-1)
            #     ) / self.regret_bound

            #     # Goal Q-value
            #     target_regret = delta

            #     # If the (avg.) return is more than the prior action q
            #     is_better_action = th.gt(
            #         current_qa_pi.squeeze(-1), prior_q_values_max
            #     ).float()

            #     # If action is from prior policy, use TD target
            #     # Elif action is not from prior policy and has no regret, use TD target
            #     # Else take the min of the TD and regret target
            #     target_qa_coin = target_qa_coin.squeeze(-1)

            #     target_qa_coin = (
            #         is_prior_action * target_qa_coin
            #         + (1 - is_prior_action) * is_better_action * target_qa_coin
            #         + (1 - is_prior_action)
            #         * (1 - is_better_action)
            #         * th.minimum(target_qa_coin, target_regret)
            #     ).to(self.device)

            #     # Target Q-value estimates
            #     target_q_coin = self.q_coin_target(replay_data.observations)

            #     target_q_coin = target_q_coin.index_put_(
            #         tuple(
            #             th.column_stack(
            #                 (
            #                     th.arange(replay_data.actions.shape[0]).to(self.device),
            #                     replay_data.actions.squeeze(-1),
            #                 )
            #             )
            #             .long()
            #             .t()
            #             .to(self.device)
            #         ),
            #         target_qa_coin,
            #     )

            #     mse_weights_actions = (
            #         is_prior_action * th.ones_like(target_qa_coin)
            #         + (1 - is_prior_action)
            #         * is_better_action
            #         * th.ones_like(target_qa_coin)
            #         + (1 - is_prior_action)
            #         * (1 - is_better_action)
            #         * eta
            #         * th.ones_like(target_qa_coin)
            #     ).to(self.device)

            #     mse_weights = th.ones_like(target_q_coin).to(self.device)
            #     mse_weights = mse_weights.index_put_(
            #         tuple(
            #             th.column_stack(
            #                 (
            #                     th.arange(replay_data.actions.shape[0]).to(self.device),
            #                     replay_data.actions.squeeze(-1),
            #                 )
            #             )
            #             .long()
            #             .t()
            #             .to(self.device)
            #         ),
            #         mse_weights_actions,
            #     )
            with th.no_grad():
                # Q-coin
                # Compute the next Q-coin values using the target network
                next_q_coin = self.q_coin_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_coin, _ = next_q_coin.max(dim=1)
                # Avoid potential broadcast issue
                next_q_coin = next_q_coin.reshape(-1, 1)
                # Add bonus to reward: r-b
                coin_rewards = replay_data.rewards - self.cumulative_bonus
                # 1-step TD target
                target_qa_coin = (
                    coin_rewards + (1 - replay_data.dones) * self.gamma * next_q_coin
                )

                # Q-pi
                # Compute the next Q-pi values using the target network
                next_q_pi = self.q_pi_target(replay_data.next_observations)
                # Follow pi: use the one from the policy
                next_qa_pi = th.gather(
                    next_q_pi, dim=1, index=replay_data.next_actions.long()
                )
                # next_q_pi_values, _ = next_q_pi_values.max(dim=1)
                # Avoid potential broadcast issue
                # 1-step TD target
                target_qa_pi = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_qa_pi
                )

                # Target Q-value estimates
                target_q_pi = self.q_pi_target(replay_data.observations)

                # Q-values of other actions must remain unchanged
                target_q_pi = target_q_pi.index_put_(
                    tuple(
                        th.column_stack(
                            (
                                th.arange(replay_data.actions.shape[0]).to(self.device),
                                replay_data.actions.squeeze(-1),
                            )
                        )
                        .long()
                        .t()
                        .to(self.device)
                    ),
                    target_qa_pi.squeeze(-1),
                )

                # Target Q-value estimates
                target_q_coin = self.q_coin_target(replay_data.observations)

                target_q_coin = target_q_coin.index_put_(
                    tuple(
                        th.column_stack(
                            (
                                th.arange(replay_data.actions.shape[0]).to(self.device),
                                replay_data.actions.squeeze(-1),
                            )
                        )
                        .long()
                        .t()
                        .to(self.device)
                    ),
                    target_qa_coin.squeeze(-1),
                )

                mse_weights = th.ones_like(target_q_coin).to(self.device)

            # Compute Huber loss (less sensitive to outliers)
            # loss_coin = F.smooth_l1_loss(current_q_coin_values, target_q_coin_values)
            # print(f"weights: {mse_weights}")
            # print(f"cur q coin: {current_q_coin}")
            # print(f"targ q coin: {target_q_coin}")

            # loss_coin = self._weighted_mse_loss(
            #     current_q_coin, target_q_coin, mse_weights
            # )
            loss_coin = F.smooth_l1_loss(current_qa_coin, target_qa_coin)
            losses_coin.append(loss_coin.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss_coin.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Compute Huber loss
            # loss_pi = F.smooth_l1_loss(current_q_pi, target_q_pi)
            loss_pi = F.mse_loss(current_q_pi, target_q_pi)
            losses_pi.append(loss_pi.item())

            # Optimize the policy
            self.q_pi_policy.optimizer.zero_grad()
            loss_pi.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(
                self.q_pi_policy.parameters(), self.max_grad_norm
            )
            self.q_pi_policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss_coin", np.mean(losses_coin))
        self.logger.record("train/loss_pi", np.mean(losses_pi))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        action, state = self.policy.predict(
            observation, state, episode_start, deterministic
        )
        return action, state

    def learn(
        self: SelfCOIN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfCOIN:
        return super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            progress_bar,
        )

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps to rollout the prior policy to estimate
            its return.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action according to policy
        unscaled_action, _ = self.predict(self._last_obs, deterministic=True)

        # Discrete case, no need to normalize or clip
        buffer_action = unscaled_action
        action = buffer_action
        return action, buffer_action

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # Next action
            buffer_new_actions, _ = self.predict(new_obs, deterministic=True)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # # Store data in replay buffer (normalized action and unnormalized observation)
            # self._store_transition(
            #     replay_buffer, buffer_actions, new_obs, rewards, dones, infos
            # )
            self._store_transition(
                replay_buffer,
                buffer_actions,
                new_obs,
                buffer_new_actions,
                rewards,
                dones,
                infos,
            )

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()

                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        if self.num_timesteps < learning_starts:
                            self.prior_return = safe_mean(
                                [ep_info["r"] for ep_info in self.ep_info_buffer]
                            )
                        else:
                            self.regret += max(
                                0, self.prior_return - self.ep_info_buffer[-1]["r"]
                            )
                        self.logger.record("coin/regret", self.regret)
                        self.logger.record(
                            "coin/b", self.cumulative_bonus, exclude="tensorboard"
                        )
                        self.logger.record(
                            "coin/B", self.regret_bound, exclude="tensorboard"
                        )

        callback.on_rollout_end()

        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        buffer_new_action: np.ndarray,
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param buffer_new_action: normalized next action
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(
                            next_obs[i, :]
                        )

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            buffer_new_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    @classmethod
    def load(
        cls,
        path,
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """
        Load the model from a zip-file
        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if (
            "policy_kwargs" in kwargs
            and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify new environments"
            )

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"]
            )
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        model._setup_prior()

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model
