from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    get_linear_fn,
    get_parameters_by_name,
    polyak_update,
)
from stable_baselines3.common.policies import BasePolicy
from rl_zoo3.gamid.policies import GamidPolicy, MlpPolicy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy import linalg

from torch.distributions import Categorical

SelfGamid = TypeVar("SelfGamid", bound="Gamid")


class Gamid(DDPG):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[GamidPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.001,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 1,
        target_policy_noise: float = 0.1,
        target_noise_clip: float = 0.0,
        n_actors: int = 1,
        n_critics: int = 1,
        temperature_initial: float = 0.9,
        temperature_final: float = 0.5,
        temperature_fraction: float = 0.5,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 0.5,
        exploration_final_eps: float = 0.1,
        actors_loss_fn: Optional[str] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = False,
    ):

        # Coefficients for actor 2 loss components
        self.n_actors = n_actors
        self.actors_loss_fn = actors_loss_fn

        self._n_calls = 0

        self.temperature_initial = temperature_initial
        self.temperature_final = temperature_final
        self.temperature_fraction = temperature_fraction

        self.temperature = 0.0
        self.temperature_schedule = None

        # Epsilon greedy selection of actors
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction

        self.exploration_rate = 0.0
        self.exploration_schedule = None

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.policy_kwargs["n_actors"] = n_actors
        self.policy_kwargs["n_critics"] = n_critics

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None

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
            self.replay_buffer = self.replay_buffer_class(
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

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(
            self.actor_target, ["running_"]
        )
        self.critic_batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )

        self.temperature_schedule = get_linear_fn(
            self.temperature_initial, self.temperature_final, self.temperature_fraction
        )

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        self.greedy_actor_count = th.zeros(self.n_actors).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1

        self.temperature = self.temperature_schedule(self._current_progress_remaining)
        self.exploration_rate = self.exploration_schedule(
            self._current_progress_remaining
        )

        actions = self.policy._predict(
            th.tensor(self._last_obs).to(self.device), deterministic=True
        )
        q_values = []
        for action in actions:
            q_values.append(
                self.policy.critic_target(
                    th.tensor(self._last_obs).to(self.device),
                    action,
                )
            )
        actions = th.stack(list(actions), dim=0)
        # Convert list to tensor
        q_values = th.FloatTensor(q_values).squeeze(dim=1)
        # Min Q from among multiple critics
        q_values, _ = th.min(q_values, dim=-1)

        _, actor_idx = th.max(q_values, dim=-1)
        actor_idx = actor_idx.to(self.device)

        self.greedy_actor_count[actor_idx] += 1
        actor_spread = Categorical(
            probs=self.greedy_actor_count / self.greedy_actor_count.sum()
        ).entropy()

        self.logger.record("train/temperature", self.temperature)
        self.logger.record("train/epsilon", self.exploration_rate)
        self.logger.record("train/actor_spread", actor_spread.item())

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        exploration_rate: float = 0,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :param actor_selection_probs: Probabilities of selecting actors.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(
            observation, state, episode_start, deterministic, exploration_rate
        )

    def _sample_action(
        self,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ):
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup phase
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(
                self._last_obs,
                deterministic=False,
                exploration_rate=self.exploration_rate,
            )

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            # scaled_action = np.power(scaled_action, 2) - np.power(scaled_action, 3) + 1
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action

        return action, buffer_action

    def log_loss(self, action_1, action_2):
        """
        Loss: Log distance between actions.
        """
        # loss = th.exp(-th.norm(action_1 - action_2, dim=0) ** 2 / 0.01).mean()
        loss = -th.log(th.norm(action_1 - action_2, dim=1) + 0.01).mean()
        return loss

    def mse_loss(self, action_1, action_2):
        """
        Loss: MSE between actions.
        """
        loss = -th.norm(action_1 - action_2, p=2, dim=1).mean()
        return loss

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, diversity_losses, critic_losses, distances_means = [], [], [], []
        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # Select action according to policy
                next_actions_all = th.stack(
                    self.actor_target(replay_data.next_observations), dim=0
                )

                # print(f"{next_actions_all}")
                # input()

                next_q_values_all = th.stack(
                    [
                        th.cat(
                            self.critic_target(
                                replay_data.next_observations, next_actions
                            ),
                            dim=1,
                        )
                        for next_actions in next_actions_all
                    ],
                    dim=0,
                ).to(self.device)

                # print(f"{next_q_values_all}")
                # input()

                next_q_values_all, _ = th.min(next_q_values_all, dim=-1)
                # print(f"{next_q_values_all}")
                # input()

                next_actors = th.argmax(next_q_values_all, dim=0).unsqueeze(dim=1)
                # print(f"{next_actors}")
                # input()

                next_actors = next_actors.expand(
                    -1, self.action_space.shape[0]
                ).unsqueeze(dim=1)
                # next_actors = th.cat((next_actors, next_actors), dim=1).unsqueeze(dim=1)
                # print(f"{next_actors}")
                # input()

                next_actions_all = th.stack(
                    self.actor_target(replay_data.next_observations), dim=1
                )
                # print(f"{next_actions_all}")
                # input()

                next_actions = th.gather(
                    next_actions_all, dim=1, index=next_actors.long()
                ).squeeze(1)
                # print(f"{next_actions}")
                # input()

                noise = replay_data.actions.clone().data.normal_(
                    0, self.target_policy_noise
                )
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (next_actions + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                # print(f"{next_q_values}")
                # input()

                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # print(f"{next_q_values}")
                # input()

                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

                # For the actor losses
                # mu_all_target = self.actor_target(replay_data.observations)
                mu_all_target = self.actor(replay_data.observations)

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # mu_all = self.actor(replay_data.observations)

                # dpg_loss, diversity_loss, distances_mean = 0, 0, 0
                # for targ_idx in range(self.n_actors):
                #     # Compute actor loss
                #     dpg_loss += -self.critic.q1_forward(
                #         replay_data.observations, mu_all[targ_idx]
                #     ).mean()
                #     for idx in range(self.n_actors):
                #         if targ_idx == idx:
                #             continue
                #         # # Compute diversity loss
                #         # diversity_loss += (1.0 / (self.n_actors - 1)) * self.mse_loss(
                #         #     mu_all_target[targ_idx], mu_all[idx]
                #         # )
                #         # diversity_loss += (1.0 / (self.n_actors - 1)) * th.exp(
                #         #     th.norm
                #         # (mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
                #         # ).mean()
                #         distances_mean += th.norm(
                #             mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1
                #         ).mean()
                #         # diversity_loss += th.exp(
                #         #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
                #         # ).mean()
                #         # diversity_loss += -th.log(
                #         #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
                #         #     ** 2
                #         # ).mean()
                #         diversity_loss += (
                #             th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
                #             ** 2
                #         ).mean()

                mu_all = self.actor(replay_data.observations)
                mu_dpg = self.actor(replay_data.observations)

                with th.no_grad():
                    q_values = []
                    for mu in mu_all:
                        # Greedy actor
                        q_values.append(
                            self.critic.q1_forward(replay_data.observations, mu)
                        )
                    # Convert list to tensor
                    q_values = th.tensor(th.stack(q_values)).squeeze()
                    _, greedy_actor = th.max(q_values, dim=-2)
                    greedy_actor = greedy_actor.to(self.device)

                mu_all = th.stack(list(mu_all), dim=0).to(self.device)

                with th.no_grad():
                    mu_all_target = th.stack(list(mu_all_target), dim=0).to(self.device)

                # mu_all_target[greedy_actor, th.arange(self.batch_size)] = 2 * th.ones(
                #     self.action_space.shape[0]
                # ).to(self.device)
                # mu_all[greedy_actor, th.arange(self.batch_size)] = 2 * th.ones(
                #     self.action_space.shape[0]
                # ).to(self.device)

                dpg_loss, diversity_loss, distances_mean = 0, 0, 0
                for targ_idx in range(self.n_actors):
                    # Compute actor loss
                    dpg_loss += -self.critic.q1_forward(
                        replay_data.observations, mu_dpg[targ_idx]
                    ).mean()
                    for idx in range(self.n_actors):
                        if targ_idx == idx:
                            continue
                        # If idx or targ_idx == greedy_actor
                        # # Compute diversity loss
                        # diversity_loss += (1.0 / (self.n_actors - 1)) * self.mse_loss(
                        #     mu_all_target[targ_idx], mu_all[idx]
                        # )
                        # diversity_loss += (1.0 / (self.n_actors - 1)) * th.exp(
                        #     th.norm
                        # (mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
                        # ).mean()
                        distances_mean += th.norm(
                            mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1
                        ).mean()

                        # diversity_loss += th.exp(
                        #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
                        # ).mean()
                        # diversity_loss += -th.log(
                        #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
                        #     ** 2
                        # ).mean()
                        # diversity_loss += (
                        #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
                        #     ** 2
                        # ).mean()
                        mask = (
                            th.logical_not(
                                th.logical_or(
                                    greedy_actor.eq(idx), greedy_actor.eq(targ_idx)
                                )
                            )
                            .float()
                            .to(self.device)
                        )

                        # diversity_loss += (
                        #     mask
                        #     * 100
                        #     * th.exp(
                        #         -th.norm(
                        #             (mu_all_target[targ_idx] - mu_all[idx]),
                        #             p=2,
                        #             dim=1,
                        #         )
                        #     )
                        # ).mean()
                        diversity_loss += (
                            mask
                            * th.norm(
                                (mu_all_target[targ_idx] - mu_all[idx]),
                                p=2,
                                dim=1,
                            )
                        ).mean()

                distances_mean /= self.n_actors
                # diversity_loss = -th.log(1 / diversity_loss)

                actor_loss = th.add(
                    (1 - self.temperature) * dpg_loss, self.temperature * diversity_loss
                )

                actor_losses.append(actor_loss.item())
                diversity_losses.append(diversity_loss.item())
                distances_means.append(distances_mean.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats,
                    self.actor_batch_norm_stats_target,
                    1.0,
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/diversity_loss", np.mean(diversity_losses))
            self.logger.record("train/distances_mean", np.mean(distances_means))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfGamid,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "Gamid",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfGamid:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


# from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

# import numpy as np
# import torch as th
# from gym import spaces
# from torch.nn import functional as F

# from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
# from stable_baselines3.common.noise import ActionNoise
# from stable_baselines3 import DDPG
# from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
# from stable_baselines3.common.utils import (
#     get_linear_fn,
#     get_parameters_by_name,
#     polyak_update,
# )
# from stable_baselines3.common.policies import BasePolicy
# from rl_zoo3.gamid.policies import GamidPolicy, MlpPolicy
# from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

# from sklearn.mixture import GaussianMixture
# from scipy.stats import multivariate_normal
# from scipy import linalg

# from torch.distributions import Categorical

# SelfGamid = TypeVar("SelfGamid", bound="Gamid")


# class Gamid(DDPG):

#     policy_aliases: Dict[str, Type[BasePolicy]] = {
#         "MlpPolicy": MlpPolicy,
#     }

#     def __init__(
#         self,
#         policy: Union[str, Type[GamidPolicy]],
#         env: Union[GymEnv, str],
#         learning_rate: Union[float, Schedule] = 0.001,
#         buffer_size: int = 1000000,
#         learning_starts: int = 100,
#         batch_size: int = 100,
#         tau: float = 0.005,
#         gamma: float = 0.99,
#         train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
#         gradient_steps: int = -1,
#         action_noise: Optional[ActionNoise] = None,
#         replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
#         replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
#         optimize_memory_usage: bool = False,
#         policy_delay: int = 1,
#         target_policy_noise: float = 0.1,
#         target_noise_clip: float = 0.0,
#         n_actors: int = 1,
#         n_critics: int = 1,
#         temperature_initial: float = 0.9,
#         temperature_final: float = 0.5,
#         temperature_fraction: float = 0.5,
#         exploration_fraction: float = 0.1,
#         exploration_initial_eps: float = 0.5,
#         exploration_final_eps: float = 0.1,
#         actors_loss_fn: Optional[str] = None,
#         tensorboard_log: Optional[str] = None,
#         policy_kwargs: Optional[Dict[str, Any]] = None,
#         verbose: int = 0,
#         seed: Optional[int] = None,
#         device: Union[th.device, str] = "auto",
#         _init_setup_model: bool = False,
#     ):

#         # Coefficients for actor 2 loss components
#         self.n_actors = n_actors
#         self.actors_loss_fn = actors_loss_fn

#         self._n_calls = 0

#         self.temperature_initial = temperature_initial
#         self.temperature_final = temperature_final
#         self.temperature_fraction = temperature_fraction

#         self.temperature = 0.0
#         self.temperature_schedule = None

#         # Epsilon greedy selection of actors
#         self.exploration_initial_eps = exploration_initial_eps
#         self.exploration_final_eps = exploration_final_eps
#         self.exploration_fraction = exploration_fraction

#         self.exploration_rate = 0.0
#         self.exploration_schedule = None

#         self.cagrad_c = self.temperature_initial

#         super().__init__(
#             policy=policy,
#             env=env,
#             learning_rate=learning_rate,
#             buffer_size=buffer_size,
#             learning_starts=learning_starts,
#             batch_size=batch_size,
#             tau=tau,
#             gamma=gamma,
#             train_freq=train_freq,
#             gradient_steps=gradient_steps,
#             action_noise=action_noise,
#             replay_buffer_class=replay_buffer_class,
#             replay_buffer_kwargs=replay_buffer_kwargs,
#             optimize_memory_usage=optimize_memory_usage,
#             tensorboard_log=tensorboard_log,
#             policy_kwargs=policy_kwargs,
#             verbose=verbose,
#             seed=seed,
#             device=device,
#             _init_setup_model=False,
#         )

#         self.policy_delay = policy_delay
#         self.target_noise_clip = target_noise_clip
#         self.target_policy_noise = target_policy_noise

#         self.policy_kwargs["n_actors"] = n_actors
#         self.policy_kwargs["n_critics"] = n_critics

#         self.actor, self.actor_target = None, None
#         self.critic, self.critic_target = None, None

#         self._setup_model()

#     def _setup_model(self) -> None:
#         self._setup_lr_schedule()
#         self.set_random_seed(self.seed)

#         # Use DictReplayBuffer if needed
#         if self.replay_buffer_class is None:
#             if isinstance(self.observation_space, spaces.Dict):
#                 self.replay_buffer_class = DictReplayBuffer
#             else:
#                 self.replay_buffer_class = ReplayBuffer

#         elif self.replay_buffer_class == HerReplayBuffer:
#             assert (
#                 self.env is not None
#             ), "You must pass an environment when using `HerReplayBuffer`"

#             # If using offline sampling, we need a classic replay buffer too
#             if self.replay_buffer_kwargs.get("online_sampling", True):
#                 replay_buffer = None
#             else:
#                 replay_buffer = DictReplayBuffer(
#                     self.buffer_size,
#                     self.observation_space,
#                     self.action_space,
#                     device=self.device,
#                     optimize_memory_usage=self.optimize_memory_usage,
#                 )

#             self.replay_buffer = HerReplayBuffer(
#                 self.env,
#                 self.buffer_size,
#                 device=self.device,
#                 replay_buffer=replay_buffer,
#                 **self.replay_buffer_kwargs,
#             )

#         if self.replay_buffer is None:
#             self.replay_buffer = self.replay_buffer_class(
#                 self.buffer_size,
#                 self.observation_space,
#                 self.action_space,
#                 device=self.device,
#                 n_envs=self.n_envs,
#                 optimize_memory_usage=self.optimize_memory_usage,
#                 **self.replay_buffer_kwargs,
#             )

#         self.policy = self.policy_class(  # pytype:disable=not-instantiable
#             self.observation_space,
#             self.action_space,
#             self.lr_schedule,
#             **self.policy_kwargs,  # pytype:disable=not-instantiable
#         )
#         self.policy = self.policy.to(self.device)

#         # Convert train freq parameter to TrainFreq object
#         self._convert_train_freq()

#         self._create_aliases()
#         # Running mean and running var
#         self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
#         self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
#         self.actor_batch_norm_stats_target = get_parameters_by_name(
#             self.actor_target, ["running_"]
#         )
#         self.critic_batch_norm_stats_target = get_parameters_by_name(
#             self.critic_target, ["running_"]
#         )

#         self.temperature_schedule = get_linear_fn(
#             self.temperature_initial, self.temperature_final, self.temperature_fraction
#         )

#         self.exploration_schedule = get_linear_fn(
#             self.exploration_initial_eps,
#             self.exploration_final_eps,
#             self.exploration_fraction,
#         )

#         self.greedy_actor_count = th.zeros(self.n_actors).to(self.device)

#     def _create_aliases(self) -> None:
#         self.actor = self.policy.actor
#         self.actor_target = self.policy.actor_target
#         self.critic = self.policy.critic
#         self.critic_target = self.policy.critic_target

#     def _compute_gradient(
#         self, losses, retain_graph: bool = True, allow_unused: bool = False
#     ):
#         """Compute the gradient."""
#         grad = []
#         for loss in losses:
#             grad.append(
#                 tuple(
#                     _grad.contiguous()
#                     for _grad in th.autograd.grad(
#                         loss,
#                         self.actor.parameters(),
#                         retain_graph=retain_graph,
#                         allow_unused=allow_unused,
#                     )
#                 )
#             )
#         return grad

#     def _set_gradient(self, grads):
#         """Set the gradients of the policy."""
#         idx = 0
#         for param in self.actor.parameters():
#             if param.requires_grad:
#                 num_param_elements = th.numel(param.grad)
#                 modified_grad = grads[idx : idx + num_param_elements]
#                 modified_grad = modified_grad.view_as(param.grad)

#                 param.grad = modified_grad  # Set the modified gradient

#                 idx += num_param_elements

#     def cagrad(self, grad_vec):
#         """Conflict-Averse Gradient Descent (CAGrad)."""
#         grads = grad_vec
#         grad_0 = grad_vec[0] + grad_vec[1]

#         w = th.ones(2, 1, requires_grad=True)
#         w_opt = th.optim.SGD([w], lr=2, momentum=0.5)

#         c = (grad_0.norm() * self.cagrad_c).to(grad_vec.device)

#         w_best = None
#         obj_best = np.inf
#         for i in range(21):
#             w_opt.zero_grad()
#             ww = th.softmax(w, 0).to(grad_vec.device)

#             gw = ww.t().mm(grad_vec)
#             g0 = grad_0.view(1, -1).to(grad_vec.device)

#             obj = (gw.mm(g0.t()) + c * gw.norm()).sum()

#             if obj.item() < obj_best:
#                 obj_best = obj.item()
#                 w_best = w.clone()
#             if i < 20:
#                 obj.backward()
#                 w_opt.step()

#         # print(f"w_best: {w_best}")
#         ww = th.softmax(w_best, 0).to(grad_vec.device)
#         gw = (ww.t().mm(grad_vec)).to(grad_vec.device)

#         gw_norm = gw.norm()

#         lmbda = gw_norm / c
#         g = (grad_0 + gw / lmbda).view(-1, 1).to(grads.device)

#         return g, ww[1].item()

#     def _on_step(self) -> None:
#         """
#         Update the exploration rate and target network if needed.
#         This method is called in ``collect_rollouts()`` after each step in the environment.
#         """
#         self._n_calls += 1

#         self.temperature = self.temperature_schedule(self._current_progress_remaining)
#         self.exploration_rate = self.exploration_schedule(
#             self._current_progress_remaining
#         )

#         actions = self.policy._predict(
#             th.tensor(self._last_obs).to(self.device), deterministic=True
#         )
#         q_values = []
#         for action in actions:
#             q_values.append(
#                 self.policy.critic_target(
#                     th.tensor(self._last_obs).to(self.device),
#                     action,
#                 )
#             )
#         actions = th.stack(list(actions), dim=0)
#         # Convert list to tensor
#         q_values = th.FloatTensor(q_values).squeeze(dim=1)
#         # Min Q from among multiple critics
#         q_values, _ = th.min(q_values, dim=-1)

#         _, actor_idx = th.max(q_values, dim=-1)
#         actor_idx = actor_idx.to(self.device)

#         self.greedy_actor_count[actor_idx] += 1
#         actor_spread = Categorical(
#             probs=self.greedy_actor_count / self.greedy_actor_count.sum()
#         ).entropy()

#         self.logger.record("train/temperature", self.temperature)
#         self.logger.record("train/epsilon", self.exploration_rate)
#         self.logger.record("train/actor_spread", actor_spread.item())

#     def predict(
#         self,
#         observation: Union[np.ndarray, Dict[str, np.ndarray]],
#         state: Optional[Tuple[np.ndarray, ...]] = None,
#         episode_start: Optional[np.ndarray] = None,
#         deterministic: bool = False,
#         exploration_rate: float = 0,
#     ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
#         """
#         Get the policy action from an observation (and optional hidden state).
#         Includes sugar-coating to handle different observations (e.g. normalizing images).

#         :param observation: the input observation
#         :param state: The last hidden states (can be None, used in recurrent policies)
#         :param episode_start: The last masks (can be None, used in recurrent policies)
#             this correspond to beginning of episodes,
#             where the hidden states of the RNN must be reset.
#         :param deterministic: Whether or not to return deterministic actions.
#         :param actor_selection_probs: Probabilities of selecting actors.
#         :return: the model's action and the next hidden state
#             (used in recurrent policies)
#         """
#         return self.policy.predict(
#             observation, state, episode_start, deterministic, exploration_rate
#         )

#     def _sample_action(
#         self,
#         learning_starts: int,
#         action_noise=None,
#         n_envs: int = 1,
#     ):
#         """
#         Sample an action according to the exploration policy.
#         This is either done by sampling the probability distribution of the policy,
#         or sampling a random action (from a uniform distribution over the action space)
#         or by adding noise to the deterministic output.

#         :param action_noise: Action noise that will be used for exploration
#             Required for deterministic policy (e.g. TD3). This can also be used
#             in addition to the stochastic policy for SAC.
#         :param learning_starts: Number of steps before learning for the warm-up phase.
#         :param n_envs:
#         :return: action to take in the environment
#             and scaled action that will be stored in the replay buffer.
#             The two differs when the action space is not normalized (bounds are not [-1, 1]).
#         """
#         # Select action randomly or according to policy
#         if self.num_timesteps < learning_starts and not (
#             self.use_sde and self.use_sde_at_warmup
#         ):
#             # Warmup phase
#             unscaled_action = np.array(
#                 [self.action_space.sample() for _ in range(n_envs)]
#             )
#         else:
#             # Note: when using continuous actions,
#             # we assume that the policy uses tanh to scale the action
#             # We use non-deterministic action in the case of SAC, for TD3, it does not matter
#             unscaled_action, _ = self.predict(
#                 self._last_obs,
#                 deterministic=False,
#                 exploration_rate=self.exploration_rate,
#             )

#         # Rescale the action from [low, high] to [-1, 1]
#         if isinstance(self.action_space, spaces.Box):
#             scaled_action = self.policy.scale_action(unscaled_action)

#             # Add noise to the action (improve exploration)
#             if action_noise is not None:
#                 scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

#             # We store the scaled action in the buffer
#             buffer_action = scaled_action
#             # scaled_action = np.power(scaled_action, 2) - np.power(scaled_action, 3) + 1
#             action = self.policy.unscale_action(scaled_action)
#         else:
#             # Discrete case, no need to normalize or clip
#             buffer_action = unscaled_action
#             action = buffer_action

#         return action, buffer_action

#     def log_loss(self, action_1, action_2):
#         """
#         Loss: Log distance between actions.
#         """
#         # loss = th.exp(-th.norm(action_1 - action_2, dim=0) ** 2 / 0.01).mean()
#         loss = -th.log(th.norm(action_1 - action_2, dim=1) + 0.01).mean()
#         return loss

#     def mse_loss(self, action_1, action_2):
#         """
#         Loss: MSE between actions.
#         """
#         loss = -th.norm(action_1 - action_2, p=2, dim=1).mean()
#         return loss

#     def train(self, gradient_steps: int, batch_size: int = 100) -> None:
#         # Switch to train mode (this affects batch norm / dropout)
#         self.policy.set_training_mode(True)

#         # Update learning rate according to lr schedule
#         self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

#         actor_losses, diversity_losses, critic_losses, distances_means = [], [], [], []
#         for _ in range(gradient_steps):

#             self._n_updates += 1
#             # Sample replay buffer
#             replay_data = self.replay_buffer.sample(
#                 batch_size, env=self._vec_normalize_env
#             )

#             with th.no_grad():
#                 # Select action according to policy
#                 next_actions_all = th.stack(
#                     self.actor_target(replay_data.next_observations), dim=0
#                 )

#                 # print(f"{next_actions_all}")
#                 # input()

#                 next_q_values_all = th.stack(
#                     [
#                         th.cat(
#                             self.critic_target(
#                                 replay_data.next_observations, next_actions
#                             ),
#                             dim=1,
#                         )
#                         for next_actions in next_actions_all
#                     ],
#                     dim=0,
#                 ).to(self.device)

#                 # print(f"{next_q_values_all}")
#                 # input()

#                 next_q_values_all, _ = th.min(next_q_values_all, dim=-1)
#                 # print(f"{next_q_values_all}")
#                 # input()

#                 next_actors = th.argmax(next_q_values_all, dim=0).unsqueeze(dim=1)
#                 # print(f"{next_actors}")
#                 # input()

#                 next_actors = next_actors.expand(
#                     -1, self.action_space.shape[0]
#                 ).unsqueeze(dim=1)
#                 # next_actors = th.cat((next_actors, next_actors), dim=1).unsqueeze(dim=1)
#                 # print(f"{next_actors}")
#                 # input()

#                 next_actions_all = th.stack(
#                     self.actor_target(replay_data.next_observations), dim=1
#                 )
#                 # print(f"{next_actions_all}")
#                 # input()

#                 next_actions = th.gather(
#                     next_actions_all, dim=1, index=next_actors.long()
#                 ).squeeze(1)
#                 # print(f"{next_actions}")
#                 # input()

#                 noise = replay_data.actions.clone().data.normal_(
#                     0, self.target_policy_noise
#                 )
#                 noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
#                 next_actions = (next_actions + noise).clamp(-1, 1)

#                 # Compute the next Q-values: min over all critics targets
#                 next_q_values = th.cat(
#                     self.critic_target(replay_data.next_observations, next_actions),
#                     dim=1,
#                 )
#                 # print(f"{next_q_values}")
#                 # input()

#                 next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
#                 # print(f"{next_q_values}")
#                 # input()

#                 target_q_values = (
#                     replay_data.rewards
#                     + (1 - replay_data.dones) * self.gamma * next_q_values
#                 )

#                 # For the actor losses
#                 # mu_all_target = self.actor_target(replay_data.observations)
#                 mu_all_target = self.actor(replay_data.observations)

#             # Get current Q-values estimates for each critic network
#             current_q_values = self.critic(
#                 replay_data.observations, replay_data.actions
#             )

#             # Compute critic loss
#             critic_loss = sum(
#                 F.mse_loss(current_q, target_q_values) for current_q in current_q_values
#             )
#             critic_losses.append(critic_loss.item())

#             # Optimize the critics
#             self.critic.optimizer.zero_grad()
#             critic_loss.backward()
#             self.critic.optimizer.step()

#             # Delayed policy updates
#             if self._n_updates % self.policy_delay == 0:
#                 # mu_all = self.actor(replay_data.observations)

#                 # dpg_loss, diversity_loss, distances_mean = 0, 0, 0
#                 # for targ_idx in range(self.n_actors):
#                 #     # Compute actor loss
#                 #     dpg_loss += -self.critic.q1_forward(
#                 #         replay_data.observations, mu_all[targ_idx]
#                 #     ).mean()
#                 #     for idx in range(self.n_actors):
#                 #         if targ_idx == idx:
#                 #             continue
#                 #         # # Compute diversity loss
#                 #         # diversity_loss += (1.0 / (self.n_actors - 1)) * self.mse_loss(
#                 #         #     mu_all_target[targ_idx], mu_all[idx]
#                 #         # )
#                 #         # diversity_loss += (1.0 / (self.n_actors - 1)) * th.exp(
#                 #         #     th.norm
#                 #         # (mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
#                 #         # ).mean()
#                 #         distances_mean += th.norm(
#                 #             mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1
#                 #         ).mean()
#                 #         # diversity_loss += th.exp(
#                 #         #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
#                 #         # ).mean()
#                 #         # diversity_loss += -th.log(
#                 #         #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
#                 #         #     ** 2
#                 #         # ).mean()
#                 #         diversity_loss += (
#                 #             th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
#                 #             ** 2
#                 #         ).mean()

#                 mu_all = self.actor(replay_data.observations)
#                 mu_dpg = self.actor(replay_data.observations)

#                 with th.no_grad():
#                     q_values = []
#                     for mu in mu_all:
#                         # Greedy actor
#                         q_values.append(
#                             self.critic.q1_forward(replay_data.observations, mu)
#                         )
#                     # Convert list to tensor
#                     q_values = th.tensor(th.stack(q_values)).squeeze()
#                     _, greedy_actor = th.max(q_values, dim=-2)
#                     greedy_actor = greedy_actor.to(self.device)

#                 mu_all = th.stack(list(mu_all), dim=0).to(self.device)

#                 with th.no_grad():
#                     mu_all_target = th.stack(list(mu_all_target), dim=0).to(self.device)

#                 # mu_all_target[greedy_actor, th.arange(self.batch_size)] = 2 * th.ones(
#                 #     self.action_space.shape[0]
#                 # ).to(self.device)
#                 # mu_all[greedy_actor, th.arange(self.batch_size)] = 2 * th.ones(
#                 #     self.action_space.shape[0]
#                 # ).to(self.device)

#                 dpg_loss, diversity_loss, distances_mean = 0, 0, 0
#                 for targ_idx in range(self.n_actors):
#                     # Compute actor loss
#                     dpg_loss += -self.critic.q1_forward(
#                         replay_data.observations, mu_dpg[targ_idx]
#                     ).mean()
#                     for idx in range(self.n_actors):
#                         if targ_idx == idx:
#                             continue
#                         # If idx or targ_idx == greedy_actor
#                         # # Compute diversity loss
#                         # diversity_loss += (1.0 / (self.n_actors - 1)) * self.mse_loss(
#                         #     mu_all_target[targ_idx], mu_all[idx]
#                         # )
#                         # diversity_loss += (1.0 / (self.n_actors - 1)) * th.exp(
#                         #     th.norm
#                         # (mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
#                         # ).mean()
#                         distances_mean += th.norm(
#                             mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1
#                         ).mean()

#                         # diversity_loss += th.exp(
#                         #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
#                         # ).mean()
#                         # diversity_loss += -th.log(
#                         #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
#                         #     ** 2
#                         # ).mean()
#                         # diversity_loss += (
#                         #     th.norm(mu_all_target[targ_idx] - mu_all[idx], p=2, dim=1)
#                         #     ** 2
#                         # ).mean()
#                         mask = (
#                             th.logical_not(
#                                 th.logical_or(
#                                     greedy_actor.eq(idx), greedy_actor.eq(targ_idx)
#                                 )
#                             )
#                             .float()
#                             .to(self.device)
#                         )
#                         if targ_idx == idx:
#                             continue

#                         diversity_loss += (
#                             mask
#                             * 100
#                             * th.exp(
#                                 -th.norm(
#                                     (mu_all_target[targ_idx] - mu_all[idx]),
#                                     p=2,
#                                     dim=1,
#                                 )
#                             )
#                         ).mean()

#                 distances_mean /= self.n_actors
#                 # diversity_loss = -th.log(1 / diversity_loss)

#                 actor_loss = (1 - self.cagrad_c) * dpg_loss
#                 actor_losses.append(actor_loss.item())

#                 if self.cagrad_c == 0:
#                     self.actor.optimizer.zero_grad()
#                     actor_loss.backward()
#                 else:
#                     self.actor.optimizer.zero_grad()
#                     actor_loss.backward(retain_graph=True)

#                     self.actor.optimizer.zero_grad()

#                     diversity_loss = self.cagrad_c * diversity_loss
#                     diversity_loss.backward(retain_graph=True)

#                     grad = self._compute_gradient([diversity_loss, dpg_loss])
#                     grad_vec = th.cat(
#                         list(
#                             map(
#                                 lambda x: th.nn.utils.parameters_to_vector(x).unsqueeze(
#                                     0
#                                 ),
#                                 grad,
#                             )
#                         ),
#                         dim=0,
#                     )
#                     regularized_cagrad, _ = self.cagrad(grad_vec)
#                     # regularized_cagrad = th.clip(
#                     #     regularized_cagrad, min=None, max=100.0
#                     # )
#                     regularized_cagrad = th.nan_to_num(regularized_cagrad, nan=0.0)

#                     self._set_gradient(regularized_cagrad)

#                 diversity_losses.append(diversity_loss.item())
#                 distances_means.append(distances_mean.item())

#                 # Optimize the actor
#                 self.actor.optimizer.step()

#                 polyak_update(
#                     self.critic.parameters(), self.critic_target.parameters(), self.tau
#                 )
#                 polyak_update(
#                     self.actor.parameters(), self.actor_target.parameters(), self.tau
#                 )
#                 # Copy running stats, see GH issue #996
#                 polyak_update(
#                     self.critic_batch_norm_stats,
#                     self.critic_batch_norm_stats_target,
#                     1.0,
#                 )
#                 polyak_update(
#                     self.actor_batch_norm_stats,
#                     self.actor_batch_norm_stats_target,
#                     1.0,
#                 )

#         self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
#         if len(actor_losses) > 0:
#             self.logger.record("train/actor_loss", np.mean(actor_losses))
#             self.logger.record("train/diversity_loss", np.mean(diversity_losses))
#             self.logger.record("train/distances_mean", np.mean(distances_means))
#         self.logger.record("train/critic_loss", np.mean(critic_losses))

#     def learn(
#         self: SelfGamid,
#         total_timesteps: int,
#         callback: MaybeCallback = None,
#         log_interval: int = 4,
#         tb_log_name: str = "Gamid",
#         reset_num_timesteps: bool = True,
#         progress_bar: bool = False,
#     ) -> SelfGamid:
#         return super().learn(
#             total_timesteps=total_timesteps,
#             callback=callback,
#             log_interval=log_interval,
#             tb_log_name=tb_log_name,
#             reset_num_timesteps=reset_num_timesteps,
#             progress_bar=progress_bar,
#         )
