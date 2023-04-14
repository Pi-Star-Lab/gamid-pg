from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import (
    get_linear_fn,
    get_parameters_by_name,
    polyak_update,
)
from stable_baselines3.common.policies import BasePolicy
from rl_zoo3.ddpg2.policies import DDPG2Policy, MlpPolicy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

SelfDDPG2 = TypeVar("SelfDDPG2", bound="DDPG2")


class DDPG2(DDPG):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DDPG2Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.001,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = ...,
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        temperature_initial: float = 0.9,
        temperature_final: float = 0.5,
        temperature_fraction: float = 0.5,
        n_actors: int = 2,
        actors_loss_fn: Optional[str] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):

        # Coefficients for actor 2 loss components
        self.n_actors = n_actors
        self.actors_loss_fn = actors_loss_fn
        self.actor_selection_probs = np.array(
            [(1.0 / n_actors) for _ in range(n_actors)]
        )
        self.temperature_initial = temperature_initial
        self.temperature_final = temperature_final
        self.temperature_fraction = temperature_fraction

        self._n_calls = 0

        self.temperature = 0.0
        self.temperature_schedule = None

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

        if "n_actors" not in self.policy_kwargs:
            self.policy_kwargs["n_actors"] = n_actors

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
        self.logger.record("train/temperature", self.temperature)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        actor_selection_probs: Optional[np.ndarray] = None,
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
            observation, state, episode_start, deterministic, actor_selection_probs
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
                actor_selection_probs=self.actor_selection_probs,
            )

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
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

        actor_losses, diversity_losses, critic_losses = [], [], []
        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # # Select action according to policy and add clipped noise
                # noise = replay_data.actions.clone().data.normal_(
                #     0, self.target_policy_noise
                # )
                # noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                # next_actions = (
                #     self.actor_target(replay_data.next_observations) + noise
                # ).clamp(-1, 1)

                next_actions_all = th.stack(
                    self.actor_target(replay_data.next_observations), dim=0
                )

                next_q_values_all = th.cat(
                    [
                        th.cat(
                            self.critic_target(
                                replay_data.next_observations, next_actions
                            )
                        )
                        for next_actions in next_actions_all
                    ],
                    dim=1,
                ).to(self.device)

                next_actors = th.argmax(next_q_values_all, dim=1).unsqueeze(dim=1)
                next_actors = next_actors.expand(
                    -1, self.action_space.shape[0]
                ).unsqueeze(dim=1)
                # next_actors = th.cat((next_actors, next_actors), dim=1).unsqueeze(dim=1)

                next_actions_all = th.stack(
                    self.actor_target(replay_data.next_observations), dim=1
                )

                next_actions = th.gather(
                    next_actions_all, dim=1, index=next_actors.long()
                ).squeeze(1)

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
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

                # For the actor losses
                mu_all_target = self.actor_target(replay_data.observations)
                # print(mu_all_target)
                # mu_mean = mu_all_target.mean(dim=0)

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
                mu_all = self.actor(replay_data.observations)

                dpg_loss, diversity_loss = 0, 0
                for targ_idx in range(self.n_actors):
                    # Compute actor loss
                    dpg_loss += -self.critic.q1_forward(
                        replay_data.observations, mu_all[targ_idx]
                    ).mean()
                    for idx in range(self.n_actors):
                        if targ_idx == idx:
                            continue
                        # Compute diversity loss
                        diversity_loss += (1.0 / (self.n_actors - 1)) * self.mse_loss(
                            mu_all_target[targ_idx], mu_all[idx]
                        )

                actor_loss = th.add(
                    (1 - self.temperature) * dpg_loss, self.temperature * diversity_loss
                )

                actor_losses.append(actor_loss.item())
                diversity_losses.append(diversity_loss.item())

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
        self.logger.record("train/critic_loss", np.mean(critic_losses))
