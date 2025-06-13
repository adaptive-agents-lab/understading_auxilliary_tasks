import os
import glob
from typing import Sequence
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from time import time

from aux_tasks.rl_types import Dataset, RLBatch


class ReplayBuffer(Dataset):
    def __init__(
        self,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        num_seeds: int,
        capacity: int,
        rollout_length: int,
        is_img_obs: bool = False,
        is_discrete_action: bool = False,
    ):
        self.is_image_obs = is_img_obs
        state_type = np.uint8 if is_img_obs else np.float32
        action_type = np.uint8 if is_discrete_action else np.float32
        # explicitly setting the arrays to 0 forces allocation and prevents later failures due to memory end
        self.states = np.empty((num_seeds, capacity, *state_shape), dtype=state_type)
        self.states.fill(0.0)
        self.actions = np.empty((num_seeds, capacity, *action_shape), dtype=action_type)
        self.actions.fill(0.0)
        self.rewards = np.empty(
            (num_seeds, capacity, 1),
            dtype=np.float32,
        )
        self.rewards.fill(0.0)
        self.next_states = np.empty(
            (num_seeds, capacity, *state_shape), dtype=state_type
        )
        self.next_states.fill(1.0)
        self.masks = np.empty(
            (num_seeds, capacity, 1),
            dtype=np.float32,
        )
        self.masks.fill(0.0)
        self.filled = 0
        self.rollout_length = rollout_length
        self.insert_index = 0
        self.capacity = capacity
        self.num_seeds = num_seeds
        self.last_saved = 0

    def insert(
        self,
        state: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        next_state: jax.Array,
    ):
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)
        done = np.array(done)

        self.states[:, self.insert_index] = state
        if action.dtype == np.int32:
            action = np.eye(self.actions.shape[-1])[action]
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index, 0] = reward
        self.masks[:, self.insert_index, 0] = np.logical_not(done)
        self.next_states[:, self.insert_index] = next_state

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def __len__(self):
        return max(0, self.filled - self.rollout_length)

    def sample(self, batch_size: int, key: jax.Array) -> RLBatch:
        indices = jax.random.randint(key, (batch_size,), 0, len(self))
        actions = []
        rewards = []
        next_states = []
        masks = []
        for i in range(self.rollout_length):
            actions.append(self.actions[:, indices + i])
            rewards.append(self.rewards[:, indices + i])
            next_states.append(self.next_states[:, indices + i])
            if i == 0:
                masks.append(self.masks[:, indices])
            else:
                masks.append(np.logical_and(self.masks[:, indices + i], masks[-1]))

        if self.num_seeds > 1:
            return RLBatch(
                state=jnp.array(self.states[:, indices], dtype=jnp.float32),
                action=jnp.stack(actions, axis=2),
                reward=jnp.stack(rewards, axis=2),
                next_state=jnp.stack(next_states, axis=2, dtype=jnp.float32),
                mask=jnp.stack(masks, axis=2),
            )
        else:
            return RLBatch(
                state=jnp.squeeze(
                    jnp.array(self.states[:, indices], dtype=jnp.float32), axis=0
                ),
                action=jnp.squeeze(jnp.stack(actions, axis=1), axis=0),
                reward=jnp.squeeze(jnp.stack(rewards, axis=1), axis=0),
                next_state=jnp.squeeze(
                    jnp.stack(next_states, axis=1, dtype=jnp.float32), axis=0
                ),
                mask=jnp.squeeze(jnp.stack(masks, axis=1), axis=0),
            )

    def get_dummy_batch(self) -> RLBatch:
        if self.num_seeds > 1:
            return RLBatch(
                state=jnp.array(self.states[:, 0]),
                action=jnp.array(self.actions[:, 0]),
                reward=jnp.array(self.rewards[:, 0]),
                next_state=jnp.array(self.next_states[:, 0]),
                mask=jnp.array(self.masks[:, 0]),
            )
        else:
            return RLBatch(
                state=jnp.array(self.states[0, 0]),
                action=jnp.array(self.actions[0, 0]),
                reward=jnp.array(self.rewards[0, 0]),
                next_state=jnp.array(self.next_states[0, 0]),
                mask=jnp.array(self.masks[0, 0]),
            )

    def save(self, path):
        start_time = time()
        print(f"Beginning to write at {self.insert_index}")
        if not os.path.exists(path):
            os.makedirs(path)
        outfile = os.path.join(path, f"replay_buffer_{self.last_saved:10d}.npz")
        np.savez_compressed(
            outfile,
            s=self.states[:, self.last_saved : self.insert_index],
            a=self.actions[:, self.last_saved : self.insert_index],
            r=self.rewards[:, self.last_saved : self.insert_index],
            sn=self.next_states[:, self.last_saved : self.insert_index],
            m=self.masks[:, self.last_saved : self.insert_index],
            ls=self.last_saved,
            ii=self.insert_index,
            size=self.filled,
        )
        self.last_saved = self.insert_index
        end_time = time()
        print(
            f"Saved replay buffer at step {self.insert_index} in {end_time - start_time} seconds"
        )

    def load(self, path):
        infile = os.path.join(path, "replay_buffer*")
        files = glob.glob(infile)
        for file in sorted(files):
            try:
                data = np.load(file)

                begin_idx = data["ls"]
                end_idx = data["ii"]

                self.states[:, begin_idx:end_idx] = data["s"]
                self.next_states[:, begin_idx:end_idx] = data["sn"]
                self.actions[:, begin_idx:end_idx] = data["a"]
                self.rewards[:, begin_idx:end_idx] = data["r"]
                self.masks[:, begin_idx:end_idx] = data["m"]
                self.insert_index = int(data["ii"])
                self.filled = int(data["size"])
            except zipfile.BadZipFile as e:
                assert file == sorted(files)[-1]
                continue
