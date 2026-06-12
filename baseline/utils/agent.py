import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.utils.config import criterion
from baseline.utils.models import DQN, get_backbone
from baseline.utils.tools import ReplayMemory
from v2.backbone.engine import DQNEngine
from v2.data.preprocess import crop_and_resize


class BaselineQNetwork(nn.Module):
    """Adapts baseline features + history to the v2 engine interface."""

    def __init__(self, q_network: DQN):
        super().__init__()
        self.q_network = q_network

    def forward(self, image_features: torch.Tensor, history_features: torch.Tensor) -> torch.Tensor:
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if history_features.dim() == 1:
            history_features = history_features.unsqueeze(0)

        features = torch.cat((image_features, history_features), dim=1)
        return self.q_network(features)


class Agent:
    def __init__(
        self,
        classe="mixing",
        alpha=0.1,
        nu=3.0,
        threshold=0.5,
        max_steps=20,
        load=False,
        device="cpu",
        extractor_name="vgg16",
        use_cache=True,
    ):
        del load  # Legacy flag kept for CLI compatibility.

        self.classe = classe
        self.alpha = alpha
        self.nu = nu
        self.threshold = threshold
        self.max_steps = max_steps
        self.history_size = 9
        self.n_actions = 9
        self.batch_size = 100
        self.gamma = 0.9
        self.epsilon = 1.0
        self.device = torch.device(device)
        self.use_cache = use_cache
        self.save_path = f"./models/q_network_{extractor_name}_{self.classe}.pth"

        self.feature_extractor = get_backbone(extractor_name).to(self.device)
        self.feature_extractor.eval()

        input_dim = self.feature_extractor.output_dim + (self.history_size * self.n_actions)
        self.q_network = DQN(input_dim=input_dim, outputs=self.n_actions).to(self.device)
        self.model = BaselineQNetwork(self.q_network).to(self.device)
        self.engine = DQNEngine(self.model, gamma=self.gamma, use_target_net=True)
        self.loss_fn = criterion
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-6)
        self.memory = ReplayMemory(10000)

        self.last_next_state = None
        self.last_mask = None

    def save_network(self):
        torch.save(self.model.state_dict(), self.save_path)

    def load_network(self):
        state_dict = torch.load(self.save_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def update_target_network(self):
        self.engine.update_target()

    def train_step(self, batch_size=20):
        if len(self.memory) < batch_size:
            return 0.0

        states, actions, next_states, rewards, dones = self._sample_batch(batch_size)
        self.optimizer.zero_grad()

        loss = self.engine.compute_loss(
            states["image"],
            states["history"],
            actions,
            rewards,
            next_states["image"],
            next_states["history"],
            dones,
            self.loss_fn,
            self.device,
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def step(self, image, history, current_mask, ground_truth, step_count, epsilon):
        self.epsilon = epsilon

        state = self._state_from_observation(image, history, current_mask)
        if step_count >= self.max_steps:
            action = 8
        else:
            action = self._select_action(state, history, ground_truth)

        next_history = history[1:] + [action]
        if action == 8:
            new_mask = current_mask
            reward = self.compute_finish_reward(current_mask, ground_truth)
            done = True
            next_state = self._empty_next_state()
        else:
            new_mask = self.compute_mask(action, current_mask)
            reward = self.compute_reward(action, current_mask, ground_truth)
            done = False
            next_state = self._state_from_observation(image, next_history, new_mask)

        self.memory.push(
            state,
            int(action),
            next_state,
            float(reward),
            bool(done),
        )
        return new_mask, reward, done, next_history

    def feature_extract(self, image, history, width, height, current_mask, skip_image=False):
        del width, height
        history_tensor = self._encode_history(history)

        if skip_image:
            image_tensor = None
        elif self.use_cache and self.last_next_state is not None and np.array_equal(current_mask, self.last_mask):
            image_tensor = self.last_next_state.clone()
        else:
            image_tensor = self._extract_features(image, current_mask)

        return image_tensor, history_tensor

    def compute_mask(self, action, current_mask):
        delta_width = self.alpha * (current_mask[2] - current_mask[0])
        delta_height = self.alpha * (current_mask[3] - current_mask[1])

        dx1 = dy1 = dx2 = dy2 = 0.0
        if action == 0:
            dx1 = delta_width
            dx2 = delta_width
        elif action == 1:
            dx1 = -delta_width
            dx2 = -delta_width
        elif action == 2:
            dy1 = delta_height
            dy2 = delta_height
        elif action == 3:
            dy1 = -delta_height
            dy2 = -delta_height
        elif action == 4:
            dx1 = -delta_width
            dx2 = delta_width
            dy1 = -delta_height
            dy2 = delta_height
        elif action == 5:
            dx1 = delta_width
            dx2 = -delta_width
            dy1 = delta_height
            dy2 = -delta_height
        elif action == 6:
            dy1 = delta_height
            dy2 = -delta_height
        elif action == 7:
            dx1 = delta_width
            dx2 = -delta_width

        new_mask = np.array(
            [
                current_mask[0] + dx1,
                current_mask[1] + dy1,
                current_mask[2] + dx2,
                current_mask[3] + dy2,
            ],
            dtype=np.float32,
        )
        return np.array(
            [
                min(new_mask[0], new_mask[2]),
                min(new_mask[1], new_mask[3]),
                max(new_mask[0], new_mask[2]),
                max(new_mask[1], new_mask[3]),
            ],
            dtype=np.float32,
        )

    def compute_iou(self, mask, ground_truth):
        dx = min(mask[2], ground_truth[2]) - max(mask[0], ground_truth[0])
        dy = min(mask[3], ground_truth[3]) - max(mask[1], ground_truth[1])

        inter_area = dx * dy if dx >= 0 and dy >= 0 else 0.0
        mask_area = (mask[2] - mask[0]) * (mask[3] - mask[1])
        ground_truth_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
        union = mask_area + ground_truth_area - inter_area
        return inter_area / union if union > 0 else 0.0

    def compute_reward(self, action, current_mask, ground_truth):
        new_mask = self.compute_mask(action, current_mask)
        iou_new = self.compute_iou(new_mask, ground_truth)
        iou_current = self.compute_iou(current_mask, ground_truth)
        return 1.0 if iou_new > iou_current else -1.0

    def compute_finish_reward(self, current_mask, ground_truth):
        iou = self.compute_iou(current_mask, ground_truth)
        return float(self.nu) if iou >= self.threshold else -float(self.nu)

    def _select_action(self, state, history, ground_truth):
        del history
        if random.random() > self.epsilon:
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state["image"].to(self.device), state["history"].to(self.device))
            self.model.train()
            return int(torch.argmax(q_values, dim=1).item())

        positive_actions = []
        negative_actions = []
        for action in range(self.n_actions):
            reward = (
                self.compute_finish_reward(state["mask"], ground_truth)
                if action == 8
                else self.compute_reward(action, state["mask"], ground_truth)
            )
            if reward >= 0:
                positive_actions.append(action)
            else:
                negative_actions.append(action)

        if positive_actions:
            return random.choice(positive_actions)
        return random.choice(negative_actions or list(range(self.n_actions)))

    def _state_from_observation(self, image, history, current_mask):
        image_features = self._extract_features(image, current_mask)
        history_features = self._encode_history(history)

        if self.use_cache:
            self.last_next_state = image_features.clone()
            self.last_mask = np.array(current_mask, copy=True)

        return {
            "image": image_features,
            "history": history_features,
            "mask": np.array(current_mask, copy=True),
        }

    def _extract_features(self, image, current_mask):
        cropped_image = crop_and_resize(image, current_mask)
        image_tensor = torch.from_numpy(np.transpose(cropped_image, (2, 0, 1))).float().unsqueeze(0)
        image_tensor = image_tensor.to(self.device) / 255.0

        with torch.no_grad():
            features = self.feature_extractor(image_tensor).detach().cpu()
        return features

    def _encode_history(self, history):
        history_tensor = torch.zeros((1, self.history_size * self.n_actions), dtype=torch.float32)
        valid_history = [action for action in history if action != -1][-self.history_size :]
        for idx, action in enumerate(valid_history):
            history_tensor[0, idx * self.n_actions + action] = 1.0
        return history_tensor

    def _empty_next_state(self):
        return {
            "image": torch.zeros((1, self.feature_extractor.output_dim), dtype=torch.float32),
            "history": torch.zeros((1, self.history_size * self.n_actions), dtype=torch.float32),
        }

    def _sample_batch(self, batch_size):
        transitions = self.memory.sample(batch_size)

        states = {
            "image": torch.cat([transition.state["image"] for transition in transitions]).to(self.device),
            "history": torch.cat([transition.state["history"] for transition in transitions]).to(self.device),
        }
        next_states = {
            "image": torch.cat([transition.next_state["image"] for transition in transitions]).to(self.device),
            "history": torch.cat([transition.next_state["history"] for transition in transitions]).to(self.device),
        }
        actions = torch.tensor([transition.action for transition in transitions], dtype=torch.long, device=self.device)
        rewards = torch.tensor([transition.reward for transition in transitions], dtype=torch.float32, device=self.device)
        dones = torch.tensor([transition.done for transition in transitions], dtype=torch.float32, device=self.device)

        return states, actions, next_states, rewards, dones
