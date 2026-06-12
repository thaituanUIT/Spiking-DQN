import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from v2.data.preprocess import crop_and_resize

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class LocalizationAgent:
    def __init__(self, model, engine=None, optimizer=None, loss_fn='huber', device='cpu', 
                 gamma=0.9, max_steps=20, action_options=9, history_size=10,
                 clip_grad=1.0, alpha=0.1, nu=3.0, threshold=0.5, use_cache=True,
                 replay_device='auto'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.replay_device = self._resolve_replay_device(replay_device)
        
        self.gamma = gamma
        self.max_steps = max_steps
        self.action_options = action_options
        self.history_size = history_size
        
        # --- THÊM BIẾN CACHE ---
        self.use_cache = use_cache
        self.last_next_state = None
        self.last_mask = None
        self._cached_image_token = None
        self._feature_cache = {}
        
        # Giữ nguyên sức chứa memory của branch bạn
        self.memory = ReplayBuffer(capacity=10000)
        
        # Loss function selection
        if loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_fn == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.HuberLoss()
            
        self.clip_grad = clip_grad
        self.alpha = alpha
        self.nu = nu
        self.threshold = threshold

        # Initialize default DQNEngine if none provided
        if engine is None:
            from v2.backbone.engine import DQNEngine
            self.engine = DQNEngine(self.model, gamma=self.gamma, use_target_net=False)
        else:
            self.engine = engine
            self.engine.gamma = self.gamma

    def _resolve_replay_device(self, replay_device):
        if replay_device == 'auto':
            if self.device.type == 'cuda':
                return torch.device('cpu')
            return self.device

        resolved = torch.device(replay_device)
        if resolved.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError("replay_device='cuda' requested but CUDA is not available.")
        return resolved
            
    def update_target_network(self):
        """Delegates target network update to the engine."""
        self.engine.update_target()

    def get_action(self, image_tensor, history_tensor, epsilon, current_mask, ground_truth):
        """
        Selects action using epsilon-greedy policy. 
        If training (ground_truth provided), it can fallback to positive reward actions.
        """
        if random.random() > epsilon:
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(image_tensor, history_tensor)
            self.model.train()
            action = torch.argmax(q_values).item()
        else:
            # Random action exploration guided by positive reward
            rewards = []
            for i in range(self.action_options):
                if i == 8:
                    reward = self.compute_finish_reward(current_mask, ground_truth)
                else:
                    reward = self.compute_reward(i, current_mask, ground_truth)
                rewards.append(reward)
                
            # Pick the best available guided action (as long as it's not a severe penalty)
            max_reward = np.max(rewards)
            
            if max_reward >= 0.0:
                positive_idx = np.where(np.array(rewards) == max_reward)[0]
                action = random.choice(positive_idx)
            else:
                action = random.choice(range(self.action_options))
        return action

    def compute_mask(self, action, current_mask):
        delta_width = self.alpha * (current_mask[2] - current_mask[0])
        delta_height = self.alpha * (current_mask[3] - current_mask[1])
        dx1, dy1, dx2, dy2 = 0, 0, 0, 0

        if action == 0:
            dx1 = delta_width; dx2 = delta_width
        elif action == 1:
            dx1 = -delta_width; dx2 = -delta_width
        elif action == 2:
            dy1 = delta_height; dy2 = delta_height
        elif action == 3:
            dy1 = -delta_height; dy2 = -delta_height
        elif action == 4:
            dx1 = -delta_width; dx2 = delta_width
            dy1 = -delta_height; dy2 = delta_height
        elif action == 5:
            dx1 = delta_width; dx2 = -delta_width
            dy1 = delta_height; dy2 = -delta_height
        elif action == 6:
            dy1 = delta_height; dy2 = -delta_height
        elif action == 7:
            dx1 = delta_width; dx2 = -delta_width

        new_mask_tmp = np.array([current_mask[0] + dx1, current_mask[1] + dy1,
                                 current_mask[2] + dx2, current_mask[3] + dy2])
        new_mask = np.array([
            min(new_mask_tmp[0], new_mask_tmp[2]),
            min(new_mask_tmp[1], new_mask_tmp[3]),
            max(new_mask_tmp[0], new_mask_tmp[2]),
            max(new_mask_tmp[1], new_mask_tmp[3])
        ])
        return new_mask

    def compute_iou(self, mask, ground_truth):
        dx = min(mask[2], ground_truth[2]) - max(mask[0], ground_truth[0])
        dy = min(mask[3], ground_truth[3]) - max(mask[1], ground_truth[1])

        inter_area = dx * dy if (dx >= 0) and (dy >= 0) else 0

        mask_area = (mask[2] - mask[0]) * (mask[3] - mask[1])
        ground_truth_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])

        union = mask_area + ground_truth_area - inter_area
        return inter_area / union if union > 0 else 0

    def compute_reward(self, action, current_mask, ground_truth):
        new_mask = self.compute_mask(action, current_mask)
        iou_new = self.compute_iou(new_mask, ground_truth)
        iou_current = self.compute_iou(current_mask, ground_truth)

        # Strictly integer-based rewards
        if iou_new > iou_current:
            # Strict logic: Agent ONLY gets positive reward if it matches ground truth (> threshold)
            if iou_new >= self.threshold:
                return 1.0
            else:
                return 0.0
        else:
            return -1.0

    def compute_finish_reward(self, current_mask, ground_truth):
        iou = self.compute_iou(current_mask, ground_truth)
        if iou >= self.threshold:
            # Scale the exponential IoU up to allow meaningful integers, then strictly round down
            reward = self.nu * (iou ** 2) * 10.0
            return float(int(reward))
        else:
            return -float(int(self.nu))

    # --- THÊM THAM SỐ skip_image CHO CACHE ---
    def feature_extract(self, img, history, width, height, current_mask, skip_image=False):
        """Converts mask to cropped image tensor and history list to tensor"""
        if not skip_image:
            feature_tensor = self._get_cached_features(img, current_mask)
        else:
            feature_tensor = None
            
        feat_hist = np.zeros(self.action_options * self.history_size)
        for i, act in enumerate(history):
            if act != -1:
                feat_hist[i * self.action_options + act] = 1
        history_tensor = torch.tensor(feat_hist, dtype=torch.float32, device=self.replay_device).unsqueeze(0)
        
        return feature_tensor, history_tensor

    def step(self, image, history, current_mask, ground_truth, step_count, epsilon):
        height, width, _ = image.shape
        self._prepare_feature_cache(image)
        
        # --- LOGIC TÁI SỬ DỤNG CACHE ---
        if self.use_cache and self.last_next_state is not None and np.array_equal(current_mask, self.last_mask):
            image_tensor = self.last_next_state
            # Still need to compute history_tensor as it changes
            _, history_tensor = self.feature_extract(image, history, width, height, current_mask, skip_image=True)
        else:
            image_tensor, history_tensor = self.feature_extract(image, history, width, height, current_mask)
        
        # Handle maximum steps termination
        if step_count >= self.max_steps:
            action = 8
        else:
            action = self.get_action(image_tensor, history_tensor, epsilon, current_mask, ground_truth)

        if action == 8:
            new_mask = current_mask
            reward = self.compute_finish_reward(current_mask, ground_truth)
            done = True
        else:
            new_mask = self.compute_mask(action, current_mask)
            reward = self.compute_reward(action, current_mask, ground_truth)
            history = history[1:] + [action]
            done = False

        next_image_tensor, next_history_tensor = self.feature_extract(image, history, width, height, new_mask)
        
        # --- CẬP NHẬT CACHE ---
        if self.use_cache:
            self.last_next_state = next_image_tensor.clone()
            self.last_mask = new_mask
        
        state = self._pack_replay_state(image_tensor, history_tensor)
        next_state = self._pack_replay_state(next_image_tensor, next_history_tensor)
        
        self.memory.push(state, action, next_state, reward, done)
        
        return new_mask, reward, done, history

    def _prepare_feature_cache(self, image):
        if not self.use_cache:
            return

        image_token = (id(image), image.shape)
        if image_token != self._cached_image_token:
            self._cached_image_token = image_token
            self._feature_cache = {}
            self.last_next_state = None
            self.last_mask = None

    def _get_cached_features(self, image, current_mask):
        self._prepare_feature_cache(image)

        cache_key = tuple(np.asarray(current_mask).astype(np.int32).tolist())
        if self.use_cache and cache_key in self._feature_cache:
            return self._feature_cache[cache_key].clone()

        cropped_img = crop_and_resize(image, current_mask)
        img_transposed = np.transpose(cropped_img, (2, 0, 1))

        image_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float().to(self.device) / 255.0

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            feature_tensor = self.model.extract_features(image_tensor).detach().to(self.replay_device)
        if was_training:
            self.model.train()

        if self.use_cache:
            self._feature_cache[cache_key] = feature_tensor
        return feature_tensor.clone()

    def _pack_replay_state(self, image_tensor, history_tensor):
        return {
            'image': self._to_replay_tensor(image_tensor),
            'history': self._to_replay_tensor(history_tensor),
        }

    def _to_replay_tensor(self, tensor):
        return tensor.detach().to(self.replay_device, copy=True).contiguous()

    def train_step(self, batch_size=20):
        if len(self.memory) < batch_size or not self.optimizer:
            return 0.0

        transitions = self.memory.sample(batch_size)
        states = [transition[0] for transition in transitions]
        actions = [transition[1] for transition in transitions]
        next_states = [transition[2] for transition in transitions]
        rewards = [transition[3] for transition in transitions]
        dones = [transition[4] for transition in transitions]

        img_states = torch.cat([s['image'] for s in states]).to(self.device)
        hist_states = torch.cat([s['history'] for s in states]).to(self.device)

        img_next = torch.cat([s['image'] for s in next_states]).to(self.device)
        hist_next = torch.cat([s['history'] for s in next_states]).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        
        loss = self.engine.compute_loss(img_states, hist_states, actions, rewards, 
                                        img_next, hist_next, dones, self.loss_fn, self.device)
        
        loss.backward()
        
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
        self.optimizer.step()
        
        return loss.item()
