from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from lzero.policy import select_action
from torch.distributions.categorical import Categorical
import torch.nn as nn

from iris.src.models.actor_critic import ActorCritic
from iris.src.models.tokenizer.tokenizer import Tokenizer
from iris.src.models.world_model import WorldModel
from iris.src.utils import extract_state_dict


class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:

        if self.actor_critic.use_original_obs:
            input_ac = obs
        else:
            tokenized_ac = self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True)
            input_ac = torch.clamp(tokenized_ac, 0, 1)

        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token

    def set_model_hidden_state(self, model_hidden_state: Tuple[np.ndarray, np.ndarray]):
        self.actor_critic.set_hidden_state(model_hidden_state)

    def get_model_hidden_state(self):
        return self.actor_critic.get_hidden_state()
