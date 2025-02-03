import copy
from typing import TYPE_CHECKING, List, Any, Union, Tuple

import numpy as np
import torch
from easydict import EasyDict

import lzero.mcts.ptree.iris_ptree_mz as tree_muzero
from iris.src.models.kv_caching import KVCache, KeysValues
from lzero.mcts.ptree import MinMaxStatsList
from lzero.policy import InverseScalarTransform, to_detach_cpu_numpy

import psutil
import os

from zoo.bsuite.config.bsuite_muzero_config import observation_shape

if TYPE_CHECKING:
    import lzero.mcts.ptree.ptree_ez as ez_ptree
    import lzero.mcts.ptree.iris_ptree_mz as mz_ptree


class IrisMCTSPtree(object):
    """
    Overview:
        The Python implementation of MCTS (batch format) for MuZero.  \
        It completes the ``roots``and ``search`` methods by calling functions in module ``ptree_mz``, \
        which are implemented in Python.
    Interfaces:
        ``__init__``, ``roots``, ``search``

    ..note::
        The benefit of searching for a batch of nodes at the same time is that \
        it can be parallelized during model inference, thus saving time.
    """

    # the default_config for MuZeroMCTSPtree.
    config = dict(
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
        # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_base=19652,
        # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_init=1.25,
        # (float) The maximum change in value allowed during the backup step of the search tree update.
        value_delta_max=0.01,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            A class method that returns a default configuration in the form of an EasyDict object.
        Returns:
            - cfg (:obj:`EasyDict`): The dict of the default configuration.
        """
        # Create a deep copy of the `config` attribute of the class.
        cfg = EasyDict(copy.deepcopy(cls.config))
        # Add a new attribute `cfg_type` to the `cfg` object.
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict = None) -> None:
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key \
            in the default configuration, the user-provided value will override the default configuration. Otherwise, \
            the default configuration will be used.
        Arguments:
            - cfg (:obj:`EasyDict`): The configuration passed in by the user.
        """
        # Get the default configuration.
        default_config = self.default_config()
        # Update the default configuration with the values provided by the user in ``cfg``.
        default_config.update(cfg)
        self._cfg = default_config
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )

    @classmethod
    def roots(cls: int, root_num: int, legal_actions: List[Any], ac_hidden_state: Tuple[np.array, np.array], wm_kv_cache: KeysValues, observation: np.array, tokens: np.array) -> "mz_ptree.Roots":
        """
        Overview:
            Initializes a batch of roots to search parallelly later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.

        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ptree_mz`` module.
        """
        return tree_muzero.Roots(root_num, legal_actions, ac_hidden_state=ac_hidden_state, wm_kv_cache=wm_kv_cache, observation=observation, tokens=tokens)

    def search(
            self,
            roots: Any,
            model: torch.nn.Module,
            observation_seq: List[torch.Tensor] = None,
            tokens_seq: List[torch.Tensor] = None,
            action_seq: List[torch.Tensor] = None,
            to_play_batch: Union[int, List[Any]] = -1,

    ) -> None:
        """
        Overview:
            Do MCTS for a batch of roots. Parallel in model inference. \
            Use Python to implement the tree search.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes.
            - model (:obj:`torch.nn.Module`): The model used for inference.
            - to_play_batch (:obj:`list`): the to_play_batch list used in in self-play-mode board games.

        .. note::
            The core functions ``batch_traverse`` and ``batch_backpropagate`` are implemented in Python.
        """

        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor

            # the data storage of latent states: storing the latent state of all the nodes in one search.
            # minimax value storage
            min_max_stats_lst = MinMaxStatsList(batch_size)
            for simulation_index in range(self._cfg.num_simulations):

                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.
                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_muzero.SearchResults(num=batch_size)

                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                _, _, last_actions, virtual_to_play_batch, hidden_states, world_model_kv_cache, selected_nodes = tree_muzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, to_play_batch
                )

                observations_to_root = selected_nodes[0].get_observations_to_root()
                observations_to_root = [ torch.from_numpy(obs).to(self._cfg.device) for obs in observations_to_root ]
                tokens_to_root = selected_nodes[0].get_tokens_to_root()
                tokens_to_root = [ torch.from_numpy(t).to(self._cfg.device) for t in tokens_to_root ]
                input_obs_seq = observation_seq + observations_to_root
                input_tokens_seq = tokens_seq + tokens_to_root
                actions_to_root = selected_nodes[0].get_actions_to_root()
                actions_to_root = [ torch.tensor(a, dtype=torch.long, device=self._cfg.device) for a in actions_to_root ]
                input_act_seq = action_seq + actions_to_root

                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """

                network_output = model.recurrent_inference(hidden_states[0], input_obs_seq, input_tokens_seq, input_act_seq)


                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    [
                        network_output.observation,
                        network_output.tokens,
                        network_output.policy_logits,
                        network_output.value,
                        network_output.reward
                    ] = to_detach_cpu_numpy(
                        [
                            network_output.observation,
                            network_output.tokens,
                            network_output.policy_logits,
                            network_output.value,
                            network_output.reward,
                        ]
                    )
                model_hidden_state = network_output.ac_hidden_state
                world_model_kv_cache = None
                value_batch = network_output.value.reshape(-1).tolist()
                reward_batch = network_output.reward.reshape(-1).tolist()
                policy_logits_batch = network_output.policy_logits.tolist()


                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.
                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1
                tree_muzero.batch_backpropagate(
                    simulation_index=current_latent_state_index,
                    model_hidden_state=model_hidden_state,
                    observation=network_output.observation,
                    tokens=network_output.tokens,
                    kv_cache=world_model_kv_cache,
                    discount_factor=discount_factor,
                    value_prefixs=reward_batch,
                    values=value_batch,
                    policies=policy_logits_batch,
                    min_max_stats_lst=min_max_stats_lst,
                    results=results,
                    to_play=virtual_to_play_batch
                )