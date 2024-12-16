import copy
import gc
import pickle
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from lzero.entry.utils import initialize_zeros_batch
from lzero.mcts.tree_search.iris_mcts_ctree import IrisMCTSTree as MCTSCtree
from lzero.mcts.tree_search.iris_mcts_ptree import IrisMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.model.utils import cal_dormant_ratio
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, negative_cosine_similarity, \
    prepare_obs, configure_optimizers
from torch.distributions import Categorical
from torch.nn import L1Loss

import psutil
import os

from zoo.atari.entry.tree_visualization import plot_images


@POLICY_REGISTRY.register('iris')
class IrisPolicy(Policy):
    """
    Overview:
        if self._cfg.model.model_type in ["conv", "mlp"]:
            The policy class for MuZero.
        if self._cfg.model.model_type == ["conv_context", "mlp_context"]:
            The policy class for MuZero w/ Context, a variant of MuZero.
            This variant retains the same training settings as MuZero but diverges during inference
            by employing a k-step recursively predicted latent representation at the root node,
            proposed in the UniZero paper https://arxiv.org/abs/2406.10667.
    """

    # The default_config for MuZero policy.
    config = dict(
        model=dict(
            # (str) The model type. For 1-dimensional vector obs, we use mlp model. For the image obs, we use conv model.
            model_type='iris',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) The stacked obs shape.
            # observation_shape=(1, 96, 96),  # if frame_stack_num=1
            observation_shape=(4, 96, 96),  # if frame_stack_num=4
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=False,
            # (bool) Whether to use discrete support to represent categorical distribution for value/reward/value_prefix.
            # reference: http://proceedings.mlr.press/v80/imani18a/imani18a.pdf, https://arxiv.org/abs/2403.03950
            categorical_distribution=True,
            # (int) The image channel in image observation.
            image_channel=1,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The number of res blocks in MuZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in MuZero model.
            num_channels=64,
            # (int) The scale of supports used in categorical distribution.
            # This variable is only effective when ``categorical_distribution=True``.
            support_scale=300,
            # (bool) whether to learn bias in the last linear layer in value and policy head.
            bias=True,
            # (str) The type of action encoding. Options are ['one_hot', 'not_one_hot']. Default to 'one_hot'.
            discrete_action_encoding_type='one_hot',
            # (bool) whether to use res connection in dynamics.
            res_connection_in_dynamics=True,
            # (str) The type of normalization in MuZero model. Options are ['BN', 'LN']. Default to 'LN'.
            norm_type='BN',
            # (bool) Whether to analyze simulation normalization.
            analysis_sim_norm=False,
            # (bool) Whether to analyze dormant ratio.
            analysis_dormant_ratio=False,
            # (bool) Whether to use HarmonyDream to balance weights between different losses. Default to False.
            # More details can be found in https://arxiv.org/abs/2310.00344.
            harmony_balance=False
        ),
        # ****** common ******
        # (bool) whether to use rnd model.
        use_rnd_model=False,
        # (bool) Whether to use multi-gpu training.
        multi_gpu=False,
        # (bool) Whether to enable the sampled-based algorithm (e.g. Sampled EfficientZero)
        # this variable is used in ``collector``.
        sampled_algo=False,
        # (bool) Whether to enable the gumbel-based algorithm (e.g. Gumbel Muzero)
        gumbel_algo=False,
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=False, # TODO: (NOTE) changed from True to False to use pMCTS instead of cMCTS
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (str) The type of environment. Options are ['not_board_games', 'board_games'].
        env_type='not_board_games',
        # (str) The type of action space. Options are ['fixed_action_space', 'varied_action_space'].
        action_type='fixed_action_space',
        # (str) The type of battle mode. Options are ['play_with_bot_mode', 'self_play_mode'].
        battle_mode='play_with_bot_mode',
        # (bool) Whether to monitor extra statistics in tensorboard.
        monitor_extra_statistics=True,
        # (int) The transition number of one ``GameSegment``.
        game_segment_length=200,
        # (bool): Indicates whether to perform an offline evaluation of the checkpoint (ckpt).
        # If set to True, the checkpoint will be evaluated after the training process is complete.
        # IMPORTANT: Setting eval_offline to True requires configuring the saving of checkpoints to align with the evaluation frequency.
        # This is done by setting the parameter learn.learner.hook.save_ckpt_after_iter to the same value as eval_freq in the train_muzero.py automatically.
        eval_offline=False,
        # (bool) Whether to calculate the dormant ratio.
        cal_dormant_ratio=False,
        # (bool) Whether to analyze simulation normalization.
        analysis_sim_norm=False,
        # (bool) Whether to analyze dormant ratio.
        analysis_dormant_ratio=False,

        # ****** observation ******
        # (bool) Whether to transform image to string to save memory.
        transform2string=False,
        # (bool) Whether to use gray scale image.
        gray_scale=False,
        # (bool) Whether to use data augmentation.
        use_augmentation=False,
        # (list) The style of augmentation.
        augmentation=['shift', 'intensity'],

        # ******* learn ******
        # (bool) Whether to ignore the done flag in the training data. Typically, this value is set to False.
        # However, for some environments with a fixed episode length, to ensure the accuracy of Q-value calculations,
        # we should set it to True to avoid the influence of the done flag.
        ignore_done=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor.
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.replay_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        replay_ratio=0.25,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network. ['SGD', 'Adam']
        optim_type='SGD',
        # (float) Learning rate for training policy network. Initial lr for manually decay schedule.
        learning_rate=0.2,
        # (int) Frequency of target network update.
        target_update_freq=100,
        # (int) Frequency of target network update.
        target_update_freq_for_intrinsic_reward=1000,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=10,
        # (int) The number of episodes in each collecting stage.
        n_episode=8,
        # (int) the number of simulations in MCTS.
        num_simulations=50,
        # (float) Discount factor (gamma) for returns.
        discount_factor=0.997,
        # (int) The number of steps for calculating target q_value.
        td_steps=5,
        # (int) The number of unroll steps in dynamics network.
        num_unroll_steps=5,
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        # (float) The weight of policy entropy loss.
        policy_entropy_loss_weight=0,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=0,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        lr_piecewise_constant_decay=True,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e4),
        # (bool) Whether to use manually decayed temperature.
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        # (bool) Whether to use the true chance in MCTS in some environments with stochastic dynamics, such as 2048.
        use_ture_chance_label_in_chance_encoder=False,
        # (bool) Whether to add noise to roots during reanalyze process.
        reanalyze_noise=True,
        # (bool) Whether to reuse the root value between batch searches.
        reuse_search=False,
        # (bool) whether to use the pure policy to collect data. If False, use the MCTS guided with policy.
        collect_with_pure_policy=False,

        # ****** Priority ******
        # (bool) Whether to use priority when sampling training data from the buffer.
        use_priority=False,
        # (float) The degree of prioritization to use. A value of 0 means no prioritization,
        # while a value of 1 means full prioritization.
        priority_prob_alpha=0.6,
        # (float) The degree of correction to use. A value of 0 means no correction,
        # while a value of 1 means full correction.
        priority_prob_beta=0.4,

        # ****** UCB ******
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,

        # ****** Explore by random collect ******
        # (int) The number of episodes to collect data randomly before training.
        random_collect_episode_num=0,

        # ****** Explore by eps greedy ******
        eps=dict(
            # (bool) Whether to use eps greedy exploration in collecting data.
            eps_greedy_exploration_in_collect=False,
            # (str) The type of decaying epsilon. Options are 'linear', 'exp'.
            type='linear',
            # (float) The start value of eps.
            start=1.,
            # (float) The end value of eps.
            end=0.05,
            # (int) The decay steps from start to end eps.
            decay=int(1e5),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'IrisModel', ['lzero.model.iris_model']

    def _init_learn(self) -> None:
        pass

    def _forward_learn(self, data: Tuple[torch.Tensor]) -> Dict[str, Union[float, int]]:
        pass

    def _init_collect(self) -> None:
        pass

    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None,
    ) -> Dict:
        pass

    def _get_target_obs_index_in_step_k(self, step):
        """
        Overview:
            Get the begin index and end index of the target obs in step k.
        Arguments:
            - step (:obj:`int`): The current step k.
        Returns:
            - beg_index (:obj:`int`): The begin index of the target obs in step k.
            - end_index (:obj:`int`): The end index of the target obs in step k.
        Examples:
            >>> self._cfg.model.model_type = 'conv'
            >>> self._cfg.model.image_channel = 3
            >>> self._cfg.model.frame_stack_num = 4
            >>> self._get_target_obs_index_in_step_k(0)
            >>> (0, 12)
        """
        if self._cfg.model.model_type in ['conv', 'conv_context']:
            beg_index = self._cfg.model.image_channel * step
            end_index = self._cfg.model.image_channel * (step + self._cfg.model.frame_stack_num)
        elif self._cfg.model.model_type in ['mlp', 'mlp_context']:
            beg_index = self._cfg.model.observation_shape * step
            end_index = self._cfg.model.observation_shape * (step + self._cfg.model.frame_stack_num)
        return beg_index, end_index

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_model = self._model
        # if self._cfg.mcts_ctree:
        #     self._mcts_eval = MCTSCtree(self._cfg)
        # else:
        self._mcts_eval = MCTSPtree(self._cfg)
        if self._cfg.model.model_type == 'conv_context':
            self.last_batch_obs = torch.zeros([3, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(3)]

        self.step = 0

        self.policy_actions = []
        self.mcts_actions = []
        self.observations = []

        self.ac_hidden_state = None
        self.wm_kv_cache = None

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1,
                      ready_env_id: np.array = None, ) -> Dict:
        """
        Overview:
            The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
            Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        gc.collect()
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        output = {i: None for i in ready_env_id}

        self.observations.append(data.detach().cpu())


        with torch.no_grad():

            policy_logits, values, hidden_state = self._eval_model.predict(obs=data, model_hidden_state=self.ac_hidden_state)
            reward_roots =  [0. for _ in range(data.size(0))]

            predicted_values = values.detach().cpu().numpy()  # shape（B, 1）
            initial_observation = data.detach().cpu().numpy()
            policy_entropy = Categorical(logits=policy_logits).entropy().detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）
            ac_action = np.argmax(policy_logits)

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)]

            roots = MCTSPtree.roots(root_num=active_eval_env_num,
                                    legal_actions=legal_actions,
                                    ac_hidden_state=self.ac_hidden_state,
                                    wm_kv_cache=self.wm_kv_cache,
                                    observation=initial_observation)

            roots.prepare_no_noise(rewards=reward_roots, policies=policy_logits, to_play=to_play)

            obs_seq = [obs.to(self._cfg.device) for obs in self.observations]
            action_seq = [torch.tensor(a, dtype=torch.long, device=self._cfg.device) for a in self.mcts_actions]

            self._mcts_eval.search(roots=roots,
                                   model=self._eval_model,
                                   to_play_batch=to_play,
                                   observation_seq=obs_seq,
                                   action_seq=action_seq)

            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                if self._cfg.num_simulations == 0:
                    action = ac_action
                    distributions, value = roots_visit_count_distributions[i], roots_values[i]
                    visit_count_distribution_entropy = 0
                    self.ac_hidden_state = hidden_state

                else:
                    distributions, value = roots_visit_count_distributions[i], roots_values[i]
                    action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=1, deterministic=True
                    )
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                    selected_child = roots.roots[i].children[action_index_in_legal_action_set]
                    self.ac_hidden_state = selected_child.ac_hidden_state
                    self.wm_kv_cache = selected_child.kv_cache

                    #obs = np.transpose(selected_child.observation[0], (1, 2, 0))
                    #initial_observation = np.transpose(initial_observation[0], (1, 2, 0))

                    #plot_images([initial_observation, obs], start_step=self.step, num_steps=2, transpose=False)

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': predicted_values,
                    'predicted_policy_logits': policy_logits,
                    'policy_entropy': policy_entropy,
                    'ac_action': ac_action, # argmax following policy
                    'same_action': int(ac_action == action)
                }

                self.policy_actions.append(ac_action)
                self.mcts_actions.append(action)

                # Enable to turn on storing the roots locally
                #self.save_data(roots)

                if self._cfg.model.model_type in ["conv_context"]:
                    batch_action.append(action)
            if self._cfg.model.model_type in ["conv_context"]:
                self.last_batch_obs = data
                self.last_batch_action = batch_action

        self.step += 1

        return output

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset the observation and action for the collector environment.
        Arguments:
            - data_id (`Optional[List[int]]`): List of data ids to reset (not used in this implementation).
        """
        if self._cfg.model.model_type in ["conv_context"]:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.collector_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.collector_env_num)]

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset the observation and action for the evaluator environment.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): List of data ids to reset (not used in this implementation).
        """
        if self._cfg.model.model_type in ["conv_context"]:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.evaluator_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.evaluator_env_num)]
    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return_list = [
            'analysis/dormant_ratio_encoder',
            'analysis/dormant_ratio_dynamics',
            'analysis/latent_state_l2_norms',
            'analysis/l2_norm_before',
            'analysis/l2_norm_after',
            'analysis/grad_norm_before',
            'analysis/grad_norm_after',

            'collect_mcts_temperature',
            'cur_lr',
            'weighted_total_loss',
            'total_loss',
            'policy_loss',
            'policy_entropy',
            'target_policy_entropy',
            'reward_loss',
            'value_loss',
            'consistency_loss',
            'value_priority',
            'target_reward',
            'target_value',
            'predicted_rewards',
            'predicted_values',
            'transformed_target_reward',
            'transformed_target_value',
            'total_grad_norm_before_clip',
        ]
        # ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
        if self._cfg.model.harmony_balance:
            harmony_list = [
                'harmony_dynamics', 'harmony_dynamics_exp_recip',
                'harmony_policy', 'harmony_policy_exp_recip',
                'harmony_value', 'harmony_value_exp_recip',
                'harmony_reward', 'harmony_reward_exp_recip',
                'harmony_entropy', 'harmony_entropy_exp_recip',
            ]
            return_list.extend(harmony_list)
        return return_list

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def __del__(self):
        if self._cfg.model.analysis_sim_norm:
            # Remove hooks after training.
            self._collect_model.encoder_hook.remove_hooks()
            self._target_model.encoder_hook.remove_hooks()

    def _process_transition(self, obs, policy_output, timestep):
        # be compatible with DI-engine Policy class
        pass

    def _get_train_sample(self, data):
        # be compatible with DI-engine Policy class
        pass


    def save_data(self, roots):
        roots.store_mcts_tree(step=self.step)  # Save the trajectories for analysis
        with open(f'./mcts/iris/mcts_actions.pkl', 'wb') as f:
            pickle.dump(self.mcts_actions, f)
        with open(f'./mcts/iris/policy_actions.pkl', 'wb') as f:
            pickle.dump(self.policy_actions, f)
        with open(f'./mcts/iris/observations.pkl', 'wb') as f:
            pickle.dump(self.observations, f)


