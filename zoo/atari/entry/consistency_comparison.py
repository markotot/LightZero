
from functools import partial
import torch

from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.utils import set_pkg_seed

from zoo.atari.config.atari_iris_model_config import get_configs

if __name__ == "__main__":


    env_id = "BreakoutNoFrameskip-v4"
    iris_main_config, iris_create_config = get_configs(env_id)
    seed = 0

    if iris_main_config.policy.cuda and torch.cuda.is_available():
        iris_main_config.policy.device = 'cuda'
    else:
        iris_main_config.policy.device = 'cpu'

    cfg = compile_config(iris_main_config, seed=seed, env=None, auto=True, create_cfg=iris_create_config, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    evaluator_env.launch()

    print("hello")

