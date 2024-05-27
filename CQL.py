#General
import wandb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#D3rlpy Library Imports
import torch
import numpy as np
from d3rlpy.algos.qlearning import CQLConfig
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer
from d3rlpy.models.q_functions import QRQFunctionFactory

#Custom Modules
from model.Custom_NNModel import CustomEncoderFactory
from data.social_nav_env import SocialNavEnv, evaluate
from data.utils import CustomTransitionPicker



#Select Device
use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Environment Defition
use_robot_model = True

env = SocialNavEnv(action_norm=False, test_mode=False,
                       use_robot_model=use_robot_model, XYAction = True, max_speed_norm=0.5, device=use_device)

eval_env = SocialNavEnv(action_norm=False, test_mode=True,
                            use_robot_model=use_robot_model, XYAction = True, max_speed_norm=0.5, device=use_device)


#Custom Network Architecture Definition
encoder = CustomEncoderFactory(256, eval_env.agent_hist, eval_env.num_agents, device=use_device)

q_func = QRQFunctionFactory(n_quantiles=32)

def replay_buffer_config():
    #Replay Buffer Config
    buffer = FIFOBuffer(limit=5000000)

    # TransitionPicker component
    transition_picker = CustomTransitionPicker()
        
    replay_buffer = ReplayBuffer(
        buffer=buffer,
        transition_picker=transition_picker,
        env=eval_env,
      )
    return replay_buffer


def agentCQL_config():

    #Agent Definition
    conservative_weight = 0.1
    agentCQL = CQLConfig(
        batch_size=100,
        #q_func_factory=q_func,
        n_action_samples=10,
        alpha_learning_rate=1e-4,
        conservative_weight=conservative_weight,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder).create(device=use_device)

    return agentCQL


def train_offline():
    """ Agent Training """
    model_name = 'CQL_1X25000epoch_NNModel_TRANSFORMER_DROPOUT0101_COLLPROB04510_ENVPLUSEVALENV_ROBOTMODEL_MOVINGTARGET_RADIOUS5_20Agents_DIRECTIONAL05_NewRSIGMOID04068_20GOAL25COL77TO.pt'
    #Dataset
    dataset = eval_env.get_dataset( agent='human', coll_done=True, render=False)
    agentCQL = agentCQL_config()
    #"""
    #FIT Online
    agentCQL.fit(
        dataset,
        n_steps=100,
        n_steps_per_epoch=50,
      )
    agentCQL.save_model(f"{model_name}")
    #"""

    return agentCQL


def eval():
    """ Agent Evaluation """

    agentCQL = agentCQL_config()
    agentCQL.build_with_env(eval_env)
    agentCQL.load_model('./Experiment_results/CQL_1X25000epoch_NNModel_TRANSFORMER_DROPOUT0101_COLLPROB04510_ENVPLUSEVALENV_ROBOTMODEL_MOVINGTARGET_RADIOUS5_20Agents_DIRECTIONAL05_NewRSIGMOID04068_20GOAL25COL77TO.pt')
  
    evaluate(agentCQL, eval_env, eval_env.agent_hist, eval_env.human_future, eval_env.goal_thresh, render=False, epoch=10)
             
    


if __name__ == '__main__':
    #train_offline()
    eval()