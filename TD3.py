#General
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#D3rlpy Library Imports
import torch
import numpy as np
from d3rlpy.algos.qlearning import TD3Config, TD3PlusBCConfig
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer

#Custom Modules
from model.Custom_NNModel import CustomEncoderFactory
from human_maze_gym.social_nav_env import SocialNavEnv, evaluate
from human_maze_gym.utils import CustomTransitionPicker

#Select Device
use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Environment Defition
use_robot_model = False

env = SocialNavEnv(action_norm=True, test_mode=False,
                       use_robot_model=use_robot_model, XYAction = True, device=use_device)

eval_env = SocialNavEnv(action_norm=True, test_mode=True,
                            use_robot_model=use_robot_model, XYAction = True, device=use_device)


#Custom Network Architecture Definition
encoder = CustomEncoderFactory(256, eval_env.agent_hist, eval_env.num_agents, device=use_device)

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

def agentTD3_config():

    #Agent Definition
    TD3 = TD3Config(
        batch_size=100,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        gamma=0.95,
        tau=0.005,
        n_critics=2,
        target_smoothing_sigma=0.1,
        target_smoothing_clip=0.3,
        update_actor_interval=2,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder
    ).create(device=use_device)
    return TD3

def agentTD3PlusBC_config():

    #Agent Definition
    Td3PlusBC = TD3PlusBCConfig(
        batch_size=100,
        actor_learning_rate=3e-3,
        critic_learning_rate=3e-3,
        alpha=1.5,
        gamma=0.95,
        tau=0.005,
        n_critics=2,
        target_smoothing_sigma=0.2,
        target_smoothing_clip=0.5,
        update_actor_interval=2,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder
    ).create(device=use_device)

    return Td3PlusBC



def transfer_learning():
    model_name = 'TD3PLUSBC_1X25000epoch_NNModel_TRANSFORMER_DROPOUT0101_COLLPROB04510_ENVPLUSEVALENV_ROBOTMODEL_MOVINGTARGET_RADIOUS5_5Agents_DIRECTIONAL05_NewRSIGMOID04068_20GOAL25COL77TO.pt'
    #model_name = 'BC_No_RobotModel_20epoch_exp0.pt'
    #Dataset
    dataset = eval_env.get_dataset(CustomTransitionPicker(), agent='human', coll_done=True, render=False)
    #Agent Definition
    agentTd3PlusBC = agentTD3PlusBC_config()

    #Fit OFFLINE
    agentTd3PlusBC.fit(
        dataset,
        n_steps=20000,
        n_steps_per_epoch=1000,
      )
    agentTd3PlusBC.save_model(f"{model_name}")
    return agentTd3PlusBC


def train():
    #Model Name
    model_name = 'PRETRAINEDTD3_3MILepoch_NNModel_TRANSFORMER_DROPOUT0101_COLLPROB04510_ENVPLUSEVALENV_ROBOTMODEL_MOVINGTARGET_RADIOUS5_5Agents_DIRECTIONAL05_NewRSIGMOID04068_20GOAL25COL77TO.pt'
    #Replay Buffer
    replay_buffer = replay_buffer_config()
    #Agent
    agentTD3 = agentTD3_config()
    agentTD3.build_with_env(eval_env)

    #Transfer Learning
    #agentTD3.load_model('TD3PlusBC_APLHA_2_5_10Agents_EXP0.pt')
    agentBC = transfer_learning()
    eval()
    agentTD3.copy_q_function_from(agentBC)
    agentTD3.copy_policy_from(agentBC)

    #Train
    #"""
    agentTD3.fit_online(eval_env,
                replay_buffer,
                n_steps=3000000,
                eval_env=eval_env,
                n_steps_per_epoch=1000,
                random_steps=0,
                update_interval=1,
                update_start_step=40000
                )
    
    
    # Saving the model parameters
    print('Model saved to: ' + model_name)
    agentTD3.save_model(model_name)
    #"""
    
    


def eval():
    eval_env = SocialNavEnv(action_norm=True,
                            use_robot_model=True, test_mode=True,
                            XYAction=True, max_speed_norm=0.5, device=use_device)
    agentTD3 = agentTD3_config()
    agentTD3.build_with_env(eval_env)
    agentTD3.load_model('./Experiment_results/PRETRAINEDTD3_3MILepoch_NNModel_TRANSFORMER_DROPOUT0101_COLLPROB04510_ENVPLUSEVALENV_ROBOTMODEL_MOVINGTARGET_RADIOUS5_5Agents_DIRECTIONAL05_NewRSIGMOID04068_20GOAL25COL77TO.pt')
    evaluate(agentTD3, eval_env, eval_env.agent_hist, eval_env.human_future, eval_env.goal_thresh, render=True, epoch=10)
             
    


if __name__ == '__main__':
   # train()
    eval()