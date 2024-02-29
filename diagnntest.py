import os
os.environ["DEV"] = "1"
os.environ["NEURODIFF_API_URL"] = "http://dev.neurodiff.io"
os.environ["NEURODIFF_API_KEY"] = 'tNaaIvvvdg72-c8VcTZRgpALsl0ns77ljEvxul6tG0E'
import warnings
warnings.filterwarnings("ignore")
import numpy
from diagnn import *
from tqdm.auto import tqdm
from diagnn_rl import *
from diagnn_rl_ppo import *

def generate_points(n):
    x = np.linspace(0, 1, n)
    return (9*(1 - (1-x)**2)+1)/10

action_space = generate_points(10)
c = DiagNN("Data/Curve_sofT_Case3of5.csv")
store_mse = Store_MSE_Loss()

scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=500, gamma=0.975)
scheduler_cb = DoSchedulerStep(scheduler=scheduler)

potential_cb = BestValidationCallback()
epochs = 10
c.curriculum = 0.2
c.solver.fit(max_epochs=epochs, 
             callbacks=[potential_cb, scheduler_cb], 
             tqdm_file=tqdm(total=epochs, dynamic_ncols=True, desc='Epochs', unit='iteration', colour='#0afa9e'))
# print(c.get_loss())
# print(c.get_residuals().shape)
# print(action_space)
env = CINNS_RL()
# print(np.array([env.action_space.sample()]).shape)
agent = PPO(env)
agent.init_hyperparameters()
agent.learn(10)
# for i in range(100):
#     state, reward, done, _, _ = env.step(env.action_space.sample())
#     print(reward)
