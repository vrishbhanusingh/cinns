import os
import copy
from re import L
import dill 
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from copy import deepcopy
from tqdm.auto import tqdm
from neurodiffeq import diff
import torch.nn.functional as F
from ordered_set import OrderedSet
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.generators import BaseGenerator
from neurodiffeq.callbacks import ActionCallback 
from neurodiffeq.generators import Generator1D, PredefinedGenerator
from neurodiffeq.conditions import BundleIVP, NoCondition, BundleDirichletBVP

large = 20
med = 16
small = 12

import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
graphsize = (4, 4)
colors = ['#66bb6a', '#558ed5', '#dd6a63', '#dcd0ff', '#ffa726', '#8c5eff', '#f44336', '#00bcd4', '#ffc107', '#9c27b0']
params = {'axes.titlesize': small,
          'legend.fontsize': small,
          'figure.figsize': graphsize,
          'axes.labelsize': small,
          'axes.linewidth': 2,
          'xtick.labelsize': small,
          'xtick.color' : '#1D1717',
          'ytick.color' : '#1D1717',
          'ytick.labelsize': small,
          'axes.edgecolor':'#1D1717',
          'figure.titlesize': med,
          'axes.prop_cycle': cycler(color = colors),}
# Define your custom colormap with #66bb6a
cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['#66bb6a', '#1D1717'])
plt.rcParams.update(params)

IN_COLAB = torch.cuda.is_available()

def V_or(phi):
    phim = 1.0
    phiq = 10
    term1 = 6 * (2*phi)**2 * (8 + 2*(2*phi)**2/(phim)**2 - 3*(2*phi)**4 / phiq )**2
    term2 = (96 +8*(2*phi)**2 + (2*phi)**4/phim**2 - (2*phi)**6/phiq)**2
    return (1/4)*(1/768)*(term1 - term2)

def DV_or(phi):
    phim = 1.0
    phiq = 10
    return (-(1/(3*phim**4*phiq**2))*phi*(phiq**2*phi**4*(-9 + 2*phi**2)+
            2*phim**2*phiq*phi**4*(3*phiq+72*phi**2 - 10*phi**4) +
            phim**4*(12*phi**8*(-45 + 4*phi**2) + phiq**2*(9 + 4*phi**2) -
           4*phiq*phi**4*(-9 + 8*phi**2))))

class CustomNN(nn.Module):
    def __init__(self, n_input_units, hidden_units, actv, n_output_units):
        super(CustomNN, self).__init__()

        # Layers list to hold all layers
        self.layers = nn.ModuleList()

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))

        # Learnable parameters mu and sigma for the firs layer
        self.mu =  torch.linspace(0,2, hidden_units[0])
        self.sigma = nn.Parameter(torch.ones(hidden_units[0])*0.1)

        # Remaining hidden layers
        for i in range(len(hidden_units) - 1):
            self.layers.append(actv())
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

        # Output layer
        self.layers.append(actv())
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)

    def forward(self, x):

        inputx = x[:,0].reshape(-1,1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply the custom operation after the first layer
            if i == 0:
                x = x * torch.exp(- (x - self.mu) ** 2 / self.sigma ** 2)

        # Output layer transformation
        x = self.fc_out(x)
        return x

class MeshGenerator(BaseGenerator):

    def __init__(self, g1, pg):

        super(MeshGenerator, self).__init__()
        self.g1 = g1
        self.pg = pg

    def get_examples(self):

        u = self.g1.get_examples()
        u = u.reshape(-1, 1, 1)

        bundle_params = self.pg.get_examples()
        if isinstance(bundle_params, torch.Tensor):
            bundle_params = (bundle_params,)
        assert len(bundle_params[0].shape) == 1, "shape error, ask shuheng"
        n_params = len(bundle_params)

        bundle_params = torch.stack(bundle_params, dim=1)
        bundle_params = bundle_params.reshape(1, -1, n_params)

        uu, bb = torch.broadcast_tensors(u, bundle_params)
        uu = uu[:, :, 0].reshape(-1)
        bb = [bb[:, :, i].reshape(-1) for i in range(n_params)]

        return uu, *bb

class minmaxScaler():
  def __init__(self, x):
    self.minx = x.min().detach().item()
    self.maxx = x.max().detach().item()

  def transform(self, x):
    return (x - self.minx)/(self.maxx - self.minx)
  
class DoSchedulerStep(ActionCallback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def __call__(self, solver):
        self.scheduler.step()

class BestValidationCallback(ActionCallback):
    def __init__(self):
        super().__init__()
        self.best_potential = None

    def __call__(self, solver):
        if solver.lowest_loss is None or solver.metrics_history['r2_loss'][-1] <= solver.lowest_loss:
            self.best_potential = copy.deepcopy(solver.V)

class Store_MSE_Loss(ActionCallback):
    def __init__(self):
        super().__init__()
        self.mse_loss_history = []

    def __call__(self, solver):
        if solver.global_epoch % 10 == 0:
          for i in range(5):
            batch = self.generator['train'].get_examples()
            r = solver.get_residuals(*batch, to_numpy = True)
            self.mse_loss_history.append((np.array(r)**2).mean())

class CustomBundleSolver1D(BundleSolver1D):
    def __init__(self, *args, **kwargs):

        self.V = kwargs.pop('V', None)

        super().__init__( *args, **kwargs)
        self.metrics_history['r2_loss'] = []
        self.metrics_history['phi_max'] = []

    def _set_loss_fn(self, criterion):
        pass

    def loss_fn(self,r,f,x):
        
        loss_r2 = (r**2).mean() 
        self.metrics_history['r2_loss'].append((r**2).mean())#.detach().item())
        self.metrics_history['phi_max'].append(f[5][-49: ].mean())#.detach().item())
        return loss_r2

    def _update_best(self, key):
        """Update ``self.lowest_loss`` and ``self.best_nets``
        if current training/validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.metrics_history['r2_loss'][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.best_nets = deepcopy(self.nets)

    def fit(self, max_epochs, callbacks=(), tqdm_file='default', **kwargs):
        r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.

        If ``callbacks`` is passed, callbacks are run, one at a time,
        after training, validating and updating best model.

        :param max_epochs: Number of epochs to run.
        :type max_epochs: int
        :param callbacks:
            A list of callback functions.
            Each function should accept the ``solver`` instance itself as its **only** argument.
        :rtype callbacks: list[callable]
        :param tqdm_file:
            File to write tqdm progress bar. If set to None, tqdm is not used at all.
            Defaults to ``sys.stderr``.
        :type tqdm_file: io.StringIO or _io.TextIOWrapper

        .. note::
            1. This method does not return solution, which is done in the ``.get_solution()`` method.
            2. A callback ``cb(solver)`` can set ``solver._stop_training`` to True to perform early stopping.
        """
        self._stop_training = False
        self._max_local_epoch = max_epochs

        self.callbacks = callbacks

        monitor = kwargs.pop('monitor', None)
        if monitor:
            warnings.warn("Passing `monitor` is deprecated, "
                          "use a MonitorCallback and pass a list of callbacks instead")
            callbacks = [monitor.to_callback()] + list(callbacks)
        if kwargs:
            raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')  # pragma: no cover

        flag=False
        if str(tqdm_file) == 'default':
            bar = tqdm(
                total = max_epochs,
                desc='Training Progress',
                colour='blue',
                dynamic_ncols=True,
            )
        elif tqdm_file is not None:
            bar = tqdm_file
        else:
            flag=True
        
            

        for local_epoch in range(max_epochs):
            # stop training if self._stop_training is set to True by a callback
            if self._stop_training:
                break

            # register local epoch (starting from 1 instead of 0) so it can be accessed by callbacks
            self.local_epoch = local_epoch + 1
            self.run_train_epoch()
            self.run_valid_epoch()
            for cb in callbacks:
                cb(self)
            if not flag:
                bar.update(1)

df_data_yago_a = pd.read_csv("Data/Yago_A.csv", header=None).values
df_data_yago_sigma= pd.read_csv("Data/Yago_Sigma.csv", header=None).values
df_data_yago_phi= pd.read_csv("Data/Yago_Phi.csv", header=None).values

A_yago = df_data_yago_a[:, 1]
u_yago = df_data_yago_a[:, 0]
Sigma_yago = df_data_yago_sigma[:, 1]
phi_yago = df_data_yago_phi[:, 1]

class DiagNN():

    def __init__(self, path, delta = 0.0, curriculum = 1.0):

        self.delta = delta
        self.curriculum = curriculum
        self.path = path
        df_data = pd.read_csv(path, header=None).values

        S_true_1= torch.tensor(df_data[50:100:2,1])
        T_true_1= torch.tensor(df_data[50:100:2,0])

        S_true_3= torch.tensor(df_data[101:200:6,1])
        T_true_3= torch.tensor(df_data[101:200:6,0])

        S_true_4= torch.tensor(df_data[201:331:20,1])
        T_true_4= torch.tensor(df_data[201:331:20,0])

        S_true_5 = torch.tensor([0])
        T_true_5 = torch.tensor([0])

        self.S_true=torch.cat([S_true_1, S_true_3, S_true_4, S_true_5],dim=0)
        self.T_true=torch.cat([T_true_1, T_true_3, T_true_4, T_true_5],dim=0)

        if IN_COLAB:
            self.S_true = self.S_true.cpu()
            self.T_true = self.T_true.cpu()

        self.Sigma_uh_all = (self.S_true/np.pi)**(1/3)
        self.Va_uh_all = (-self.T_true*4*np.pi)

        if IN_COLAB:
            self.Sigma_uh_all = self.Sigma_uh_all.detach().numpy()
            self.Va_uh_all= self.Va_uh_all.detach().numpy()

        self.pg = PredefinedGenerator(self.Sigma_uh_all, self.Va_uh_all)
        self.g1 = Generator1D(128, 0, self.curriculum, method='equally-spaced-noisy')
        self.g2 = Generator1D(16, 0, 1, method='equally-spaced')
        self.train_generator =  MeshGenerator(self.g1, self.pg)
        self.valid_generator =  MeshGenerator(self.g2, self.pg)
        
        self.V = CustomNN(n_input_units = 1, hidden_units = [128, 512] ,actv = nn.SiLU, n_output_units = 1)
        
        self.conditions = [ NoCondition(),  # no condition on Vs
                            BundleIVP(1, None, bundle_param_lookup=dict(u_0=1)), #condition on Va = -4 pi T
                            BundleIVP(0, 1),   # Vphi(0) ==1
                            BundleDirichletBVP(0, 1, 1, None, bundle_param_lookup=dict(u_1=0)),  # Sigma_{u=0} = 1, Sigma_{u=1}=(S/pi)**(1/3)
                            BundleDirichletBVP(0, 1, 1, 0),   # A (0) == 1  A(1)=0
                            BundleIVP(0, 0),  #phi(0)=0 #BundleDirichletBVP(0, 0,1, phi_yago[-1])#
                        ]
        self.nets = [FCNN(n_input_units=3, hidden_units=[16,16,16]) for _ in range(6)]
        
        self.adam = torch.optim.Adam(OrderedSet([p for net in self.nets + [self.V] for p in net.parameters()]), \
                        lr=1e-3,  betas=(0.9, 0.99))
        
        self.lbfgs = torch.optim.LBFGS(OrderedSet([p for net in self.nets + [self.V] for p in net.parameters()]), \
                        lr=1e-2)
        
        self.solver = CustomBundleSolver1D( ode_system=self.equations,
                                            conditions=self.conditions,
                                            t_min=self.delta,
                                            t_max=1,
                                            train_generator=self.train_generator,
                                            valid_generator=self.valid_generator,
                                            optimizer=self.adam,
                                            nets=self.nets,
                                            n_batches_valid=0,
                                            eq_param_index=(),
                                            V = self.V
                                        )

    def update_generator(self, curriculum = 1.0, valid_method = 'equally-spaced'):

        g1 = Generator1D(128, 0, curriculum, method='chebyshev2')
        g2 = Generator1D(16, 0, 1.0, method=valid_method)
        train_generator =  MeshGenerator(g1, self.pg)
        valid_generator =  MeshGenerator(g2, self.pg)

        self.solver.generator={'train': train_generator, 'valid': valid_generator}

    def set_curriculum(self, start = 0.0, end = 1.0, valid_method = 'equally-spaced'):

        g1 = Generator1D(128, start, end, method='chebyshev2')
        g2 = Generator1D(16, 0, 1.0, method=valid_method)
        train_generator =  MeshGenerator(g1, self.pg)
        valid_generator =  MeshGenerator(g2, self.pg)

        self.solver.generator={'train': train_generator, 'valid': valid_generator}
    
    def equations(self, Vs, Va, Vp, Sigma, A, phi, u):

        # create the derivative of the V wrt to phi
        VF = diff(self.V(phi), phi, shape_check= False)

        ORIGP_FLAG = 0

        # the equations
        eq1 = Vs - diff(Sigma, u, order=1)
        eq2 = Va - diff(A, u, order=1)
        eq3 = Vp - diff(phi, u, order=1)
        eq4 = diff(Vs, u,  order=1) + (2 / 3) *Sigma * Vp ** 2

        eq5 = (u ** 2) * Sigma * diff(Va, u, order=1) + 8 / (3) * ( (1-ORIGP_FLAG)* self.V(phi)  \
                                    + ORIGP_FLAG* V_or(phi) ) * Sigma  \
                                    + Va * (3 * u ** 2 * Vs - 5 * Sigma * u) \
                                    + A * (8 * Sigma - 6 * u * Vs)



        eq6 = u ** 2 * Sigma * A * diff(Vp, u, order=1) - Sigma * (  (1-ORIGP_FLAG)*VF + ORIGP_FLAG* DV_or(phi)) \
            + Vp * (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs)


        eq7 =  (u * Vs-Sigma) * \
            ( u**2 * Sigma * Va + 2 * A * u**2 * Vs- 4 * u * A * Sigma) \
            -(2/3)*(u*Sigma**2)*(u**2 * A* Vp**2 - \
                                2 * ((1-ORIGP_FLAG)*self.V(phi) + ORIGP_FLAG*V_or(phi)))

        return [eq1, eq2, eq3, eq4 , eq5, eq6, eq7]
    
    def plot_initial(self):
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # plot the first subplot
        axs[0].scatter((self.T_true),(self.S_true), label='true')
        axs[0].hlines(1.5028, min(self.T_true), max(self.T_true))
        axs[0].set_xlabel('T')
        axs[0].set_ylabel('S')
        axs[0].legend()
        
        # plot the second subplot
        if IN_COLAB:
            axs[1].scatter(self.Va_uh_all,self.Sigma_uh_all, label='true')
            axs[1].hlines(0.78, min(-self.T_true*4*np.pi), max(-self.T_true*4*np.pi))
            axs[1].vlines(-10.0369, 0.4, 1.3)
            axs[1].set_xlabel('Va_h')
            axs[1].set_ylabel('Sigma_h')
            axs[1].legend()
        
        else:
            axs[1].scatter(self.Va_uh_all.detach().numpy() ,self.Sigma_uh_all.detach().numpy() , label='true')
            axs[1].hlines(0.78, min(-self.T_true*4*np.pi), max(-self.T_true*4*np.pi))
            axs[1].vlines(-10.0369, 0.4, 1.3)
            axs[1].set_xlabel('Va_h')
            axs[1].set_ylabel('Sigma_h')
            axs[1].legend()
        
        plt.show() 

    def get_loss(self):

        residuals = self.get_residuals()
        batch = [v.reshape(-1, 1) for v in self.valid_generator.get_examples()]
        funcs = [self.solver.compute_func_val(a, b, *batch) for a, b in zip(self.solver.nets, self.solver.conditions)]
        if IN_COLAB:
            return self.solver.loss_fn(residuals, funcs, batch) + self.solver.additional_loss(residuals, funcs, batch)#.detach().cpu().numpy()

        else:
            return self.solver.loss_fn(residuals, funcs, batch) + self.solver.additional_loss(residuals, funcs, batch)#.detach().numpy()
        
    def get_residuals(self, display = False):
        
        u, sigma, Va = self.valid_generator.get_examples()
        res = self.solver.get_residuals(u, sigma, Va, best=True)
        dim = int((res[0].shape[0])/16)
        res_eq = np.zeros((7, 16, dim)) 
        for i, r in enumerate(res):
            res_eq[i, :,:] =r.cpu().detach().reshape(16, dim)
        if display:
            print(f'Mean of residuals : {round((torch.cat(res) ** 2).mean().item(),9)}.')
        return res_eq
    
    def plot_residuals(self):
              
        residuals = self.get_residuals()
        
        fig, ax = plt.subplots(3,2, figsize=(6,18))
        ax = ax.flatten()

        vmax = 0.04

        levels = np.arange(0, vmax, .001)
        for eqn in np.arange(6):  
            im = ax[eqn].imshow( (np.abs(residuals[eqn,:,:].T)),  vmin=0, vmax=vmax, interpolation='bilinear', cmap=cmap)
            ax[eqn].contour(    (np.abs(residuals[eqn,:,:].T)), levels,   extend='both')
            ax[eqn].set_title(f"Eq{eqn+1}")
        # Add a colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.9, 0.2, 0.05, 0.6])  # Adjust the position of the colorbar
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    def plot_loss(self):
        
        trace = self.solver.metrics_history
        plt.figure(figsize=graphsize)
        plt.plot(np.log10(trace['train_loss']), label='train loss')
        if len(trace['valid_loss'])!=0:
            plt.plot(np.log10(trace['valid_loss']), label='validation loss')
        if 'train__res_eq1' in trace:
            for i in range(7): 
                plt.plot(np.log10(trace[f'train__res_eq{i+1}']), label = f'eq{i+1} residuals', alpha=0.6)
        
        plt.xlabel('epochs')
        plt.ylabel('DE Residual Square Loss')
        plt.grid()
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_result(self, best = False):

        u = np.linspace(0.0001, 1, 100)
        solution = self.solver.get_solution(best = best)
        phi_h = np.ones(self.S_true.shape)   
        u_max = np.ones(self.S_true.shape)

        if IN_COLAB:
            
            for i,S in enumerate(self.S_true):
                T = self.T_true[i]
                Sigma_v = (S/np.pi)**(1/3)
                Va_v = (-T*4*np.pi)
                Sigma_uh = Sigma_v*np.ones_like(u)
                Va_uh = Va_v*np.ones_like(u)
                Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh.cuda(), Va_uh.cuda(), to_numpy=True)
                phi_h[i] = phi.max()

                i_max = phi.argmax()
                u_max[i] = u[i_max]   

        else:
            
            for i,S in enumerate(self.S_true):
                T = self.T_true[i]
                Sigma_v = (S/np.pi)**(1/3)
                Va_v = (-T*4*np.pi)
                Sigma_uh = Sigma_v*np.ones_like(u)
                Va_uh = Va_v*np.ones_like(u)
                Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh, Va_uh, to_numpy=True)
                phi_h[i] = phi.max()

                i_max = phi.argmax()
                u_max[i] = u[i_max]     
        
        phi= torch.reshape(torch.linspace(0,1.8,100),[100,1]) 
        phi = torch.Tensor(phi)
        phi.requires_grad = True
        qphi = phi.detach().cpu().numpy().reshape(-1,)
        Vv = self.V(phi)

        plt.figure(figsize=graphsize)
        plt.plot(qphi,  Vv.detach().cpu() , label='Ours')
        plt.plot(qphi, V_or(qphi), label='Known')
        plt.vlines(max(phi_h), -10, 0, colors[2])
        plt.vlines(0, -10, 0, colors[2])
        plt.xlabel('phi')
        plt.ylabel("V(phi)")
        if best:
            plt.title('Best Result')
        else:
            plt.title('Latest Result')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def compare_to_yago(self, best=False, s=1.4914411398619525, t=0.3706278568458316):

        u = np.linspace(0.0001, 1, 100)
        Sigma_h = (s*np.ones_like(u)/np.pi)**(1/3)
        Va_h = (-t*np.ones_like(u)*4*np.pi)

        solution = self.solver.get_solution(best=best)
        Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_h,  Va_h, to_numpy=True)

        plt.figure(figsize=(6, 6))
        plt.title(rf'$\phi_h={2.17}$ ')
        plt.xlabel('u')

        plt.plot(u, Sigma, 'r-', label='Sigma')
        plt.plot(u_yago , Sigma_yago, 'r-.', label='Sigma_true')

        plt.plot(u, A, 'b-', label='A')
        plt.plot(u_yago, A_yago, 'b-.', label = 'A_true')


        plt.plot(u, phi, 'g-', label='phi')
        plt.plot(u_yago, phi_yago, 'g-.', label='phi_true')

        plt.tight_layout()
        plt.legend()
        plt.show()

    def render(self):

        self.plot_loss()
        self.plot_residuals()
        self.plot_result()
        self.compare_to_yago()

    def save_results(self, path):

        self.solver.save(path=path)
        with open(path, 'rb') as file:
            data = dill.load(file)
        os.remove(path)
        try:
            data['V_best'] = self.solver.callbacks[0].best_potential.state_dict()
            data['V_latest'] = self.V.state_dict()

        except:
            data['V_latest'] = self.V.state_dict()
        with open(path, 'wb') as file:
            dill.dump(data, file)

    def load_results(self, path):

        with open(path, 'rb') as file:
            data = dill.load(file)

        self.saved_data = data   
        
        try:     
            self.V.load_state_dict(data['V_best'])
        except:
            self.V.load_state_dict(data['V_latest'])

        train_generator = data['generator']['train']
        valid_generator = data['generator']['valid']
        de_system = data['diff_eqs']
        cond = data['conditions']
        nets = data['nets']
        best_nets = data['best_nets']
        train_loss = data['train_loss_history']
        valid_loss = data['valid_loss_history']
        optimizer = data['optimizer_class'](OrderedSet([p for net in data['nets'] + [self.V] for p in net.parameters()]))
        optimizer.load_state_dict(data['optimizer_state'])
        if data['generator']['train'].generator:
            t_min = data['generator']['train'].generator.__dict__['g1'].__dict__['t_min']
            t_max = data['generator']['train'].generator.__dict__['g1'].__dict__['t_max']
        else:
            t_min = data['generator']['train'].__dict__['g1'].__dict__['t_min']
            t_max = data['generator']['train'].__dict__['g1'].__dict__['t_max']

        self.solver = CustomBundleSolver1D( ode_system=de_system,
                                            conditions=cond,
                                            t_min=t_min,
                                            t_max=t_max,
                                            train_generator=train_generator,
                                            valid_generator=valid_generator,
                                            optimizer=optimizer,
                                            nets=nets,
                                            n_batches_valid=0,
                                            eq_param_index=(),
                                            V = self.V
                                        )

        if best_nets != None:
            self.solver.best_nets = best_nets
        self.solver.metrics_history['train_loss'] = train_loss
        self.solver.metrics_history['valid_loss'] = valid_loss
        self.solver.diff_eqs_source = data['diff_equation_details']['equation']