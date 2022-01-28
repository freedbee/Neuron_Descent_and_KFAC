from typing import NoReturn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


class MyLinear(torch.nn.Linear):
    """
    like nn.Linear, but with proper kaiming initialisation. 
    """
    def __init__(self, in_features, out_features, bias=True, nonlinearity='relu'):
        self.nonlinearity = nonlinearity
        super(MyLinear, self).__init__(in_features, out_features, bias)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        #proper kaiming init, which is not the default
        nn.init.kaiming_normal_(self.weight, nonlinearity=self.nonlinearity)
        if self.bias is not None:
            bound = 0
            nn.init.uniform_(self.bias, -bound, bound)


class MyConv2d(torch.nn.Conv2d):
    """
    like nn.Conv2d, but with proper kaiming initialisation. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
            dilation=1, groups=1, bias=True, padding_mode='zeros', nonlinearity='relu'):
        self.nonlinearity = nonlinearity
        super(MyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        #proper kaiming init, which is not the default
        nn.init.kaiming_normal_(self.weight, nonlinearity=self.nonlinearity)
        if self.bias is not None:
            bound = 0
            nn.init.uniform_(self.bias, -bound, bound)


class FOOFLayer(nn.Module):
    """"
    Abstract class for FOOF layers. This class supports natural gradient, 
    KFAC and FOOF computations.
    """
    def __init__(self):
        self.register_hooks()
        
        # Register buffer to store the current gradient estimate (including momentum).
        # This estimate can/will be pre-conditioned by FOOF/KFAC/Natural.
        self.register_buffer('sgd_momentum', torch.zeros_like(self.weight).detach())
        
        # Register Buffers to store the Kronecker Factors of KFAC, and their inverse.
        a = self.get_size_kf_A()
        self.register_buffer('kf_A', torch.zeros(size=(a,a)))
        self.register_buffer('kf_A_inv', torch.zeros(size=(a,a)))
        b = self.get_size_kf_E()
        self.register_buffer('kf_E', torch.zeros(size=(b,b)))
        self.register_buffer('kf_E_inv', torch.zeros(size=(b,b)))

        # Misc inits
        self.input_act_fisher = None
        self.output_grad_fisher = None
        self.kf_n = 0 
            
    def register_hooks(self):
        self.register_full_backward_hook(self._back_hook_fn)
        self.register_forward_hook(self._forward_hook_fn)
        self.fisher_hook = False
        self.forward_hook = False
        # For KFAC and naural, fisher_hook and forward_hook hook should be executed 
        # simultaneously. 
       
    def _forward_hook_fn(self, module, input_act, output_act):
        """
        The forward hook has the following functionality:
        - For FOOF+KFAC: It computes/updates the first Kronecker factor of KFAC.
        - For Natural+Natural_bd_: It stores information needed to perform implicit 
            fisher computations.
        Note: Both these functionalities are not necessarily executed at every step, 
        but are subject to amortisation, which are controlled by layer attributes
        self.fisher_hook, self.forward_hook.
        """
        if self._optimizer is None:
            return

        if (self._optimizer in ['natural', 'natural_bd']) and self.fisher_hook: 
            if self.input_act_fisher is None:
                self.input_act_fisher = input_act[0].detach()
            else:
                self.input_act_fisher = torch.cat((self.input_act_fisher, 
                                                    input_act[0].detach())) 
        
        if (self._optimizer in ['foof', 'kfac']) and self.forward_hook:
            self.update_kf_A(input_act)
    
    def _back_hook_fn(self, module, input_grad, output_grad):
        """
        The backward hook has the following functionality:
        Either:
            - It updates the (momentum) estimate of the gradients
            - For FOOF+KFAC+Natural_bd: It directly computes and applies the 
            parameter update.
            [Note:  The optimizer 'natural' is treated differently, since the 
            update cannot be computed for each layer. 
            See FOOFSequential.compute_update_natural() for details.]
        Or:
            - For KFAC: It computes/updates the second Kronecker factor of KFAC.
            - For Natural+Natural_bd: It stores information needed to perform implicit 
            fisher computations.
        Which of these two options is executed, depends on the attribute self.fisher_hook. 
        """
        if self._optimizer is None:
            return

        if not self.fisher_hook:
            self.sgd_momentum *= self.heavy_ball_m
            self.sgd_momentum += self.weight.grad
            if self._optimizer == 'foof':
                self.compute_foof_udpate()
                self.apply_update()
            if self._optimizer == 'kfac':
                self.compute_kfac_udpate()
                self.apply_update()
            if self._optimizer == 'natural_bd':
                self.compute_natural_bd_udpate()
                self.apply_update()
        else: 
            if self._optimizer in ['natural', 'natural_bd']:
                if self.output_grad_fisher is None:
                    self.output_grad_fisher = output_grad[0].detach()
                else:
                    self.output_grad_fisher = torch.cat((self.output_grad_fisher, 
                                                        output_grad[0].detach())) 
            if self._optimizer=='kfac':
                self.update_kf_E(output_grad)
         
    def compute_foof_udpate(self):
        grad = self.reshape_for_kf(self.sgd_momentum, opt='to_kf')
        update_direction = grad @ self.kf_A_inv 
        self.update_direction = self.reshape_for_kf(update_direction, opt='from_kf')
    
    def compute_kfac_udpate(self):
        grad = self.reshape_for_kf(self.sgd_momentum, opt='to_kf')
        if self.heuristic_damping:
            update_direction = self.kf_E_inv @ grad @ self.kf_A_inv 
        # Else, apply correct damping
        else: 
            # Transform gradient into KFE basis
            grad_kfe = (self.kf_UE @ grad @ self.kf_UA) 
            # Divide by damped eigenvalues of kronecker factored curvature
            grad_kfe /= self.kf_damped_eigvals
            # Transform back to standard basis
            update_direction = self.kf_UE.t() @ (grad_kfe) @ self.kf_UA.t()   
        self.update_direction = self.reshape_for_kf(update_direction, opt='from_kf')

    def apply_update(self):
        self.weight.requires_grad_(False)
        self.weight -= self.lr * self.update_direction
        self.weight.requires_grad_(True)

    def reshape_for_kf(self, grad, opt):
        """
        Function to reshape vectors, so that they can directly be 
        (matrix-)multiplied by Kronecker factors; the function should 
        also be able to invert this operation.
        Arguments:
            - grad: the vector that is reshaped. 
                (typically gradient or pre-conditioned gradient).
            - opt: expected to be 'to_kf' or 'from_kf'. 
                Determines whether vector is put into shape to be multiplied by kf, 
                or whether it is put back into its original shape (after it has been 
                multiplied by kronecker factors.)
        """
        raise NotImplementedError("This method needs to be implemented for FOOF/KFAC")

    def compute_natural_update(self, w):
        """"
        Arguments: 
        - w: This vector is assumed to be equal to:
            w = (\lamda*I + GˆT G)ˆ{-1} g
            where g is the gradient, and GGˆT is the (subsampled) Fisher.
        """
        raise NotImplementedError("This method needs to be implemented for natural")
    
    def compute_natural_bd_udpate(self):
        GTg = self.compute_GTg()
        w = torch.mv(self.gram_inv, GTg)
        self.compute_natural_update(w)
    
    def get_size_kf_A(self):
        raise NotImplementedError("This method needs to be implemented for FOOF/KFAC")

    def get_size_kf_E(self):
        raise NotImplementedError("This method needs to be implemented for KFAC")
    
    def update_kf_A(self, input_act):
        raise NotImplementedError("This method needs to be implemented for FOOF/KFAC")

    def update_kf_E(self, output_grad):
        raise NotImplementedError("This method needs to be implemented for KFAC")

    def compute_layer_gram(self):
        raise NotImplementedError("This method needs to be implemented for Natural")

    def compute_GTg(self):
        """"
        Computes matrix vector product GˆT g, where g is the gradient 
        and G GˆT is a factorisation of the (subsampled) Fisher.
        """
        raise NotImplementedError("This method needs to be implemented for Natural")
    
    def invert_kfs(self):
        lam = self.damp

        if self._optimizer == 'kfac' and self.heuristic_damping:
            pi = torch.sqrt((torch.trace(self.kf_A)/self.kf_A.shape[0]) / (torch.trace(self.kf_E)/self.kf_E.shape[0]) )
            self.kf_A_inv = torch.inverse(1/self.kf_n * self.kf_A  
                                + pi * np.sqrt(lam) * torch.eye(self.kf_A.shape[0], device=self.kf_A.device))
            self.kf_E_inv = torch.inverse(1/self.kf_n * self.kf_E  
                                + 1/pi * np.sqrt(lam) * torch.eye(self.kf_E.shape[0], device=self.kf_E.device))
        
        if self._optimizer == 'kfac'  and  (not self.heuristic_damping):
            self.kf_SA, self.kf_UA = torch.linalg.eigh(1/self.kf_n*self.kf_A)
            self.kf_SE, self.kf_UE = torch.linalg.eigh(1/self.kf_n*self.kf_E)
            eigvals = self.kf_SE.view(self.kf_SE.shape[0], 1) @ self.kf_SA.view(1, self.kf_SA.shape[0])
            self.kf_damped_eigvals = eigvals + lam
            
        if self._optimizer == 'foof':
            self.kf_A_inv = torch.inverse(1/self.kf_n * self.kf_A 
                                + lam * torch.eye(self.kf_A.shape[0], device=self.kf_A.device))

    
class FOOFLinear(FOOFLayer, MyLinear):
    """
    Linear Layer supporting natural gradient, KFAC and FOOF computations.
    """
    def __init__(self, n_in_features, n_out_features, bias=True, 
                 nonlinearity='relu'):
        MyLinear.__init__(self, n_in_features, n_out_features, 
                            bias=bias, nonlinearity=nonlinearity)
        FOOFLayer.__init__(self)
        
    def reshape_for_kf(self, grad, opt):
        return grad
        
    def compute_natural_update(self, w):
        lam = self.damp
        self.update_direction =  1/lam * self.sgd_momentum \
                                 - 1/(lam**2) * torch.mm(self.output_grad_fisher.t(), \
                                    self.input_act_fisher * w.unsqueeze(dim=1) ) 
            
    def get_size_kf_A(self):
        return self.weight.shape[1]
      
    def get_size_kf_E(self):
        return self.weight.shape[0]
       
    def update_kf_A(self, input_act):
        input_act = input_act[0].detach()
        if self.kf_n_update_data is not None:
            input_act = input_act[:self.kf_n_update_data]
        self.kf_n *= self.kf_m
        self.kf_n += (1-self.kf_m) * 1.
        self.kf_A *= self.kf_m
        self.kf_A += (1-self.kf_m) * torch.mm(input_act.t(), input_act)
        
    def update_kf_E(self, output_grad):
        output_grad = output_grad[0].detach()
        if self.kf_n_update_data is not None:
            output_grad = output_grad[:self.kf_n_update_data]
        self.kf_E *= self.kf_m
        self.kf_E += (1-self.kf_m) * torch.mm(output_grad.t(), output_grad)
        
        
    def compute_layer_gram(self):
        self.gram = torch.mm(self.input_act_fisher, self.input_act_fisher.t())   \
                    * torch.mm(self.output_grad_fisher, self.output_grad_fisher.t())
        return self.gram
        
    def compute_GTg(self):
        return ((self.output_grad_fisher @ self.sgd_momentum) * self.input_act_fisher).sum(dim=1)
        # Equivalently:
        # return torch.diag(self.output_grad_fisher @ self.sgd_momentum @ self.input_act_fisher.t())

    
class FOOFConv2d(FOOFLayer, MyConv2d):
    """
    Conv2d Layer, supporting natural gradient, KFAC and FOOF computations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
            dilation=1, groups=1, bias=True, padding_mode='zeros', nonlinearity='relu'):
        MyConv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, nonlinearity='relu')
        FOOFLayer.__init__(self)
        
        self.unfold = torch.nn.Unfold(kernel_size, stride=stride,\
                                        padding=padding)
    
    def reshape_for_kf(self, grad, opt):
        if opt=='to_kf':
            return grad.view(self.weight.shape[0], -1)
        if opt=='from_kf':
            return grad.view(self.weight.shape)
        
    def compute_natural_update(self, w):
        lam = self.damp
        x = (self.all_grad_f * w[:, None, None]).sum(dim=0).view(self.weight.shape)
        self.update_direction = 1/lam * self.sgd_momentum \
                                - 1/(lam**2) *  x
                    
    def get_size_kf_A(self):
        return self.weight.shape[1]*self.weight.shape[2]*self.weight.shape[3]
      
    def get_size_kf_E(self):
        return self.weight.shape[0]
       
    def update_kf_A(self, input_act):
        input_act = input_act[0].detach()
        if self.kf_n_update_data is not None:
            input_act = input_act[:self.kf_n_update_data]
        a = self.unfold(input_act)
        self.kf_A *= self.kf_m
        self.kf_A += (1-self.kf_m) * torch.bmm(a, a.transpose(1,2)).mean(dim=0)
        self.kf_n *= self.kf_m
        self.kf_n += (1-self.kf_m) * 1.
        
    def update_kf_E(self, output_grad):
        output_grad = output_grad[0].detach()
        if self.kf_n_update_data is not None:
            output_grad = output_grad[:self.kf_n_update_data]
        e = output_grad.view(output_grad.shape[0], output_grad.shape[1], -1)
        self.kf_E *= self.kf_m
        self.kf_E += (1-self.kf_m) * torch.bmm(e, e.transpose(1,2)).sum(dim=0)
        
    def compute_layer_gram(self):
        a = self.unfold(self.input_act_fisher).transpose(1,2)
        e = self.output_grad_fisher.view(self.output_grad_fisher.shape[0], 
                                         self.output_grad_fisher.shape[1], -1)
        self.all_grad_f = e.bmm(a)
        ag = self.all_grad_f.view(self.input_act_fisher.shape[0], -1)
        self.gram = torch.mm(ag, ag.t())
        return self.gram
        
    def compute_GTg(self):
        ag = self.all_grad_f.view(self.input_act_fisher.shape[0], -1)
        return torch.mv(ag, self.sgd_momentum.view(-1))
        



class FOOFSequential(nn.Sequential):
    """"Network which can use optimizers FOOF, KFAC or Natural."""
    def __init__(self, module_list, optimizer=None, lr=3e1, damp=1e2, momentum=0.0, kf_m=0.95, 
                heuristic_damping=True, inversion_period=100, 
                kf_n_update_data=None, kf_n_update_steps=None,
                mc_fisher=True, output_dim=10):
        """"
        Arguments:
        - Optimizer: Can Be None, foof, kfac, natural, natural_bd. 
            (where 'bd' is short for block diagonal).
            If None, a standard PyTorch Optimizer can be used to optimize that net
            and all other arguments are ignored. 
            If not None, the network can be optimized by calls to the method 
            "parameter_update".
            Note: The natural gradient method does not use the same mini-batch to estimate gradients 
            and Fisher. Rather, it uses independent mini-batches. In this implementation, it uses
            a previous mini-batch. See paper for details and ablation. The experiments in the
            paper use a slightly different implementation, and completely independent mini-batches.
        
        The following arguments have an effect as long as optimzier is not None:
        - lr: (inital) learning rate.
        - damp: (initial) damping term. 
        Note: It is reasonable to think of the ratio lr/damp as analogous to the 
            learning rate of SGD.
        - inversion_period: Determines how frequently kronecker factors are inverted or
            how frequently the (block diagonal) fisher is updated (and implicilty inverted).
            Choosing a large value leads to faster runtimes, potentially at the risk of 
            slightly stale information.
        - momentum: heavy ball momentum of SGD gradient estimates. These estimates will
            be pre-conditioned with FOOF/KFAC/Natural(_bd).

        The following arguments are only used if natural/natural_bd is used:
        - MC_fisher: If True, the Fisher is approximated by drawing one label from the model
            distribution per datapoint. 
            Else, the Fisher is computed fully (on the current minibatch).
        - output_dim: Number of labels of the dataset. Is required to compute the Full Fisher. 
      
        The following arguments only have an effect, if used in conjunction with foof/kfac.
        In particular, they have no effect for natural, and natural_bd methods.
        - kf_m: Momentum used for EMA of kronecker factors in FOOF/KFAC.
        - kf_n_update_data: Determines how many datapoints from a given batch are used to
            update kronecker factors. If None, entire batch is used. This may consume 
            a lot of memory, especially for conv layers. 
            Chosing a small value reduces memory footprint at the risk of slightly inaccurate
            kronecker factors.
        - kf_n_update_steps: Determines how many batches are used to update kronecker factors. 
            kf_n_update_steps corresponds to variable 'S' in pseudocode in paper, 
            see there for details.
            If None, all batches are used.  
            Choosing a small value speeds up computations, at the risk of stale kronecker 
            factors.
        - heuristic damping: Determines whether KFAC is used with heuristic or correct
            damping. See paper for details. 
            This argument has no effect, if optimizer != kfac.
        """
        super(FOOFSequential, self).__init__(*module_list)
        self.set_damp(damp) 
        self.set_lr(lr)
        self.set_momentum(momentum)
        self.set_kf_m(kf_m)
        self.set_heuristic_damping(heuristic_damping)
        self.set_kf_n_update_data(kf_n_update_data)
        self.set_inversion_period(inversion_period)
        self.set_kf_n_update_steps(kf_n_update_steps)
        self.set_MC_fisher(mc_fisher)
        self.set_output_dim(output_dim)
        self.set_optimizer(optimizer)
        self.set_hooks(forward_hook=False, fisher_hook=False)
        
          
    def set_momentum(self, momentum):
        self._heavy_ball_m = momentum
        for layer in self.my_modules():
            layer.heavy_ball_m = self._heavy_ball_m

    def set_lr(self, lr):
        self._lr = lr
        for layer in self.my_modules():
            layer.lr = self._lr

    def set_damp(self, damp):
        self._damp = damp
        for layer in self.my_modules():
            layer.damp = self._damp

    def set_kf_m(self, kf_m):
        self._kf_m = kf_m
        for layer in self.my_modules():
            layer.kf_m = self._kf_m

    def set_heuristic_damping(self, hd):
        self._heuristic_damping = hd
        for layer in self.my_modules():
            layer.heuristic_damping = self._heuristic_damping

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        for layer in self.my_modules():
            layer._optimizer = self._optimizer

    def set_inversion_period(self, inversion_period):
        self._update_counter = 0
        self._inversion_period = inversion_period
        for layer in self.my_modules():
            layer.inversion_period = self._inversion_period

    def set_kf_n_update_data(self, kf_n_update_data):
        self._kf_n_update_data = kf_n_update_data
        for layer in self.my_modules():
            layer.kf_n_update_data = self._kf_n_update_data

    def set_kf_n_update_steps(self, kf_n_update_steps):
        self.kf_n_update_steps = kf_n_update_steps
        
    def set_MC_fisher(self, MC_fisher):
        self.MC_fisher = MC_fisher
    
    def set_output_dim(self, output_dim):
        self.output_dim = output_dim
    
    def print_optimization_setting(self):
        raise NotImplementedError

    def set_hooks(self, fisher_hook=None, forward_hook=None):
        if fisher_hook is not None:
            self.fisher_hook = fisher_hook
        if forward_hook is not None:
            self.forward_hook = forward_hook
        for layer in self.my_modules():
            layer.fisher_hook = self.fisher_hook
            layer.forward_hook = self.forward_hook
    
    def get_device(self):
        return next(self.parameters()).device
    
    def parameter_update(self, X, y, yf=None):
        """
        Arguments: 
        - X,y: Standard input and labels
        - yf: Labels used to compute the Fisher (if applicable).
            If None, then labels will be sampled from model distribution.
        """
        self._update_counter += 1
        if self._optimizer in ['natural', 'natural_bd']:
            self.parameter_update_natural(X,y, yf=yf)
        if self._optimizer in ['foof', 'kfac']:
            self.parameter_update_foof_kfac(X,y, yf=yf)
    
    def parameter_update_foof_kfac(self, X, y, yf=None):
        # Do for- and backward pass to estimate the Fisher (if needed).
        if self._optimizer == 'kfac' and self.forward_hook: 
            self.set_hooks(fisher_hook=True)
            criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            X.requires_grad_(True)
            output = self(X)
            if yf is None:
                self.sampled_labels = Categorical(logits=output).sample()
            else:
                self.sampled_labels = yf
            loss = criterion(output, self.sampled_labels) / torch.sqrt(torch.tensor(1.0*X.shape[0]))
            loss.backward(retain_graph=True)
            self.set_hooks(fisher_hook=False)
            self.zero_grad()

        # Do standard for- and backward pass to get gradients.
        criterion = torch.nn.CrossEntropyLoss()
        X.requires_grad_(True)
        if self._optimizer != 'kfac' or not self.forward_hook:
            output = self(X)
        loss = criterion(output, y)
        loss.backward()
        X.requires_grad_(False)
        self.zero_grad()

        # After Update has been computed, update KFs and amortisation settings.
        T = self._update_counter % self._inversion_period 
        if T == 0:
            self.invert_kfs()
            if self.kf_n_update_steps is not None:
                self.set_hooks(forward_hook=False)
        if self.kf_n_update_steps is not None:
            if T  == (self._inversion_period - self.kf_n_update_steps):
                self.set_hooks(forward_hook=True)
    
    def update_fisher(self, X, y=None):
        # Delete old information for Fisher:
        for layer in self.my_modules():
            layer.input_act_fisher = None
            layer.output_grad_fisher = None
        
        # Accumulate Gradients required to compute Fisher.
        # Either use MC Fisher
        if self.MC_fisher:
            self.set_hooks(fisher_hook=True)
            criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            X.requires_grad_(True)
            output = self(X)
            if y is None:
                self.sampled_labels = Categorical(logits=output).sample()
            else:
                self.sampled_labels = y
            loss = criterion(output, self.sampled_labels) / torch.sqrt(torch.tensor(1.0*X.shape[0]))
            loss.backward()
            X.requires_grad_(False)
            self.zero_grad()
            self.set_hooks(fisher_hook=False)
        # Or use the 'Full' Fisher.
        else: 
            output_dim = self.output_dim
            for layer in self.my_modules():
                self.set_hooks(fisher_hook=True)
            log_soft = torch.nn.LogSoftmax(dim=1)
            for label in range(output_dim):
                X1 = X.clone()
                X1.requires_grad_(True)
                output = self(X1)
                probs = torch.nn.functional.softmax(output,dim=1).detach()
                log_probs = log_soft(output)
                weights = torch.sqrt(probs[:,label])
                loss = -torch.dot(weights, log_probs[:,label]) / np.sqrt(X1.shape[0])
                loss.backward()
                self.zero_grad()
            X.requires_grad_(False)
            self.set_hooks(fisher_hook=False)
        
        # compute and invert gram matrix/matrices
        self.gram = None
        for layer in self.my_modules():
            if self.gram is None:
                self.gram = layer.compute_layer_gram()
            else:
                self.gram += layer.compute_layer_gram()
            if self._optimizer == 'natural_bd':
                layer.gram_inv = torch.inverse(1/layer.damp * layer.gram 
                                    + torch.eye(layer.gram.shape[0],device=layer.gram.device))
        if self._optimizer == 'natural':
            self.gram_inv = torch.inverse(1/self._damp * self.gram 
                                    + torch.eye(self.gram.shape[0],device=self.gram.device))
        
    def parameter_update_natural(self, X, y, yf=None):
        # Compute gradient in standard for- and backward pass.
        self.set_hooks(fisher_hook=False)
        criterion = torch.nn.CrossEntropyLoss()
        X.requires_grad_(True)
        output = self(X)
        loss = criterion(output, y)
        loss.backward()
        X.requires_grad_(False)
        self.zero_grad()

        # For natural_bd the parameter updates are computed in the backward_hook. 
        # For natural, accumulate information across layers and compute update:
        if self._optimizer == 'natural':
            GTg = torch.zeros(self.gram_inv.shape[0], device=self.get_device())
            for layer in self.my_modules():
                GTg += layer.compute_GTg()
            w = torch.mv(self.gram_inv, GTg)
            for layer in self.my_modules():
                layer.compute_natural_update(w)
                layer.apply_update()
        
        # After update:
        if self._update_counter % self._inversion_period == 0:
            self.update_fisher(X, y=yf)



    def initialise_kf_fisher(self, trainloader, T=10):
        if self._optimizer in ['natural', 'natural_bd']:
            X = None
            for XX, y in trainloader:
                X = XX.to(self.get_device())
                break
            self.update_fisher(X)
            return
        self.set_hooks(forward_hook=True)
        if self._optimizer == 'kfac':
            self.set_hooks(fisher_hook=True)
        for t, (X, _) in enumerate(trainloader):
            if self._optimizer == 'kfac':
                X.requires_grad_(True)
            output = self(X.to(self.get_device()))
            if self._optimizer == 'kfac':
                criterion = torch.nn.CrossEntropyLoss(reduction='sum')
                self.sampled_labels = Categorical(logits=output).sample()
                loss = criterion(output, self.sampled_labels) / torch.sqrt(torch.tensor(1.0*X.shape[0]))
                loss.backward()
                self.zero_grad()
                X.requires_grad_(True)
            if t==T:
                break
        self.invert_kfs()
        self.set_hooks(forward_hook=False, fisher_hook=False)
        if self.kf_n_update_steps is None:
            self.set_hooks(forward_hook=True, fisher_hook=False)

    def invert_kfs(self, lam=None):
        for layer in self.my_modules():
            layer.invert_kfs()

    def my_modules(self):
        """"
        Returns list of modules that will/can be optimized 
        by FOOF, KFAC or Natural.
        Current implementation only includes Linear and Conv Layers
        without biases. 
        In particular, batch norm layers are not optimized.
        """
        out = []
        for mod in self.modules():
            if isinstance(mod, FOOFLayer):
                out.append(mod)
        return out
