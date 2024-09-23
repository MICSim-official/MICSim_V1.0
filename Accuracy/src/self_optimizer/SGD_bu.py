import torch
from torch.optim.optimizer import Optimizer,required
from layers.WA import wage_quantizer

class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, adaptive_lr = False, quantize_momentum = False, momentum_wl=32):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.adaptive_lr = adaptive_lr
        self.quantize_momentum = quantize_momentum
        self.momentum_wl = momentum_wl
        self.nl_array_list = None
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None,batch_idx = 0,epoch=0,grad_scale=1,args=None,logger =None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        counter = 1
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            if self.nl_array_list is None and args.nl_level==-1:
                self.nl_array_list = []
                for i,p in enumerate(group['params']):
                    self.nl_array_list.append(nonlinear_array.remap_weight_gpu(list(p.shape)))

            for i,p in enumerate(group['params']):
                if p.grad is None:
                    continue

                #normalize gradien outside QG for following calculation range
                if self.quantize_momentum is True:
                    d_p = wage_quantizer.QE(p.grad.data,args.wl_grad)
                else:
                    d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    #print("here")
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                        #print(buf.shape, buf.mean(), buf.std(), buf.max(), buf.min())
                    else:
                        if self.quantize_momentum is True:
                            buf = param_state['momentum_buffer']
                            v_new = buf*momentum+(1-dampening)*d_p
                            param_state['momentum_buffer'] = buf = wage_quantizer.QM(v_new, self.momentum_wl)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                    #print(buf.std(),buf.mean(),buf.max(),buf.min())

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf


                if self.adaptive_lr:
                    param_state = self.state[p]
                    if 'lr_buffer' not in param_state:
                        buf = param_state['lr_buffer'] = torch.ones_like(p.data)
                        param_state['sign_buffer'] = torch.sign(d_p.data)
                        d_p = buf*d_p
                    else:
                        buf = param_state['lr_buffer']
                        sign_buf = param_state['sign_buffer'].clone()
                        param_state['sign_buffer'] = torch.sign(d_p.data)
                        scale_n = scale_p = sign_buf*param_state['sign_buffer']
                        #for sign changed case,scale down
                        scale_n = scale_n*(-0.875)
                        scale_n[scale_n<=0] = 1
                        buf.mul_(scale_n)

                        #for sign unchanged case, increse
                        scale_p[scale_p < 0] = 0
                        #print(scale_p.max())
                        buf.add_(0.125*scale_p)
                        buf = torch.clamp(buf,0.1,10)
                        #print(buf.max(),buf.min())
                        param_state['lr_buffer'] = buf
                        d_p = buf*d_p


                delta_weight = - wage_quantizer.QG(d_p.clone(), grad_scale, args.wl_grad)

                if args.debug == 1:
                        logger( "before NL: delta weight std {:.5f} delta weight max {:.5f} delta weight min {:.5f} delta weight abs mean {:,.5f}".format(delta_weight.std(), delta_weight.max(),delta_weight.min(), delta_weight.abs().mean() ))
                if args.nl_level > 0:
                    d_p = nonlinear.remap_delta_weight_gpu(delta_weight,p.data,args)
                elif args.nl_level==-1:
                    d_p = self.nl_array_list[i].foward(delta_weight,p.data)
                else:
                    d_p = delta_weight

                if args.c2c_variation_std > 0:
                    bias = torch.randn_like(d_p)*args.c2c_variation_std*2
                    bias[d_p==0] = 0
                    temp = d_p.clone()
                    d_p = d_p + bias
                if args.debug == 1:
                        logger("After NL: delta weight std {:.5f} delta weight max {:.5f} delta weight min {:.5f} delta weight abs mean {:,.5f}".format(d_p.std(), d_p.max(), d_p.min(), d_p.abs().mean()))

                if 0 and batch_idx % 100 == 0:
                    weight_vector = p.cpu().data.numpy().flatten()
                    delta_weight_before_vector = -delta_weight.cpu().data.numpy().flatten()
                    delta_weight_after_vector = -d_p.cpu().data.numpy().flatten()
                    name = 'layer' + str(counter)
                    h1_name = args.logdir + '/delta_weight_stat/before_nl_delta_weight_statistics_' + str(name) + str(
                        epoch) + str(batch_idx) + '.npy'
                    h2_name = args.logdir + '/delta_weight_stat/after__nl_delta_weight_statistics_' + str(name) + str(
                        epoch) + str(batch_idx) + '.npy'
                    x1_name = args.logdir + '/delta_weight_stat/before_nl_x_edges.npy'
                    y1_name = args.logdir + '/delta_weight_stat/before_nl_y_edges.npy'
                    x2_name = args.logdir + '/delta_weight_stat/after__nl_x_edges.npy'
                    y2_name = args.logdir + '/delta_weight_stat/after__nl_y_edges.npy'
                    name_array = [h1_name, x1_name, y1_name, h2_name, x2_name, y2_name]
                    hist3d.hist3d_2_save_file(weight_vector, delta_weight_before_vector, delta_weight_after_vector,
                                              name_array, args.wl_grad, 5)
                    counter = counter + 1
                if 0 and batch_idx % 100 == 0:
                    weight_vector = p.cpu().data.numpy().flatten()
                    delta_weight_before_vector = -delta_weight.cpu().data.numpy().flatten()
                    delta_weight_bv_vector = temp.cpu().data.numpy().flatten()
                    delta_weight_after_vector = d_p.cpu().data.numpy().flatten()
                    name = 'layer' + str(counter)
                    h1_name = args.logdir + '/delta_weight_stat/before_nl_delta_weight_statistics_' + str(name) +"_"+ str(
                        epoch) +"_"+ str(batch_idx) + '.npy'
                    h2_name = args.logdir + '/delta_weight_stat/after__nl_delta_weight_statistics_' + str(name) + str(
                        epoch) + str(batch_idx) + '.npy'

                    hist3d.hist2d_save_file2(weight_vector,delta_weight_bv_vector,delta_weight_after_vector,h1_name,h2_name,args.wl_grad)
                    counter = counter + 1
                #print(d_p.shape, d_p.mean(), d_p.std(), d_p.max(), d_p.min())
                p.data.add_(1, d_p)

        return loss
