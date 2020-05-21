import os
import time
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

# custom packages
import PatchAttack.utils as utils
from PatchAttack.PatchAttack_config import PA_cfg

# global variables
torch_cuda = 0


class robot():
    
    class p_pi(nn.Module):
        '''
        policy (and value) network
        '''
        def __init__(self, space, embedding_size=30, stable=True, v_theta=False):
            super().__init__()
            self.embedding_size = embedding_size
            embedding_space = [224] + space[:-1]
            # create embedding space
            self.embedding_list = nn.ModuleList([nn.Embedding(embedding_space[i], self.embedding_size) 
                                                 for i in range(len(embedding_space))])
            if stable:
                self._stable_first_embedding()
            # create linear heads
            self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)#(batch, seq, features)
            self.linear_list = nn.ModuleList([nn.Linear(self.embedding_size, space[i])
                                              for i in range(len(space))])
            # create v_theta head, actor-critic mode
            self.v_theta = v_theta
            if self.v_theta:
                self.theta = nn.ModuleList([nn.Linear(self.embedding_size, 1)
                                            for i in range(len(space))])
            # set necessary parameters
            self.stage = 0
            self.hidden = None

        def forward(self, x):
            x = self.embedding_list[self.stage](x)
            # extract feature of current state
            x, self.hidden = self.lstm(x, self.hidden) # hidden: hidden state plus cell state
            # get action prob given the current state
            prob = self.linear_list[self.stage](x.view(x.size(0), -1))
            # get state value given the current state
            if self.v_theta:
                value = self.theta[self.stage](x.view(x.size(0), -1))
                return prob, value
            else:
                return prob

        def increment_stage(self):
            self.stage +=1

        def _stable_first_embedding(self):
            target = self.embedding_list[0]
            for param in target.parameters():
                param.requires_grad = False

        def reset(self):
            '''
            reset stage to 0
            clear hidden state
            '''
            self.stage = 0
            self.hidden = None
    
    def __init__(self, critic, space, rl_batch, gamma, lr, 
                 stable=True):
        # policy network
        self.critic = critic
        self.mind = self.p_pi(space, stable=stable, v_theta=critic)
        
        # reward setting
        self.gamma = gamma # back prop rewards
        # optimizer
        self.optimizer = optim.Adam(self.mind.parameters(), lr=lr)
        
        # useful parameters
        self.combo_size = len(space)
        self.rl_batch = rl_batch
        
    def select_action(self, state):
        '''generate one parameter
        input:
        state: torch.longtensor with size (bs, 1), the sampled action at the last step
        
        return:
        action: torch.longtensor with size (bs, 1)
        log_p_action: torch.floattensor with size (bs, 1)
        value: [optional] torch.floattensor with size (bs, 1)
        '''
        if self.critic:
            p_a, value = self.mind(state)
            p_a = F.softmax(p_a, dim=1)
            
            # select action with prob
            dist = Categorical(probs=p_a)
            action = dist.sample()
            log_p_action = dist.log_prob(action)
            
            return action.unsqueeze(-1), log_p_action.unsqueeze(-1), value
        else:
            p_a = F.softmax(self.mind(state), dim=1)
            
            # select action with prob
            dist = Categorical(probs=p_a)
            action = dist.sample()
            log_p_action = dist.log_prob(action)
            
            return action.unsqueeze(-1), log_p_action.unsqueeze(-1)
    
    def select_combo(self):
        '''generate the whole sequence of parameters
        
        return:
        combo: torch.longtensor with size (bs, space.size(0): 
               (PREVIOUS STATEMENT) num_occlu \times 4 or 7 if color==True)
        log_p_combo: torch.floattensor with size (bs, space.size(0))
        rewards_critic: torch.floatensor with size (bs, space.size(0))
        '''
        state = torch.zeros((self.rl_batch, 1)).long().cuda(torch_cuda)
        combo = []
        log_p_combo = []
        if self.critic:
            # plus r_critic
            rewards_critic = []
            for _ in range(self.combo_size):
                action, log_p_action, r_critic = self.select_action(state)
                combo.append(action)
                log_p_combo.append(log_p_action)
                rewards_critic.append(r_critic)
                
                state = action
                self.mind.increment_stage()
            
            combo = torch.cat(combo, dim=1)
            log_p_combo = torch.cat(log_p_combo, dim=1)
            rewards_critic = torch.cat(rewards_critic, dim=1)
            
            return combo, log_p_combo, rewards_critic
        else:
            for _ in range(self.combo_size):
                action, log_p_action = self.select_action(state)
                combo.append(action)
                log_p_combo.append(log_p_action)
                
                state = action
                self.mind.increment_stage()
            
            combo = torch.cat(combo, dim=1)
            log_p_combo = torch.cat(log_p_combo, dim=1)
            
            return combo, log_p_combo


class MPA_agent(robot):
    
    def __init__(self, model, image_tensor, target_tensor, num_occlu, color, sigma, shrink=1):
        '''
        the __init__ function needs to create action space because this relates with 
        the __init__ of the policy network 
        '''
        # build environment
        self.model = model
        self.image_tensor = image_tensor
        self.target_tensor = target_tensor
        # build action space
        self.num_occlu = num_occlu
        self.color = color
        self.space = self.create_searching_space(num_occlu, color, 
                                                 H=image_tensor.size(-2), 
                                                 W=image_tensor.size(-1))
        self.shrink = shrink
        #print('environment and action space have been determined')
        # specific reward param
        self.sigma = sigma
        #print('remember to build robot')
        
    def build_robot(self, critic, rl_batch, gamma, lr, stable=True):
        super().__init__(critic, self.space, rl_batch, gamma, lr, stable)
        #print('robot built!')
        
    @staticmethod
    def create_searching_space(num_occlu, color=False, H=224, W=224, R=255, G=255, B=255):
        '''
        input: 
        num_occlu: the number of occlusion masks
        color: wheather to optimize the color, if it is true, 
               7 parameters for each occlusion mask
        H, W: for futher decrease the color size

        notice: when parameterizing the mask, height comes first. e.g. c_dim. After consideration, 
                I decide to use two coordinate pairs to parameterize the mask.

        return: list with size 7*num_occlu if color else 4*num_occlu, each item indicates the option number
 
        '''
        # limit search space if H!=W which relates to the create_mask function
        if W > H:
            W = W // self.shrink
        elif H > W:
            H = H // self.shrink
        
        # create space
        search_space = []
        if color:
            for n in range(num_occlu):
                search_space += [int(H), int(W), int(H), int(W), 
                                 int(R), int(G), int(B)]
        else:
            for n in range(num_occlu):
                search_space += [int(H), int(W), int(H), int(W)]
        return search_space
    
    @staticmethod
    def create_mask(points, distributed_area=False, distributed_mask=False, H=224, W=224):
        '''
        input: 
        points: the pixel coordinates in the image, torch.LongTensor 
                with size (bs, num_occlu \times 4 or 7 if color is true)
                if points.size(-1) is a multiple of 7, then distributed_mask=True
        distributed_area: flag, if it is true, calculate the distributed combined areas
        distributed_mask: flag, if it is true, calculate the distributed masks

        return: 
        mask: torch.floattensor with size (bs, 224, 224)
        mask: [optional, distributed_mask=True]torch.floattensor with size (bs, num_occlu, 224, 224) 
        area: torch.floattensor with size (bs, 1)
        area: [optional, distributed_area=True]: torch.floattensor with size (bs, num_occlu)
              the cth col records the occlued area when applying c occlusions on the masks.

        cost a lot of time. I have updated the procedure to generate the mask, but still kind of slow.
        It may less than 1.5 seconds to generate all the masks given combo, which means currently it only
        takes around 3.2 seconds to learn the policy network and the value network.
        '''       
        
        bs = points.size(0)
        total = points.size(-1)
        
        if total % 4 == 0:
            p_l = 4
            num_occlu = total // 4
        elif total % 7 == 0:
            p_l = 7
            num_occlu = total // 7
            assert distributed_mask == True,\
            'accourding to num_occlu, distributed_mask should be true'
        else:
            assert False,\
            'occlusion num should be a multiple of 4 or 7'

        # post process combo
        p_combo = []
        for o in range(num_occlu):
            p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o*p_l+0, o*p_l+2]).cuda(torch_cuda))
                           .sort(dim=1, descending=False)[0])
            p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o*p_l+1, o*p_l+3]).cuda(torch_cuda))
                           .sort(dim=1, descending=False)[0])
        p_combo = torch.cat(p_combo, dim=1)
        
        if distributed_area:
            if distributed_mask:
                
                mask = torch.ones((bs, num_occlu, H, W))
                area = torch.zeros(bs, num_occlu)
                
                # make masks
                if H > W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item, o][p_combo[item][o*4+0]*self.shrink:p_combo[item][o*4+1]*self.shrink+1, 
                                       p_combo[item][o*4+2]:p_combo[item][o*4+3]+1] = 0.
                            area[item, o] = (mask[item, o]==0.).sum()
                elif H < W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item, o][p_combo[item][o*4+0]:p_combo[item][o*4+1]+1, 
                                       p_combo[item][o*4+2]*self.shrink:p_combo[item][o*4+3]*self.shrink+1] = 0.
                            area[item, o] = (mask[item, o]==0.).sum()
                elif H == W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item, o][p_combo[item][o*4+0]:p_combo[item][o*4+1]+1, 
                                       p_combo[item][o*4+2]:p_combo[item][o*4+3]+1] = 0.
                            area[item, o] = (mask[item, o]==0.).sum()

                return mask, area
            else:
                
                mask = torch.ones((bs, H, W))
                area = torch.zeros(bs, num_occlu)

                # make masks
                if H > W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item][p_combo[item][o*4+0]*self.shrink:p_combo[item][o*4+1]*self.shrink+1, 
                                       p_combo[item][o*4+2]:p_combo[item][o*4+3]+1] = 0.
                            area[item, o] = (mask[item]==0.).sum()
                elif H < W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item][p_combo[item][o*4+0]:p_combo[item][o*4+1]+1, 
                                       p_combo[item][o*4+2]*self.shrink:p_combo[item][o*4+3]*self.shrink+1] = 0.
                            area[item, o] = (mask[item]==0.).sum()
                elif H == W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item][p_combo[item][o*4+0]:p_combo[item][o*4+1]+1, 
                                       p_combo[item][o*4+2]:p_combo[item][o*4+3]+1] = 0.
                            area[item, o] = (mask[item]==0.).sum()
                
                return mask, area
            
        else:
            if distributed_mask:
                
                mask = torch.ones((bs, num_occlu, H, W))
                area = torch.zeros(bs, 1)
                mask_overlay = torch.ones(H, W)

                # make masks
                if H > W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item, o][p_combo[item][o*4+0]*self.shrink:p_combo[item][o*4+1]*self.shrink+1, 
                                       p_combo[item][o*4+2]:p_combo[item][o*4+3]+1] = 0.
                            mask_overlay = mask_overlay * mask[item, o] # calculate overlay
                        area[item] = (mask_overlay==0.).sum()
                        mask_overlay.fill_(1.)
                elif H < W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item, o][p_combo[item][o*4+0]:p_combo[item][o*4+1]+1, 
                                       p_combo[item][o*4+2]*self.shrink:p_combo[item][o*4+3]*self.shrink+1] = 0.
                            mask_overlay = mask_overlay * mask[item, o] # calculate overlay
                        area[item] = (mask_overlay==0.).sum()
                        mask_overlay.fill_(1.)
                elif H == W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item, o][p_combo[item][o*4+0]:p_combo[item][o*4+1]+1, 
                                       p_combo[item][o*4+2]:p_combo[item][o*4+3]+1] = 0.
                            mask_overlay = mask_overlay * mask[item, o] # calculate overlay
                        area[item] = (mask_overlay==0.).sum()
                        mask_overlay.fill_(1.)
                    
                return mask, area
            else:
                
                mask = torch.ones((bs, H, W))
                area = torch.zeros(bs, 1)
                
                # make masks
                if H > W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item][p_combo[item][o*4+0]*self.shrink:p_combo[item][o*4+1]*self.shrink+1, 
                                       p_combo[item][o*4+2]:p_combo[item][o*4+3]+1] = 0.
                        area[item] = (mask[item]==0.).sum()
                elif H < W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item][p_combo[item][o*4+0]:p_combo[item][o*4+1]+1, 
                                       p_combo[item][o*4+2]*self.shrink:p_combo[item][o*4+3]*self.shrink+1] = 0.
                        area[item] = (mask[item]==0.).sum()
                elif H == W:
                    for item in range(bs):
                        for o in range(num_occlu):
                            mask[item][p_combo[item][o*4+0]:p_combo[item][o*4+1]+1, 
                                       p_combo[item][o*4+2]:p_combo[item][o*4+3]+1] = 0.
                        area[item] = (mask[item]==0.).sum()
                        
                return mask, area
    
    @staticmethod
    def create_painting(combo):
        '''
        create the RGB painting for the masks

        input:
        combo: torch.longtensor with size (bs, space.size(0): 
               num_occlu \times 7)

        return:
        RGB_painting: torch.floattensor with size (bs, num_occlu, 3). 
                      The last dim is RGB three channels respectively.
        '''
        assert combo.size(-1) % 7 == 0,\
        'occlusion_num should be a multiple of 4 or 7 when creating painting'
        
        num_occlu = combo.size(-1) // 7
        RGB_painting = torch.cat([combo[:, i*7+4:i*7+7] for i in range(num_occlu)], dim=1)
        
        return RGB_painting.view(combo.size(0), num_occlu, 3)
        
    @staticmethod
    def paint(input_tensor, mask, RGB_painting):
        '''
        input:
        input_tensor: torch.floattensor with size (bs, 3, H=224, W=224)
        mask: torch.floattensor with size (bs, num_occlu, H=224, W=224)
        RGB_painting: torch.floattensor with size (bs, num_occlu, 3)
        
        return:
        painted_input_tensor with size (bs, 3, H=224, W=224)
        '''
        # change the RGB value to input_tensor-scale value
        RGB_painting = RGB_painting.clone().float()
        RGB_painting = RGB_painting / 255.
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda(torch_cuda)
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda(torch_cuda)
        RGB_painting = (RGB_painting - mean) / std
        
        painted_input_tensor = input_tensor.clone()
        painting_mask = 1 - mask
        bs = input_tensor.size(0)
        num_occlu = painting_mask.size(1)
        for item in range(bs):
            for o in range(num_occlu):
                for c in range(3):                    
                    painted_input_tensor[item][c][painting_mask[item][o].bool()] = RGB_painting[item][o][c]
        # current setting: the pixel colors are overlayed by the last occlusion mask.
        return painted_input_tensor

    @staticmethod
    def get_reward(model, input_tensor, target_tensor, mask, area,
                   target_type='non-target', sigma=200, RGB_painting=None):
        '''
        input:
        model: utils.agent.model
        input_tensor: torch.floattensor with size (bs, 3, 224, 224)
        target_tensor: torch.longtensor with size (bs, 1)
        mask: torch.floattensr with size (bs, 224, 224) or 
              (bs, num_occlu, 224, 224) if RGB_painting != None
        area: torch.floattensor with size (bs, 1)
        target_type: 'non-target': non-target attack; 
                     'random-target': random target, target is not fixed during the attacking
                     ('random-fix-target', attacking_target): random target, target is fixed during the attacking
                     'least_target': use the least confident one, target fixed during the attacking process
        attacking_target: same as target_tensor, torch.longtensor with size (bs, 1), attacker targets
        sigma: controls penalization for the area, the smaller, the more powerful
        wrong_mask: flag to return wrong filter which is used to calculated accuracy.
        RGB_painting: if not None, it should be a floattensor with size (bs, num_occlu, 3) 
                      which is used to color the masked regions.

        return:
        reward: torch.floattensor with size (bs, 1)
        acc: list of accs, label_acc and target_acc [default None]
        avg_area: the average area, scalar [0, 1.]
        '''
        assert input_tensor.size(0) == mask.size(0),\
        'the first dim of the input tensor and the mask should be the same'

        with torch.no_grad():
            
            if type(model) == list:
                model[0].cuda(torch_cuda)
                model[0].eval()
            else:
                model.cuda(torch_cuda)
                model.eval()
                
            input_tensor, target_tensor, mask, area \
            = input_tensor.cuda(torch_cuda), target_tensor.cuda(torch_cuda),\
            mask.cuda(torch_cuda), area.cuda(torch_cuda)

            if type(RGB_painting) == torch.Tensor:
                # painting
                masked_input_tensor = MPA_agent.paint(input_tensor, mask, RGB_painting)
            else:
                masked_input_tensor = input_tensor * mask.unsqueeze(1)        
            
            if type(model) == list:
                output_tensor, _ = model[0](masked_input_tensor, bool_magic=True)
            else:
                output_tensor = model(masked_input_tensor)
            
            output_tensor = F.softmax(output_tensor, dim=1)
            pred = output_tensor.argmax(dim=1)
            
            label_filter = pred==target_tensor.view(-1)
            target_filter = None
            
            label_acc = label_filter.float().mean()
            target_acc = None

            if target_type == 'non-target':
                p_cl = 1. - torch.gather(input=output_tensor, dim=1, index=target_tensor)
            
            elif target_type[0] == 'random-fix-target':
                attacking_target = target_type[-1]
                p_cl = torch.gather(input=output_tensor, dim=1, index=attacking_target.cuda(torch_cuda))
                target_filter = pred==attacking_target.view(-1)
                target_acc = target_filter.float().mean()
                
            reward = torch.log(p_cl+utils.eps) + (- area / (sigma**2))
            avg_area = area.mean() / (mask.size(-2) * mask.size(-1))
            
            acc = [label_acc, target_acc]
            filters = [label_filter, target_filter]
            return reward, acc, filters, avg_area

    @staticmethod
    def reward_backward(rewards, gamma):
        '''
        input:
        reward: torch.floattensor with size (bs, something)
        gamma: discount factor
        
        return:
        updated_reward: torch.floattensor with the same size as input
        '''
        gamma = 1
        R = 0
        updated_rewards = torch.zeros(rewards.size()).cuda(torch_cuda)
        for i in range(rewards.size(-1)):
            R = rewards[:, -(i+1)] + gamma * R
            updated_rewards[:, -(i+1)] = R
        return updated_rewards
    
    def reinforcement_learn(self, steps=150, distributed_area=True, 
                            baseline_subtraction=False, 
                            target_type='non-target'):
        '''
        input:
        steps: the steps to interact with the environment for the agent
        distributed_area: whether to activate distributed area penalizations in the intermediate process.
        baseline_subtraction: flag to use baseline subtraction technique.
        target_type: 'non_target': non-target attack; 
                     'random_target': random target, target is not fixed during the attacking
                     ('random_fix_target', attacking_target): random target, target is fixed during the attacking
                     'least_target': use the least confident one, target fixed during the attacking process
                     attacking_target: same as self.target_tensor with size (), attacker targets
        
        return:
        mask: torch.floattensor with size (224, 224)
        mask [optional, if self.color == True]: torch.floattensor with size (num_occlu, 224, 224)
        RGB_painting: None
        RGB_painting [optional, if self.color == True]: torch.floattensor with size (num_occlu, 3)
        floating_combo: torch.LongTensor with size ([1, len(space)])
        area: torch.floattensor with size (1)
        '''
        H = self.image_tensor.size(-2)
        W = self.image_tensor.size(-1)
        queries = 0
        
        image_batch = self.image_tensor.expand(self.rl_batch, 
                                               self.image_tensor.size(-3), 
                                               self.image_tensor.size(-2), 
                                               self.image_tensor.size(-1)).contiguous()
        target_batch = self.target_tensor.expand(self.rl_batch, 1).contiguous()
        t_atk = target_type[0] == 'random-fix-target'
        
        if t_atk:
            assert type(target_type) == tuple,\
            'wrong format of target type for random_fix_target attack, should be tuple'
            target_type = (target_type[0], target_type[1].expand(self.rl_batch, 1).contiguous())
        
        self.mind.cuda(torch_cuda)
        self.mind.train()
        self.optimizer.zero_grad()
        
        # set up non-target attack records
        RGB_painting = None
        floating_mask = None
        floating_RGB_painting = None
        floating_area = torch.Tensor([H*W*2])
        floating_combo = None
        
        # set up target attack records
        t_RGB_painting = None
        t_floating_mask = None
        t_floating_RGB_painting = None
        t_floating_area = torch.Tensor([H*W*2])
        t_floating_combo = None
        
        # set up record for early stop
        orig_r_record = []
        
        # start learning, interacting with the environments
        if self.critic:
            for s in range(steps):
                
                # make combo and get reward
                combo, log_p_combo, rewards_critic = self.select_combo()
                rewards = torch.zeros(combo.size()).cuda(torch_cuda)
                
                # update RGB painting if color is optimized
                if self.color:
                    RGB_painting = self.create_painting(combo)
                
                if distributed_area:
                    mask, area = self.create_mask(combo, 
                                                  distributed_area=distributed_area,
                                                  distributed_mask=self.color,
                                                  H=H, W=W)
                    for o in range(self.num_occlu-1):
                        r = (- area[:, o].view(-1, 1) / (self.sigma**2))
                        rewards[:, o*4+3] = r.squeeze(-1)
                    
                    # calculate ending reward
                    area = area[:, -1].view(-1, 1)
                    r, acc, filters, avg_area = self.get_reward(self.model, image_batch, target_batch,
                                                                mask, area, target_type=target_type,
                                                                sigma=self.sigma, RGB_painting=RGB_painting)
                    queries += image_batch.size(0)
                    orig_r_record.append(r.mean())
                    rewards[:, -1] = r.squeeze(-1)
                else:
                    mask, area = self.create_mask(combo, 
                                                  distributed_mask=self.color,
                                                  H=H, W=W)
                    r, acc, filters, avg_area = self.get_reward(self.model, image_batch, target_batch, 
                                                                mask, area, target_type=target_type,
                                                                sigma=self.sigma, RGB_painting=RGB_painting)
                    queries += image_batch.size(0)
                    orig_r_record.append(r.mean())
                    rewards[:, -1] = r.squeeze(-1)
                rewards = self.reward_backward(rewards, self.gamma)
                
                # update non-target records
                wrong_filter = ~filters[0]
                if acc[0] != 1:
                    if distributed_area:
                        area_candidate = area[:, -1][wrong_filter]
                    else:
                        area_candidate = area[wrong_filter]
                    temp_floating_area, temp = area_candidate.min(dim=0)
                    if temp_floating_area < floating_area:
                        floating_mask = mask[wrong_filter][temp].squeeze(0)
                        floating_combo = combo[wrong_filter][temp]
                        floating_area = temp_floating_area
                        if self.color:
                            floating_RGB_painting = RGB_painting[wrong_filter][temp]
                
                # update target records
                if t_atk:
                    if acc[1] != 0:
                        target_filter = filters[1]
                        if distributed_area:
                            area_candidate = area[:, -1][target_filter]
                        else:
                            area_candidate = area[target_filter]
                        temp_floating_area, temp = area_candidate.min(dim=0)
                        if temp_floating_area < t_floating_area:
                            t_floating_mask = mask[target_filter][temp].squeeze(0)
                            t_floating_combo = combo[target_filter][temp]
                            t_floating_area = temp_floating_area
                            if self.color:
                                t_floating_RGB_painting = RGB_painting[target_filter][temp]
                
                # baseline_substraction
                if baseline_subtraction:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + utils.eps)
                
                # calculate loss
                advantages = rewards - rewards_critic
                loss1 = (-log_p_combo * advantages.detach()).sum(dim=1).mean()
                loss2 = advantages.pow(2).sum(dim=1).mean()
                loss = loss1 + loss2
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # reset mind to continuously interact with the environment
                self.mind.reset()            
                
                # early stop
                if s >= 2:
                    if abs(orig_r_record[-1] + orig_r_record[-3] - 2*orig_r_record[-2]) < PA_cfg.es_bnd:
                        break
                
        else:
            for s in range(steps):
                
                # make combo and get reward
                combo, log_p_combo = self.select_combo()
                rewards = torch.zeros(combo.size()).cuda(torch_cuda)
                
                # update RGB painting if color is optimized
                if self.color:
                    RGB_painting = self.create_painting(combo)
                
                if distributed_area:
                    mask, area = self.create_mask(combo, 
                                                  distributed_area=distributed_area,
                                                  distributed_mask=self.color,
                                                  H=H, W=W)
                    for o in range(self.num_occlu-1):
                        r = (- area[:, o].view(-1, 1) / (self.sigma**2))
                        rewards[:, o*4+3] = r.squeeze(-1)
                    area = area[:, -1].view(-1, 1)

                    r, acc, filters, avg_area = self.get_reward(self.model, image_batch, target_batch,
                                                                mask, area, target_type=target_type,
                                                                sigma=self.sigma, RGB_painting=RGB_painting)
                    queries += image_batch.size(0)
                    orig_r_record.append(r.mean())
                    rewards[:, -1] = r.squeeze(-1)
                else:
                    mask, area = self.create_mask(combo, 
                                                  distributed_mask=self.color,
                                                  H=H, W=W)
                    r, acc, filters, avg_area = self.get_reward(self.model, image_batch, target_batch, 
                                                                mask, area, target_type=target_type,
                                                                sigma=self.sigma, RGB_painting=RGB_painting)
                    queries += image_batch.size(0)
                    orig_r_record.append(r.mean())
                    rewards[:, -1] = r.squeeze(-1)
                rewards = self.reward_backward(rewards, self.gamma)
                
                # update non-target records
                wrong_filter = ~filters[0]
                if acc[0] != 1:
                    if distributed_area:
                        area_candidate = area[:, -1][wrong_filter]
                    else:
                        area_candidate = area[wrong_filter]
                    temp_floating_area, temp = area_candidate.min(dim=0)
                    if temp_floating_area < floating_area:
                        floating_mask = mask[wrong_filter][temp].squeeze(0)
                        floating_combo = combo[wrong_filter][temp]
                        floating_area = temp_floating_area
                        if self.color:
                            floating_RGB_painting = RGB_painting[wrong_filter][temp]
                
                # update target records
                if t_atk:
                    if acc[1] != 0:
                        target_filter = filters[1]
                        if distributed_area:
                            area_candidate = area[:, -1][target_filter]
                        else:
                            area_candidate = area[target_filter]
                        temp_floating_area, temp = area_candidate.min(dim=0)
                        if temp_floating_area < t_floating_area:
                            t_floating_mask = mask[target_filter][temp].squeeze(0)
                            t_floating_combo = combo[target_filter][temp]
                            t_floating_area = temp_floating_area
                            if self.color:
                                t_floating_RGB_painting = RGB_painting[target_filter][temp]
                
                # baseline subtraction
                if baseline_subtraction:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + utils.eps)
                
                # calculate loss 
                loss = (-log_p_combo * rewards).sum(dim=1).mean()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # reset mind to continuously interact with the environment
                self.mind.reset()            
                
                # early stop
                if s >= 2:
                    if abs(orig_r_record[-1] + orig_r_record[-3] - 2*orig_r_record[-2]) < PA_cfg.es_bnd:
                        break
                
        non_target_success = floating_mask != None
        target_success = t_floating_mask != None
        success = [non_target_success, target_success]
        if t_atk:
            if target_success:
                return t_floating_mask, t_floating_RGB_painting if self.color else None,\
                       t_floating_combo, t_floating_area, success, queries
            else:
                return floating_mask, floating_RGB_painting if self.color else None,\
                       floating_combo, floating_area, success, queries
        else:
            return floating_mask, floating_RGB_painting if self.color else None,\
                   floating_combo, floating_area, success, queries

    @staticmethod
    def attack(model, input_tensor, target_tensor, sigma, target_type='non_target', lr=0.1, distributed_area=False,
               critic=False, baseline_subtraction=True, color=False, num_occlu=3, rl_batch=75, steps=50):
        '''
        input:
        model: pytorch model
        input_tensor: torch.floattensor with size (3, 224, 224)
        target_tensor: torch.longtensor with size ()
        sigma: scalar, contrain the area of the occlusion
        target_type:     'non_target': non-target attack; 
                         'random_target': random target, target is not fixed during the attacking
                         ('random_fix_target', attacking_target): random target, target is fixed during the attacking
                         'least_target': use the least confident one, target fixed during the attacking process
                     attacking_target: same as target_tensor with size (), attacker targets
        lr: learning rate for p_pi, scalar
        distributed_area: distributed_area flag
        critic: whether to switch on the critic part v_theta
        baseline_subtraction: flag to use reward normalization
        color: flag to search the RGB channel values

        return:
        mask: torch.floattensor with size (224, 224)
        mask [optional, if color == True]: torch.floattensor with size (num_occlu, 224, 224)
        RGB_painting: None
        RGB_painting [optional, if self.color == True]: torch.floattensor with size (num_occlu, 3)
        combo: torch.LongTensor([1, len(space)])
        area: scalar with size (1)
        '''
        # time to start
        attack_begin = time.time()

        actor = MPA_agent(model, input_tensor, target_tensor, num_occlu, color, sigma)
        actor.build_robot(critic, rl_batch=rl_batch, gamma=1, lr=lr, stable=True)
        mask, RGB_painting, combo, area, success, queries = actor.reinforcement_learn(
            steps=steps, distributed_area=distributed_area, 
            baseline_subtraction=baseline_subtraction, 
            target_type=target_type
        )
        return mask, RGB_painting, combo, area/(input_tensor.size(-2)*input_tensor.size(-1)),\
               success, queries, time.time() - attack_begin
    

class TPA_agent(robot):
    
    def __init__(self, model, image_tensor, label_tensor, noises, noises_label, num_occlu, area_occlu):
        '''
        the __init__ function needs to create action space because this relates with 
        the __init__ of the policy network 
        '''
        # BUILD ENVIRONMENT
        self.model = model
        self.image_tensor = image_tensor
        self.label_tensor = label_tensor
        self.noises = noises
        self.noises_label = noises_label
        # build action space
        self.num_occlu = num_occlu
        self.area_occlu = area_occlu
        self.action_space = self.create_action_space()
        #print('environment and action space have been determined')
        #print('remember to build robot')
        
        # query counter
        self.queries = 0
        
    def create_action_space(self):
        H, W = self.image_tensor.size()[-2:]
        self.H, self.W = H, W
        action_space = []
        for o in range(self.num_occlu):
            size_occlu = torch.Tensor([H*W*self.area_occlu]).sqrt_().floor_().long().item()
            self.size_occlu = size_occlu
            action_space += [int(H)-size_occlu, int(W)-size_occlu, len(self.noises), 
                                self.noises[-1].size(-2)-size_occlu, self.noises[-1].size(-1)-size_occlu]
        return action_space
    
    def build_robot(self, critic, rl_batch, gamma, lr, stable=True):
        super().__init__(critic, self.action_space, rl_batch, gamma, lr, stable)
        #print('robot built!')
        
    def receive_reward(self, p_images, area_occlu, label_batch, target_batch, attack_type):
        '''
        input:
        p_images: torch.floattensor with size (bs, 3, self.H, self.W)
        area_occlu: torch.floattensor with size (bs, 1), torch.Tensor([0])
        label_batch: torch.longtensor with size (bs, 1)
        target_batch: torch.longtensor with size (bs, 1)
        attack_type: 'target' or 'non_target'
        
        return:
        reward: torch.floattensor with size (bs, 1)
        acc: list of accs [label_acc, target_acc]
        avg_area: the average area, scalar [0, 1.]
        filters: list of filters [label_filter, target_filter]
        '''
        with torch.no_grad():
            self.model.cuda(torch_cuda)
            self.model.eval()
            p_images, label_batch, target_batch, area_occlu =\
            p_images.cuda(torch_cuda), label_batch.cuda(torch_cuda),\
            target_batch.cuda(torch_cuda), area_occlu.cuda(torch_cuda)
            
            output_tensor = self.model(p_images)
            output_tensor = F.softmax(output_tensor, dim=1)
            
            label_acc = utils.accuracy(output_tensor, label_batch)
            label_filter = output_tensor.argmax(dim=1) == label_batch.view(-1)
            
            target_acc = None
            target_filter = None
            
            if attack_type == 'target':
                target_acc = utils.accuracy(output_tensor, target_batch)
                target_filter = output_tensor.argmax(dim=1) == target_batch.view(-1)
                p_cl = torch.gather(input=output_tensor, dim=1, index=target_batch)
            elif attack_type == 'non-target':
                p_cl = 1. - torch.gather(input=output_tensor, dim=1, index=label_batch)
                

            reward = torch.log(p_cl+utils.eps)
            avg_area = area_occlu.mean() / (self.H * self.W)

            acc = [label_acc, target_acc]
            filters = [label_filter, target_filter]
            return reward, acc, avg_area, filters
    
    def reward_backward(self, rewards):
        '''
        input:
        reward: torch.floattensor with size (bs, something)
        
        return:
        updated_reward: torch.floattensor with the same size as input
        '''
        R = 0
        updated_rewards = torch.zeros(rewards.size()).cuda(torch_cuda)
        for i in range(rewards.size(-1)):
            R = rewards[:, -(i+1)] + self.gamma * R
            updated_rewards[:, -(i+1)] = R
        return updated_rewards
    
    def learn(self, attack_type='target', steps=50, distributed_area=False):

        # create image batch
        image_batch = self.image_tensor.expand(self.rl_batch, 
                                               self.image_tensor.size(-3), 
                                               self.H, 
                                               self.W).contiguous()
        label_batch = self.label_tensor.expand(self.rl_batch, 1).contiguous()
        target_batch = self.noises_label.expand(self.rl_batch, 1).contiguous()
        
        # set up training env
        self.mind.cuda(torch_cuda)
        self.mind.train()
        self.optimizer.zero_grad()
        
        # set up non-target attack records
        floating_combo = None
        floating_r = torch.Tensor([-1000]).cuda(torch_cuda)
        
        # set up target attack records
        t_floating_combo = None
        t_floating_r = torch.Tensor([-1000]).cuda(torch_cuda)
        
        # set up orig_reward record to early stop
        orig_r_record = []
        
        # start interacting with the env
        if self.critic:
            pass
        else:
            for s in range(steps):
                
                # add queries
                self.queries += self.rl_batch
                
                # select combo
                combo, log_p_combo = self.select_combo()
                # receive rewards
                rewards = torch.zeros(combo.size()).cuda(torch_cuda)
                
                if distributed_area:
                    pass
                else:
                    p_images = self.combo_to_image(
                        combo, self.num_occlu, None,
                        image_batch, self.noises, size_occlu=self.size_occlu
                    )

                    # receive reward
                    r, acc, avg_area, filters = self.receive_reward(
                        p_images, 
                        torch.Tensor([0]), 
                        label_batch, target_batch, 
                        attack_type, 
                    )
                    rewards[:, -1] = r.squeeze(-1)
                    
                    # orig_r_record
                    orig_r_record.append(r.mean())
                    
                    # backprop rewards
                    rewards = self.reward_backward(rewards)
                    
                    # update floating variables
                    best_v, best_i = r.max(dim=0)
                    if floating_r < best_v:
                        floating_r = best_v
                        floating_combo = combo[best_i]
                        
                    # baseline subtraction
                    rewards = (rewards - rewards.mean()) / (rewards.std() + utils.eps)
                    
                    # calculate loss
                    loss = (-log_p_combo * rewards).sum(dim=1).mean()
                    
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # reset mind
                    self.mind.reset()
                    
                    # early stop
                    if s >= 2:
                        if attack_type == 'target':
                            if acc[1][0] != 0:
                                break
                        elif attack_type == 'non-target':
                            if acc[0][0] != 100:
                                break
                        if abs(orig_r_record[-1] + orig_r_record[-3] - 2*orig_r_record[-2]) < PA_cfg.es_bnd:
                            break
                    
            return floating_combo, floating_r
            
    @staticmethod
    def combo_to_image(combo, num_occlu, mask, image_batch, noises,
                       size_occlu=None, output_p_masks=False):
        '''
        input:
        combo: torch.floattensor with size (bs, combo_size)
        num_occlu: int
        mask: torch.floattensor with size (bs, num_occlu, H, W), return from the 
              create_mask function with distributed_mask being True.
              the mask consists of 1s with the occluded area being 0s.
        image_batch: torch.floattensor with size (bs, 3, 224, 224)
        noises: list of tensor.floattensor with size (3, 224, 224)
        
        return:
        temp_images: torch.floattensor with size (bs, 3, 224, 224)
        '''
        # process size_occlu
        if type(size_occlu) == int:
            size_occlu = [size_occlu]*num_occlu
        
        bs, combo_size = combo.size()
        p_l = combo_size // num_occlu

        temp_images = image_batch.clone()

        # post process combo
        p_combo_image = []
        p_combo_choose = []
        p_combo_noise = []
        for o in range(num_occlu):
            p_combo_image.append(combo[:, o*p_l+0: o*p_l+2])
            p_combo_choose.append(torch.index_select(combo, dim=1, 
                                                        index=torch.LongTensor([o*p_l+2]).cuda(torch_cuda)))
            p_combo_noise.append(combo[:, o*p_l+3: o*p_l+5])
        p_combo_image = torch.cat(p_combo_image, dim=1)
        p_combo_choose = torch.cat(p_combo_choose, dim=1)
        p_combo_noise = torch.cat(p_combo_noise, dim=1)

        
        if noises is None:
            # create masks
            H, W = image_batch.size()[-2:]
            p_masks = torch.zeros(bs, H, W)

            for item in range(bs):
                for o in range(num_occlu):
                    p_masks[item][
                                    p_combo_image[item, o*2+0]: 
                                    p_combo_image[item, o*2+0]+size_occlu[o],
                                    p_combo_image[item, o*2+1]:
                                    p_combo_image[item, o*2+1]+size_occlu[o]]\
                    = 1.
            return p_masks
            
        else:
            # get mask and create p_image
            for item in range(bs):
                for o in range(num_occlu):
                    temp_images[item][:, 
                                        p_combo_image[item, o*2+0]: 
                                        p_combo_image[item, o*2+0]+size_occlu[o],
                                        p_combo_image[item, o*2+1]:
                                        p_combo_image[item, o*2+1]+size_occlu[o]]\
                    = noises[p_combo_choose[item, o]][:,
                                        p_combo_noise[item, o*2+0]: 
                                        p_combo_noise[item, o*2+0]+size_occlu[o],
                                        p_combo_noise[item, o*2+1]:
                                        p_combo_noise[item, o*2+1]+size_occlu[o]]

            # output p_images
            if output_p_masks:

                # create masks
                H, W = image_batch.size()[-2:]
                p_masks = torch.zeros(bs, H, W)

                for item in range(bs):
                    for o in range(num_occlu):
                        p_masks[item][
                                        p_combo_image[item, o*2+0]: 
                                        p_combo_image[item, o*2+0]+size_occlu[o],
                                        p_combo_image[item, o*2+1]:
                                        p_combo_image[item, o*2+1]+size_occlu[o]]\
                        = 1.

                return temp_images, p_masks
            else:
                return temp_images

            
    @staticmethod
    def from_combos_to_images(x, p_combos, model, area_occlu, noises_used):
        '''
        input:
        x: torch.floattensor with size (bs, 3, 224, 224)
        p_combos: combos returned by DC attack, list of length bs
        model: pytorch model
        area_occlu: float, like 0.04
        noises_used: list of torch.floattensor with size (3, 224, 224)
        
        return:
        areas: list of length bs, occlued areas
        preds: list of length bs, predicted labels
        p_images: list of torch.flosttensor with size (3, 224, 224)
        '''
        model = model.cuda(torch_cuda).eval()
        H, W = x.size()[-2:]
        p_images = []
        areas = []
        preds = []
        for index in range(len(p_combos)):
            item = p_combos[index]
            temp_combos = torch.cat(item, dim=1)

            p_image, p_masks = TPA_agent.combo_to_image(
                combo=temp_combos, 
                num_occlu=len(item), 
                mask=None, 
                image_batch=x[index].unsqueeze(0),
                noises=noises_used, 
                size_occlu=torch.Tensor([H*W*area_occlu]).sqrt_().floor_().long().item(),
                output_p_masks=True)
            areas.append(p_masks.sum() / (H*W) )

            with torch.no_grad():
                output = F.softmax(model(p_image), dim=1)
                preds.append(output.argmax())

            p_images.append(p_image.squeeze(0))
            print('index of x: {}'.format(index))
        preds = torch.stack(preds, dim=0)
        return areas, preds, p_images
    
    @staticmethod
    def DC(model, p_image, label, noises_used, noises_label, 
           area_sched, n_boxes, attack_type, num_occlu=1, lr=0.03, rl_batch=500, 
           steps=80, n_pre_agents=0, to_load=None, load_index=None):
        '''
        input:
        p_image: torch.floattensor with size (3, 224, 224)
        return:
        '''
        # time to start
        attack_begin = time.time()
        
        # divide and conquer, initialization
        H, W = p_image.size()[-2:]
        p_mask = torch.zeros(H, W).bool()
        
        # set up records
        optimal_combos = [[] for _ in range(n_boxes)]
        non_target_success_rcd = [[] for _ in range(n_boxes)]
        target_success_rcd = [[] for _ in range(n_boxes)]
        queries_rcd = [[] for _ in range(n_boxes)]
        time_used_rcd = [[] for _ in range(n_boxes)]
        
        # load
        if n_pre_agents!=0:
            
            n_pre_combos = len(to_load.combos[load_index])
            
            # load records
            for a_i in range(n_pre_agents-1, n_boxes):
                optimal_combos[a_i] = copy.deepcopy(to_load.combos[load_index])
                non_target_success_rcd[a_i].append(copy.deepcopy(to_load.non_target_success[load_index]))
                target_success_rcd[a_i].append(copy.deepcopy(to_load.target_success[load_index]))
                queries_rcd[a_i].append(copy.deepcopy(to_load.queries[load_index]))
                time_used_rcd[a_i].append(copy.deepcopy(to_load.time_used[load_index]))
            
            # print info
            print('*** loaded pre_maximum_agents: {}'.format(n_pre_agents))
            
            # check success
            stop = to_load.target_success[load_index].item() if attack_type == 'target'\
            else to_load.non_target_success[load_index].item()
            
            # apply combos
            pre_size_occlu = [torch.Tensor([H*W*item]).sqrt_().floor_().long().item()
                              for item in area_sched[:n_pre_agents]]
            p_image = TPA_agent.combo_to_image(
                combo=torch.cat(to_load.combos[load_index], dim=1), 
                num_occlu=n_pre_combos,
                mask=None, 
                image_batch=p_image.unsqueeze(0),  
                noises=noises_used,
                size_occlu=pre_size_occlu, 
                output_p_masks=False
            ).squeeze(0)
            
            if stop:
                # process records
                for a_i in range(n_pre_agents, n_boxes):
                    non_target_success_rcd[a_i] = non_target_success_rcd[a_i][-1]
                    target_success_rcd[a_i] = target_success_rcd[a_i][-1]
                    queries_rcd[a_i] = queries_rcd[a_i][-1]
                    time_used_rcd[a_i] = time_used_rcd[a_i][-1]
                    
                # summury
                print('*** combos taken: {} | non-target success: {} | target success: {} | queries: {} | '
                      .format((n_pre_combos), to_load.non_target_success[load_index].item(), 
                              to_load.target_success[load_index].item(), 
                              to_load.queries[load_index], ))
                    
                return p_image, optimal_combos, non_target_success_rcd, target_success_rcd, time_used_rcd, queries_rcd
            
            else:
                assert n_pre_combos == n_pre_agents, 'check your loaded records'

                
            
        # time to restart
        attack_begin = time.time()
        
        # attacking loop
        queries = to_load.queries[load_index] if n_pre_agents!=0 else 0
        for box in range(n_pre_agents, n_boxes):
            actor = TPA_agent(
                model=model,
                image_tensor=p_image, 
                label_tensor=label, 
                noises=noises_used, 
                noises_label=noises_label,
                num_occlu=num_occlu, 
                area_occlu=area_sched[box]
            )
            actor.build_robot(critic=False, rl_batch=rl_batch, gamma=1, lr=lr, stable=True)
            selected_combo, r = actor.learn(attack_type=attack_type, steps=steps)
            
            # get queries
            queries += actor.queries # query counter
            
            # update p_image, p_mask
            p_image, temp_p_mask = actor.combo_to_image(
                combo=selected_combo, 
                num_occlu=num_occlu, 
                mask=None, 
                image_batch=p_image.unsqueeze(0),
                noises=noises_used, 
                size_occlu=actor.size_occlu,
                output_p_masks=True,
            )
            p_image = p_image.squeeze(0)
            p_mask = p_mask | temp_p_mask.unsqueeze(0).bool()
            
            # check pred
            with torch.no_grad():
                output = F.softmax(model(p_image.unsqueeze(0)), dim=1)
                score, pred = output.max(dim=1)
            
            # get success
            non_target_success = pred != label
            target_success = pred == noises_label

            # show info
            print('combos taken: {} | '
                  'pred: {} | pred_confidence: {:.4f} | '
                  'GT confidence: {:.4f} | target_confidence: {:.4f} | '
                  .format(box+1, 
                          pred.item(), 
                          score.item(), 
                          output[0, label].item(), 
                          output[0, noises_label].item(),
                         )
                 )
            
            # update records
            for temp_box in range(box, n_boxes):
                optimal_combos[temp_box].append(selected_combo)
                non_target_success_rcd[temp_box].append(non_target_success)
                target_success_rcd[temp_box].append(target_success)
                queries_rcd[temp_box].append(queries)
                time_used = time.time() - attack_begin
                time_used_rcd[temp_box].append(time_used)
                
            # check success
            stop = target_success if attack_type == 'target' else non_target_success
            if stop:
                break
        
        # process records
        if n_pre_agents == n_boxes:
            a_i = n_pre_agents - 1
            non_target_success_rcd[a_i] = non_target_success_rcd[a_i][-1]
            target_success_rcd[a_i] = target_success_rcd[a_i][-1]
            queries_rcd[a_i] = queries_rcd[a_i][-1]
            time_used_rcd[a_i] = time_used_rcd[a_i][-1]
        else:
            for a_i in range(n_pre_agents, n_boxes):
                non_target_success_rcd[a_i] = non_target_success_rcd[a_i][-1]
                target_success_rcd[a_i] = target_success_rcd[a_i][-1]
                queries_rcd[a_i] = queries_rcd[a_i][-1]
                time_used_rcd[a_i] = time_used_rcd[a_i][-1]
        
        # summury
        print('*** combos taken: {} | non-target success: {} | target success: {} | queries: {} | '
              .format((box+1) if n_pre_agents!=n_boxes else n_pre_agents, non_target_success_rcd[-1].item(),
              target_success_rcd[-1].item(), queries_rcd[-1], ))
        
        return p_image, optimal_combos, non_target_success_rcd, target_success_rcd, time_used_rcd, queries_rcd
    

class HPA_agent():
    '''
    sampling-based agent, instead of RL agent
    '''
    
    class p():
        def __init__(self, model, sigma=2000., target=False):
            self.model = model
            self.model.cuda()
            self.model.eval()
            self.sigma = sigma
            self.target = target
            self.queries = 0

        @staticmethod
        def transfer_theta(theta):
            '''
            input
            theta: bs \times 4 \times b numpy array.
                   bs: batch size
                   4: center x, y, height, width
                   b: the number of the masks
            return
            theta: bs \times 4 \times b numpy array.
                   bs: batch size
                   4: the left upper and right lower point coordinates.
                   b: the number of the masks
            '''
            proposal = np.zeros(theta.shape)
            proposal[:, 0, :] = theta[:, 0, :] - 0.5 * theta[:, 2, :]
            proposal[:, 1, :] = theta[:, 1, :] - 0.5 * theta[:, 3, :]
            proposal[:, 2, :] = theta[:, 0, :] + 0.5 * theta[:, 2, :]
            proposal[:, 3, :] = theta[:, 1, :] + 0.5 * theta[:, 3, :]
            temp = np.clip(np.rint(proposal), 0, 223).astype(int)

            return temp

        @staticmethod
        def occlude_image(x, theta):
            '''
            input:
            x:     input tensor image, pixel range [0, 1]
            theta: bs \times 4 \times b numpy array.
                   bs: batch size
                   4: the left upper and right lower point coordinates.
                   b: the number of the masks

            return:
            occluded x: cuda tensor
            occluded area: numpy array with size (bs, 1)
            '''
            assert theta.shape[0] == x.size(0), 'the batch size of x and theta should be the same'
            assert theta.shape[1] == 4, 'the second dimension of theta should be 4 b'
            b = theta.shape[-1]

            mask = np.ones((x.size(0), x.size(2), x.size(3)))
            area = np.zeros((x.size(0)))
            for i in range(x.size(0)):
                for j in range(b):
                    mask[i, theta[i, :, j][0]:theta[i, :, j][2], theta[i, :, j][1]:theta[i, :, j][3]] = 0.
                area[i] = mask[i][mask[i]==0.].shape[0]

            return x * torch.from_numpy(mask).cuda().unsqueeze(1).float(), area          

        def prob(self, x, y, theta):

            '''
            input:
            x:     input tensor image, pixel range [0, 1]
            theta:  bs \times 4 \times b numpy array.
                   bs: batch size
                   4: center x, y, height, width
                   b: the number of the masks
            y: cuda torch tensor with size (bs)

            return: 
            torch array with size (bs, 1)
            '''
            with torch.no_grad():
                theta = self.transfer_theta(theta)

                x, area = self.occlude_image(x, theta)
                output = self.model(x)
                output = F.softmax(output, dim=1)
                self.queries += x.size(0)
                if self.target:
                    p_cl = torch.gather(input=output, dim=1, index=y.unsqueeze(-1))
                else:
                    p_cl = 1. - torch.gather(input=output, dim=1, index=y.unsqueeze(-1))

                # write the distribution on the nusiance space, normalized
                p_theta = np.exp(- area / (self.sigma**2)) # the std is sqrt(1000*2000) 1414.21
                return p_cl * torch.from_numpy(p_theta).cuda().unsqueeze(-1).float()
            
    class q():
        def __init__(self, sigma=5.):
            self.sigma=sigma

        def sample(self, mean):
            '''
            input:
            mean:  bs \times 4 \times b numpy array.
                   bs: batch size
                   4: center x, y, height, width
                   b: the number of the masks

            return:
            numpy array with the same size as mean
            '''

            proposal = np.random.normal(loc=mean, scale=self.sigma)

            # clipping proposal
            proposal[:, :2, :] = np.clip(a=proposal[:,:2,:], a_min=0., a_max=223.)
            proposal_maxheight =  2 * np.maximum(proposal[:, 0, :], 223. - proposal[:, 0, :])
            proposal_maxwidth =  2 * np.maximum(proposal[:, 1, :], 223. - proposal[:, 1, :])
            proposal[:, 2, :] = np.clip(a=proposal[:, 2, :], a_min=0., a_max=proposal_maxheight)
            proposal[:, 3, :] = np.clip(a=proposal[:, 3, :], a_min=0., a_max=proposal_maxwidth)

            return proposal


        def prob(self, theta, mean):
            '''
            calculate the the unnormalized conditional probability
            input:
            theta:  bs \times 4 \times b numpy array.
                   bs: batch size
                   4: center x, y, height, width
                   b: the number of the masks
            mean: has the same type as theta

            return:
            torch array with size (bs, 1)
            '''

            res = theta - mean
            probs = np.exp((res ** 2) / (-2 * (self.sigma ** 2)))
            probs = torch.from_numpy(probs).cuda().float()

            return torch.stack([torch.prod(probs[item]) for item in range(probs.size(0))], 0).unsqueeze(-1)
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input_tensors, label_tensors, occlu_num, sigma, steps, t_atk=False, target_tensors=None):
        '''
        input:
        input_tensors: bs, 3, H, W
        label_tensors: bs (ground truth labels)
        target_tensors: bs (target tensors or label tensors)
        
        NOTE: batch mode is available
        '''
        H,W = input_tensors.size()[-2:]

        start_time = time.time()
        if not t_atk:
            p_sampler = self.p(self.model, sigma=sigma, target=False)
            q_sampler = self.q()
            testing_samples, p_samples, queries = self.Metropolis_Hasting(
                p=p_sampler, q=q_sampler, steps=steps,
                s_sample=self.get_s_sample(b=occlu_num, size=input_tensors.size(0), H=H, W=W),
                x=input_tensors, y=label_tensors,
            )
            attacking_masks = self.find_the_attacker(p_samples, testing_samples, 
                                                     input_tensors, label_tensors)
            acc, filters, avg_occlu_area = self.attacking_rate(
                model=self.model,
                attacking_masks=attacking_masks,
                input_tensor=input_tensors,
                label_tensor=label_tensors,
                target_tensor=target_tensors,
            )
            return attacking_masks, acc, filters, avg_occlu_area, p_sampler.queries, \
                   time.time()-start_time
        else:
            p_sampler = self.p(self.model, sigma=sigma, target=True)
            q_sampler = self.q()
            testing_samples, p_samples, queries = self.Metropolis_Hasting(
                p=p_sampler, q=q_sampler, steps=steps,
                s_sample=self.get_s_sample(b=occlu_num, size=input_tensors.size(0), H=H, W=W),
                x=input_tensors, y=target_tensors,
            )
            attacking_masks = self.find_the_attacker(p_samples, testing_samples, 
                                                     input_tensors, target_tensors)
            acc, filters, avg_occlu_area = self.attacking_rate(
                model=self.model,
                attacking_masks=attacking_masks,
                input_tensor=input_tensors,
                label_tensor=label_tensors,
                target_tensor=target_tensors,
            )
            return attacking_masks, acc, filters, avg_occlu_area, p_sampler.queries, \
                   time.time()-start_time
    
    @staticmethod
    def get_s_sample(b, size, H=224, W=224):
        s_sample = np.ones((size, 4, b))
        #s_sample[:, :2, :] = np.random.randint(low=0, high=224, size=(2, s_sample.shape[-1]))
        s_sample[:, 0, :] = np.random.randint(low=0, high=H, size=(1, s_sample.shape[-1]))
        s_sample[:, 1, :] = np.random.randint(low=0, high=W, size=(1, s_sample.shape[-1]))
        s_sample[:, 2, :] = np.sqrt(np.sqrt(1000*2000) / b)
        s_sample[:, 3, :] = np.sqrt(np.sqrt(1000*2000) / b)
        return s_sample
    
    @staticmethod
    def Metropolis_Hasting(p, q, steps, s_sample, x, y):
        '''
        input:
        p: the target distribution to sample from
        q: proposal distribution
        steps: iteration number
        s_sample: starting sample, numpy array with size (bs, 4, 3)
        x: input_tensor with size (bs, 3, 224, 224)
        y: target_tensor with size (bs)
        return:
        samples from the target distribution
        '''
        theta_list = []
        p_theta_list = []

        for s in tqdm(range(steps)):
            if s == 0:
                theta = q.sample(mean=s_sample)
                theta_list.append(theta)
                p_theta_list.append(p.prob(x, y, theta))
            else:
                theta_old = theta_list[-1].copy()
                p_theta_old = p_theta_list[-1].clone()
                theta = q.sample(mean=theta_old)
                p_theta = p.prob(x, y, theta)
                
                p_accept = ((p_theta * q.prob(theta_old, mean=theta)) / 
                            (p_theta_old * q.prob(theta, mean=theta_old)))
                p_accept = torch.min(torch.ones(p_accept.size()).cuda(), p_accept)

                # convert to torch tensor
                theta_old = torch.from_numpy(theta_old).cuda().float()
                theta = torch.from_numpy(theta).cuda().float()

                u = torch.rand(p_accept.size()).cuda()
                mask = (p_accept > u).view(-1)

                for i in range(mask.size()[0]):
                    if mask[i]:
                        theta_old[i] = theta[i]
                        p_theta_old[i] = p_theta[i]

                theta_list.append(theta_old.cpu().numpy())
                p_theta_list.append(p_theta_old)

        assert len(theta_list) == steps, 'not completed sampling'
        return theta_list, p_theta_list, p.queries
    
    @staticmethod
    def find_the_attacker(p_samples, testing_samples, input_tensor, target_tensor):
        '''find the argmax of the distributions found by Metropolis-Hasting
        input:
        testing_samples: list of thetas each of which is (numpy array)
                theta:  bs \times 4 \times b numpy array.
                           bs: batch size
                           4: center x, y, height, width
                           b: the number of the masks
        input_tensor: torch tensor with size (bs, 3, 224, 224)
        target_tensor: torch tensor with size (bs)

        return:
        attacking_masks: numpy array with size (bs, 4, num_occluders_per_image), 
                         which can be fed into p.occlude_image directly
        '''
        temp = torch.stack(p_samples, 0).squeeze(-1).t()
        candidates_probs, candidates = torch.max(temp, dim=1)
        
        attacking_thetas = [testing_samples[candidates[ins]][ins] 
                            for ins in range(input_tensor.size(0))]
        attacking_thetas = np.stack(attacking_thetas, axis=0)
        attacking_masks = HPA_agent.p.transfer_theta(attacking_thetas)
        return attacking_masks
    
    @staticmethod
    def attacking_rate(model, attacking_masks, input_tensor, 
                       label_tensor=None, target_tensor=None):
        with torch.no_grad():
            x_occluded, occluded_area = HPA_agent.p.occlude_image(x=input_tensor, theta=attacking_masks)
            avg_occlu_area = (occluded_area/(input_tensor.size(-1)*input_tensor.size(-2))).mean()
            
            # check result
            output = model(x_occluded)
            pred = output.argmax(dim=1)
            label_acc = utils.accuracy(output, label_tensor) if label_tensor != None else None
            target_acc = utils.accuracy(output, target_tensor) if target_tensor != None else None
            
            label_filter = pred==label_tensor.view(-1) if label_tensor != None else None
            target_filter = pred==target_tensor.view(-1) if target_tensor != None else None
            
        return [label_acc, target_acc], [label_filter, target_filter], avg_occlu_area

    