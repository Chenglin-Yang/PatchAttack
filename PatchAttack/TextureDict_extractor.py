import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as Models
import torch.optim as optim

import PatchAttack.utils as utils

# global variables
torch_cuda = 0


class GradCam():
    
    class model_wrapper(nn.Module):
        def __init__(self, features, classifier):
            super().__init__()
            self.features = features
            self.classifier = classifier
            
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class ModelOutputs():
        """ Class for making a forward pass, and getting:
        1. The network output.
        2. Activations from intermeddiate targetted layers.
        3. Gradients from intermeddiate targetted layers. """
        def __init__(self, model, target_layers):
            self.model = model
            self.feature_extractor = self.FeatureExtractor(self.model.features, target_layers)

        def get_gradients(self):
            return self.feature_extractor.gradients

        def __call__(self, x):
            target_activations, output  = self.feature_extractor(x)
            # for non-square input
            #if output.size(-1) != 7 or output.size(-2) != 7:, but this only suits VGG
            if output.size(-1) != output.size(-2): # if I append avgpool in features for VGG, this will never be met
                output = self.model.avgpool(output)
            output = output.view(output.size(0), -1)
            output = self.model.classifier(output)
            return target_activations, output
        
        class FeatureExtractor():
            """ Class for extracting activations and 
            registering gradients from targetted intermediate layers """
            def __init__(self, model, target_layers):
                self.model = model
                self.target_layers = target_layers
                self.gradients = []

            def save_gradient(self, grad):
                self.gradients.append(grad)

            def __call__(self, x):
                del self.gradients
                self.gradients = []
                
                outputs = []
                self.gradients = []
                for name, module in self.model._modules.items():
                    x = module(x)
                    if name in self.target_layers:
                        x.register_hook(self.save_gradient)
                        outputs += [x]
                        x.retain_grad()
                return outputs, x
    
    def __init__(self, model, target_layer_names, use_cuda=True):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda(torch_cuda)

        self.extractor = self.ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        cam_H, cam_W = input.size(-2), input.size(-1)
        
        if self.cuda:
            features, output = self.extractor(input.cuda(torch_cuda))
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_()
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda(torch_cuda) * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        ori_cam = cam
        cam = cv2.resize(cam, (cam_W, cam_H)) # bilinear interpolation
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        self.current_cam = cam
        self.current_input = input
        
        return cam, ori_cam

    def show_current_cam(self, show=True, save_dir=None, dpi=300, tight=True):
        heatmap = cv2.applyColorMap(np.uint8(255*self.current_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)/255
        cv2_img = utils.data_agent.inv_normalize(self.current_input.squeeze(0)).permute((1, 2, 0)).cpu().numpy()
        
        show_img = heatmap + cv2_img
        show_img -= show_img.min()
        show_img /= show_img.max()
        
        plt.figure()

        if show:
            plt.imshow(show_img)

        if save_dir is not None:
            if tight:
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(fname=save_dir,
                dpi=dpi, facecolor='w', edgecolor='w', format='png')
        
        return show_img


class vgg19_extractor(nn.Module):

    # default values, and are overwritten in practical use
    style_choice = [1, 6, 11, 20, 29]
    content_choice = [22]
    attention_threshold = 0.7
    normal_spatial_size = [50176, 12544, 3136, 784, 196]
    
    def __init__(self):
        super().__init__()
        self.model = Models.vgg19(pretrained=True).cuda(torch_cuda)
        self.style_layers = self.style_choice
        self.content_layers = self.content_choice
        print('style_layer choice', self.style_layers)
        self.activations = []
        self.contents = []
        
        # change max pool to average pool
        for name, child in self.model.features.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.model.features[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)
                
        # lock the gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        del self.activations
        del self.contents
        self.activations = []
        self.contents = []
        for name, child in self.model.features.named_children():
            x = child(x)
            if int(name) in self.style_layers:
                self.activations.append(x)
            # add content
            if int(name) in self.content_layers:
                self.contents.append(x)
        return x
    
    def get_style(self, x):
        '''
        the standard scale corresponds to the case where the input size is 224 by 224
        '''
        _ = self.forward(x)
        alphas = [i/(j.size(-1)*j.size(-2)) 
                  for i, j in zip(self.normal_spatial_size, self.activations)]
        return [self.gram_matrix(item)*alpha 
                for item, alpha in zip(self.activations, alphas)]
    
    def get_attention_style(self, x, cam):
        _ = self.forward(x)
        attention_style = []
        
        # offset for conv4 or lower style_choice
        offset = int(round(np.log2(self.activations[-1].size()[-1] / cam.shape[-1])))

        for i in range(len(self.activations)):
            activation = self.activations[-(i+1)]
            #mask = self.choose_region(self.bilinear_upsample(cam, 
            #                                                 H_scale=pow(2, i+offset), 
            #                                                 W_scale=pow(2, i+offset)))
            mask = self.choose_region(self.bilinear_upsample(cam, 
                                                             target_H=activation.size(-2), 
                                                             target_W=activation.size(-1)))
            mask = torch.from_numpy(mask).expand(activation.size()).contiguous()
            attention_style.append(self.gram_matrix_with_mask(activation, mask))
            # add attention_content
            if i+offset == 1: # conv4-2
                self.attention_contents = self.content_with_mask(self.contents, mask)
        
        r_attention_style = [attention_style[-(i+1)] for i in range(len(attention_style))]
        alphas = [i/(j.size(-1)*j.size(-2)) 
                  for i, j in zip(self.normal_spatial_size, self.activations)]

        return [item*alpha for item, alpha in zip(r_attention_style, alphas)]
    
    def get_mask_style(self, x, mask):
        _ = self.forward(x)
        mask_style = []
        # check mask
        if len(mask.size()) == 3:
            mask = mask.unsqueeze(0)
        assert len(mask.size()) == 4,\
        'the mask should be 3 or 4 dims'
        assert mask.size(1) == 1,\
        'channel number of mask should be 1'
        
        h, w = mask.size()[-2:]
        for i in range(len(self.activations)):
            #temp_mask = F.upsample_bilinear(mask.float(), size=(int(h/pow(2, i)), 
            #                                                    int(w/pow(2, i))))
            temp_mask = F.upsample_bilinear(mask.float(), size=(self.activations[i].size(-2), 
                                                                self.activations[i].size(-1)))
            temp_mask = temp_mask.expand(self.activations[i].size()).contiguous()
            mask_style.append(self.gram_matrix_with_mask(self.activations[i], temp_mask.bool()))
            
        return mask_style
    
    @staticmethod
    def spatial_repeat(x, scale):
        '''x: torch.floattensor with size (bs, c, h, w) or (c, h, w)'''
        temp = torch.cat([x]*scale, dim=-2)
        temp = torch.cat([temp]*scale, dim=-1)
        return temp
    
    @staticmethod
    def generate_image_from_style(x, save_dir, attention=True, iterations=10000, 
                                  label=None, cls_w=0, lr=0.01, scale=1,
                                  noise_dims=[1, 3, 224, 224], noise_init='randn',
                                  noise_optimization_mode='normal', pgd_eps=None, 
                                  custom_noise_init=None, mask=None, observer=None):
        '''
        currently, I can limit the optimization space of the pixels
        '''
        
        if os.path.exists(save_dir):
            return torch.load(os.path.join(save_dir, 'iter_{}.pt'.format(iterations-1)))
        else:
            cnn = Models.vgg19(pretrained=True).cuda(torch_cuda).eval()
            # create the folder for the generated images
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # prepare extractor to extract the styles
            style_extractor = vgg19_extractor()
            # check input: from style image or just style
            if type(x) == torch.Tensor:
                style_image = x
                # get dims
                bs, c, h, w = style_image.size()
                if attention:
                    grad_cam = GradCam(model=Models.vgg19(pretrained=True), 
                                       target_layer_names=["35"])
                    _, ori_cam = grad_cam(style_image)
                    target_style = style_extractor.get_attention_style(style_image, ori_cam)
                else:
                    target_style = style_extractor.get_style(style_image)
            else:
                bs, c, h, w = noise_dims
                target_style = x
            # generate starting point: noise
            if noise_init == 'randn':
                noise = torch.randn(bs, c, int(h/scale), int(w/scale)).cuda(torch_cuda)
            elif noise_init == 'zeros':
                noise = torch.zeros(bs, c, int(h/scale), int(w/scale)).cuda(torch_cuda)
            elif noise_init == 'custom':
                noise = custom_noise_init
            vgg19_extractor.normalize(noise)
            noise = noise.requires_grad_()
            # set upper bound and lower bound if necessary
            if noise_optimization_mode == 'pgd':
                anchor_max = noise.detach().clone() + pgd_eps
                anchor_min = noise.detach().clone() - pgd_eps
            # set optimizer
            optimizer = optim.Adam(params=[noise], lr=lr)
            # optimize
            for iteration in range(iterations):
                # zero grad
                optimizer.zero_grad()
                # repeat noise
                if scale != 1:
                    noise_image = style_extractor.spatial_repeat(noise, scale)
                else:
                    noise_image = noise
                # extract style from noise
                if type(mask) == torch.Tensor:
                    noise_style = style_extractor.get_mask_style(noise_image, mask)
                else:
                    noise_style = style_extractor.get_style(noise_image)
                # calculate style loss
                style_loss = vgg19_extractor.style_loss(noise_style, 
                                                        target_style,
                                                        style_extractor.activations)
                style_loss = 1e6 * style_loss / len(noise_style)
                # classification loss
                if cls_w != 0:
                    l_c = F.cross_entropy(input=cnn(noise_image), target=label.cuda(torch_cuda)) * cls_w
                else:
                    l_c = torch.Tensor([0]).cuda(torch_cuda)
                
                # overall loss
                loss = style_loss + l_c
                # backward and update params
                loss.backward()
                # applying mask if necessary
                if type(mask) == torch.Tensor:
                    noise.grad.data[~mask.expand(noise.size())] = 0
                # take a step
                optimizer.step()
                
                if noise_optimization_mode == 'normal':
                    # normalize the image
                    vgg19_extractor.normalize(noise)
                elif noise_optimization_mode == 'pgd':
                    # pgd clamping
                    vgg19_extractor.clamp_optimizer(noise, mode='pgd', 
                                                    anchor_min=anchor_min, 
                                                    anchor_max=anchor_max)
                # show progress
                if iteration % 3000 == 0 or iteration == iterations-1:
                    print("Iteration: {}, Style Loss: {:.3f}, classification Loss: {:.3f}"
                          .format(iteration, style_loss.item(), l_c.item()))
                    # check noise prediction
                    with torch.no_grad():
                        output = F.softmax(cnn(noise), dim=1)
                        print('vgg19 predict: {} | vgg19 confidence: {:.3f}'.format(output.argmax().item(), 
                                                                                    output.max().item()))
                        if observer != None:
                            output_obs = F.softmax(observer(noise), dim=1)
                            print('observer predict: {} | observer confidence: {:.3f}'
                                  .format(output_obs.argmax().item(), 
                                          output_obs.max().item()))
                        
                # generate the image
                if iteration % 3000 == 0 or iteration == iterations-1:
                    torch.save(noise.squeeze(0).cpu().detach(), 
                               save_dir+'/iter_{}.pt'.format(iteration))
                    # release memory
                    torch.cuda.empty_cache()
            return noise
    
    @staticmethod
    def get_kmeans_style(indices, dataset, save_dir, n_clusters=30):
        '''
        input:
        indices: return of data_agent.get_indices()
        dataset: which dataset to use to extract the styles, should be same as 
                 that used in data_agent.get_indices()
        save_dir: dir to save the .pt file
        return:
        kmeans: numpy object
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        file_name = os.path.join(save_dir, 'style_kmeans_{}.pt'
                                 .format(n_clusters))
        if os.path.exists(file_name):
            kmeans = torch.load(file_name)
        else:
            # get attention styles from all indices
            grad_cam = GradCam(model=Models.vgg19(pretrained=True), target_layer_names=["35"])
            style_extractor = vgg19_extractor()
            indices_style = []
            for index in indices:
                image, _ = dataset.__getitem__(index)
                image = image.cuda(torch_cuda).unsqueeze(0)
                _, ori_cam = grad_cam(image)
                attention_style = style_extractor.get_attention_style(image, ori_cam)
                # move attention_style to cpu in case of memory issues
                attention_style = [item.cpu() for item in attention_style]
                indices_style.append(attention_style)
            
            # clustering
            from sklearn.cluster import KMeans
            # memory issue may arise here if indices sytle is on GPU
            X = style_extractor.flatten_style(indices_style).cpu().numpy()
            print('clustering...')
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
            print('finished! Saving...')
            torch.save(kmeans, file_name)
        # check the assignment situation of each cluster
        assignment = [(kmeans.labels_ == i).sum() for i in range(n_clusters)]
        print('the assignment of the clusters are: ', assignment)
        print('Remeber to inversely flatten the style clusters,\
               \nif you need to process the clusters of the returned numpy kmeans object')
        return kmeans
    
    @staticmethod
    def style_loss(style_1, style_2, activations):
        '''
        input:
        style_1 : list of torch.floattensor with size (bs, c_i, c_i)
        style_2 : list of torch.floattensor with size (bs, c_i, c_i)
        activations: list of activation, torch.floattensor with size (bs, c_i, h_i, w_i) 
        return:
        loss
        '''
        assert len(style_1) == len(style_2),\
        'inconsistent dims of two styles'
        layer_num = len(style_1)
        
        loss = 0
        for i in range(layer_num):
            bs, c, h, w = activations[i].size()
            loss += (style_1[i]-style_2[i]).pow(2).sum() / (4*(c**2)*((h*w)**2))
        return loss

    @staticmethod
    def normalize(noise):
        if len(noise.size()) == 4:
            noise.data[:, 0, :, :].clamp_(min=-2.1179, max=2.2490)
            noise.data[:, 1, :, :].clamp_(min=-2.0357, max=2.4285)
            noise.data[:, 2, :, :].clamp_(min=-1.8044, max=2.6400)
        else:
            noise.data[0, :, :].clamp_(min=-2.1179, max=2.2490)
            noise.data[1, :, :].clamp_(min=-2.0357, max=2.4285)
            noise.data[2, :, :].clamp_(min=-1.8044, max=2.6400)
        
    @staticmethod
    def gram_matrix_with_mask(matrix, mask):
        '''
        input:
        matrix: torch.floattensor with size (bs, c, h, w)
        mask: torch.floattensor with size (bs, c, h, w)
        return:
        gram: torch.tensor tensor with size (bs, c, c)
        '''
        bs, c, h, w = matrix.size()
        mx = matrix.view(bs, c, -1)
        mk = mask.view(bs, c, -1)
        
        alpha = mk.sum(dim=2, keepdims=True).float()
        alpha = float(h*w)/alpha[0, 0]
        alpha = alpha.cuda(torch_cuda)
        
        mx = mx[mk].view(bs, c, -1)
        return torch.bmm(mx, mx.permute(0, 2, 1)) * alpha
    
    @staticmethod
    def content_with_mask(contents, mask):
        '''
        input:
        contents: list with one element, torch.floattensor with size (bs, c, h, w)
        mask: torch.floattensor with size (bs, c, h, w)
        return: attention_content, torch.floattensor with size (bs, c, h', w')
        '''
        return [contents[0][mask].view(contents[0].size(0), contents[0].size(1), -1)]
        
    @staticmethod
    def bilinear_upsample(cam, H_scale=2, W_scale=2, target_H=None, target_W=None):
        if target_H is None or target_W is None:
            h, w = cam.shape
            temp_cam = cv2.resize(cam, (w*W_scale, h*H_scale))
        else:
            temp_cam = cv2.resize(cam, (target_W, target_H))
        temp_cam = temp_cam - np.min(temp_cam)
        temp_cam = temp_cam / np.max(temp_cam)
        return temp_cam
    
    @staticmethod
    def choose_region(cam):
        threshold = vgg19_extractor.attention_threshold
        temp_cam = cam - cam.min()
        temp_cam = temp_cam / temp_cam.max()
        mask = temp_cam >= threshold
        return mask
    
    @staticmethod
    def gram_matrix(matrix):
        '''
        input:
        matrix: torch.floattensor with size (bs, c, h, w)
        return:
        gram: torch.floattensor with size (bs, c, c)
        '''
        bs, c, h, w = matrix.size()
        
        m = matrix.view(bs, c, -1)
        return torch.bmm(m, m.permute(0, 2, 1))
    
    @staticmethod
    def flatten_style(x, inv=False, dims=[64, 128, 256, 512, 512]):
        '''
        input:
        x: [inv=False]  torch.floattensor with size (bs, 64**2+...+512**2)
        x': [inv=True]  list of size bs, each element is a list of size 5 containing: 
               torch.floattensor with size (1, 64, 64),..., (1, 512, 512) 
        return:
        x to x'
        x' to x
        CURRENTLY DO NOT SUPPORT BATCH CALCULATION
        '''
        if not inv:
            indices_style = x
            return torch.stack([torch.cat([item.view(-1) for item in style], dim=0) 
                                for style in indices_style], dim=0)
        else:
            periods = [0]
            counter = 0
            for dim in dims:
                counter += dim*dim
                periods.append(counter)

            index1 = periods[:-1]
            index2 = periods[1:]

            if x.size(0) == 1:
                activations = []
                for dim,i1,i2 in zip(dims, index1, index2):
                    activations.append(x[0, i1:i2].view(1, dim, dim))
                return activations
            else:
                multi_activations = []
                for item in x:
                    activations = []
                    for dim,i1,i2 in zip(dims, index1, index2):
                        activations.append(item[i1:i2].view(1, dim, dim))
                    multi_activations.append(activations)
                return multi_activations
            
    @staticmethod
    def clamp_optimizer(noise, mode, anchor_min=None, anchor_max=None):
        if mode == 'iter_gs':
            pass
        elif mode == 'pgd':
            temp_index = noise < anchor_min
            noise.data[temp_index] = anchor_min.data[temp_index]
            temp_index = noise > anchor_max
            noise.data[temp_index] = anchor_max.data[temp_index]
            vgg19_extractor.normalize(noise)
            
