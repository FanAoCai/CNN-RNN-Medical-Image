import os
import sys
import openslide
from PIL import Image
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import init

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 attention aggregator training script')
parser.add_argument('--train_lib', type=str, default='7cls_data/train_data_7cls', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='7cls_data/val_data_7cls', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider (default: 10)')
parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model', default='checkpoint_best.pth', type=str, help='path to trained model checkpoint')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--shuffle', action='store_true', help='to shuffle order of sequence')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    
    #load libraries
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_dset = attentiondata(args.train_lib, args.s, args.shuffle, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_dset = attentiondata(args.val_lib, args.s, False, trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #make model
    embedder = ResNetEncoder(args.model)
    for param in embedder.parameters():
        param.requires_grad = False
    embedder = embedder.cuda()
    embedder.eval()

    attention = attention_single(512,4)
    attention = attention.cuda()
    
    #optimization
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.SGD(attention.parameters(), 0.001, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    cudnn.benchmark = True

    fconv = open(os.path.join(args.output, 'convergence.csv'), 'w')
    fconv.write('epoch,train.loss,train.fpr,train.fnr,val.loss,val.fpr,val.fnr\n')
    fconv.close()

    #
    for epoch in range(args.nepochs):
        train_loss, train_fpr, train_fnr = train_single(epoch, embedder, attention, train_loader, criterion, optimizer)
        val_loss, val_fpr, val_fnr = test_single(epoch, embedder, attention, val_loader, criterion)

        fconv = open(os.path.join(args.output,'convergence.csv'), 'a')
        fconv.write('{},{},{},{},{},{},{}\n'.format(epoch+1, train_loss, train_fpr, train_fnr, val_loss, val_fpr, val_fnr))
        fconv.close()

        val_err = (val_fpr + val_fnr)/2
        if 1-val_err >= best_acc:
            best_acc = 1-val_err
            obj = {
                'epoch': epoch+1,
                'state_dict': attention.state_dict()
            }
            torch.save(obj, os.path.join(args.output,'attention_checkpoint_best.pth'))

def train_single(epoch, embedder, attention, loader, criterion, optimizer):
    attention.train()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    for i,(inputs,target) in enumerate(loader):
        print('Training - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1, args.nepochs, i+1, len(loader)))

        attention.zero_grad()
        middle = torch.zeros(len(inputs), embedder(inputs[0].cuda())[1].shape[0], embedder(inputs[0].cuda())[1].shape[1])
        middle = middle.cuda()
        for s in range(len(inputs)):
            input = inputs[s].cuda()
            # print('input1: ', input.shape)
            _, input = embedder(input)
            middle[s] = input
        output = attention(middle)

        target = target.cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*target.size(0)
        fps, fns = errors(output.detach(), target.cpu())
        running_fps += fps
        running_fns += fns

    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Training - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def test_single(epoch, embedder, attention, loader, criterion):
    attention.eval()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Validating - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1,args.nepochs,i+1,len(loader)))
            
            attention.zero_grad()
            middle = torch.zeros(len(inputs), embedder(inputs[0].cuda())[1].shape[0], embedder(inputs[0].cuda())[1].shape[1])
            middle = middle.cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                # print('input1: ', input.shape)
                _, input = embedder(input)
                middle[s] = input
            output = attention(middle)
            
            target = target.cuda()
            loss = criterion(output,target)
            
            running_loss += loss.item()*target.size(0)
            fps, fns = errors(output.detach(), target.cpu())
            running_fps += fps
            running_fns += fns
            
    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Validating - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def errors(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    neq = pred!=real
    fps = float(np.logical_and(pred==1,neq).sum())
    fns = float(np.logical_and(pred==0,neq).sum())
    return fps,fns

class ResNetEncoder(nn.Module):

    def __init__(self, path):
        super(ResNetEncoder, self).__init__()

        temp = models.resnet34()
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(path)
        temp.load_state_dict(ch['state_dict'])
        self.features = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x), x
    
class attention_single(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.deal = nn.Linear(7*512, 2)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.flatten(1)
        x = self.deal(x)
        print(x.shape)
        return x

# class attention_single(nn.Module):

#     def __init__(self, in_channels,c_m,c_n,reconstruct = True):
#         super().__init__()
#         self.in_channels=in_channels
#         self.c_m=c_m
#         self.c_n=c_n
#         self.reconstruct = reconstruct
#         self.convA=nn.Conv1d(in_channels,c_m,1)
#         self.convB=nn.Conv1d(in_channels,c_n,1)
#         self.convV=nn.Conv1d(in_channels,c_n,1)
#         if self.reconstruct:
#             self.conv_reconstruct = nn.Conv1d(c_m, in_channels, kernel_size = 1)

#     def forward(self, x):
#         b, c, h=x.shape
#         assert c==self.in_channels
#         A=self.convA(x) #b,c_m,h,w
#         print('A.shape', A.shape)
#         B=self.convB(x) #b,c_n,h,w
#         print('B.shape', B.shape)
#         V=self.convV(x) #b,c_n,h,w
#         print('V.shape', V.shape)
#         tmpA=A.view(b,self.c_m,-1)
#         attention_maps=F.softmax(B.view(b,self.c_n,-1))
#         print('attention_maps', attention_maps.shape)
#         attention_vectors=F.softmax(V.view(b,self.c_n,-1))
#         print('attention_vectors', attention_vectors.shape)
#         # step 1: feature gating
#         global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
#         print('global_descriptors', global_descriptors.shape)
#         # step 2: feature distribution
#         tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
#         print('tmpZ', tmpZ.shape)
#         tmpZ=tmpZ.view(b,self.c_m,h) #b,c_m,h,w
#         print('final', tmpZ.shape)
#         if self.reconstruct:
#             tmpZ=self.conv_reconstruct(tmpZ)
#         print('con', tmpZ.shape)

#         return tmpZ

class attentiondata(data.Dataset):

    def __init__(self, path, s, shuffle=False, transform=None):

        lib = torch.load(path)
        self.s = s
        self.transform = transform
        self.slidenames = lib['slides']
        self.targets = lib['targets']
        self.grid = lib['grid']
        self.level = lib['level']
        self.mult = lib['mult']
        self.size = int(224*lib['mult'])
        self.shuffle = shuffle

        slides = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
        print('')
        self.slides = slides

    def __getitem__(self,index):

        slide = self.slides[index]
        grid = self.grid[index]
        if self.shuffle:
            grid = random.sample(grid,len(grid))

        s = min(self.s, len(grid))
        out = torch.zeros(s, 3, self.size, self.size)
        for i in range(s):
            img = slide.read_region(grid[i], self.level, (self.size, self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            out[i] = img
        
        return out, self.targets[index]

    def __len__(self):
        
        return len(self.targets)

if __name__ == '__main__':
    main()
