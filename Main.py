import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import time
import os
from Arguments import get_args
from Model import DeepNet
from load_imglist import ImageList

global iterations
global epoch

iterations = 0


def Adjust_lr_scheduler(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*0.5



def train(train_loader, model, criterion, optimizer, scheduler, epoch, step):
    losses = AverageMeter()    
    
    global iterations

    model.train()

    end = time.time()
    for i, (ip, gt) in enumerate(train_loader):
        ip = ip.to(device)
        gt = gt.to(device)
        # compute output
        op = model(ip)
        loss = criterion(op, gt)
        # record loss
        losses.update(loss.item(), ip.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(i)
        if i % 50 == 0:
            #print('Iterations \t:',iterations)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'train_Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch+1, i, len(train_loader), loss = losses))
    iterations += 1
    if iterations % 2000 == 0:
        scheduler(optimizer)

    return losses.avg


def validate(val_loader, model, criterion):
    global epoch
    losses = AverageMeter()
    for i, (ip, gt) in enumerate(val_loader):
        ip = ip.to(device)
        gt = gt.to(device)
        
        with torch.no_grad():
            # compute output
            op = model(ip)
            loss = criterion(op, gt)
            # record loss
            losses.update(loss.item(), ip.size(0))

    print('\nValidation set: Average loss: {}\n'.format(losses.avg))
    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.summ = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.summ += val * n
        self.count += n
        self.avg = self.summ / self.count

def save_checkpoint(state, filename):
    torch.save(state, filename)

global args
global device

args = get_args()

device = torch.device("cuda" if args.cuda else "cpu")

# Instantiate model 
model = DeepNet(dataset = args.dataset).to(device)

# Instantiate optimizer and learning rate scheduler
optimizer = torch.optim.Adam(params = model.parameters(),
                            lr = args.lr,
                            weight_decay = args.weight_decay,
                             )


scheduler = Adjust_lr_scheduler#(optimizer)
# Instantiate loss function
criterion = nn.MSELoss()

# optionally resume from a checkpoint

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

#Instantiate image loaders

train_loader = torch.utils.data.DataLoader(
                ImageList(root = args.root_path, fileList = args.train_list, type_of_data = "train",transform = None),
                batch_size = args.batch_size, 
                shuffle = False,
                num_workers = 0,
                pin_memory = True)

val_loader = torch.utils.data.DataLoader(
                ImageList(root = args.root_path, fileList = args.val_list,type_of_data = "val",transform = None),
                batch_size = args.batch_size, 
                shuffle = False,
                num_workers = 0,
                pin_memory = True)

print("start training...")
# train and validate    
for epoch in range(args.start_epoch, args.epochs):
#    scheduler.step()
    print("Epoch = ",epoch+1)
    print("learning rate = ",args.lr)
    # train for one epoch
    train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, args.step)
    # evaluate on validation set
    val_loss = validate(val_loader, model, criterion)

    save_name = args.save_path + 'DeepConvNet_' +str(epoch+1) + '.pth.tar'
    save_checkpoint({
        'epoch' : epoch + 1,
        'model' : args.model,
        'datset' : args.dataset,
        'state_dict': model.state_dict(),
        'train_loss' : train_loss,
        'val_loss' : val_loss},
         save_name)


