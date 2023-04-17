""" 
This file is a sample code of model.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
model_configs : all the arguments necessary for your model design.

EX) model_configs = {"num_blocks" : 6, "activation_func" : 'relu', "norm_layer" : 'batch_norm'} 
"""
#model_configs = {} # fill in your model configs




#model
def conv_block_1(in_dim, out_dim, activation,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model



def conv_block_3(in_dim, out_dim, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model

####################################################################################

class SEModule(nn.Module):
    
    def __init__(self, in_dim, r=4):       # SE block에 들어가는 채널수는 out_dim과 동일
        super(SEModule, self).__init__()
        self.Avgpool = nn.AdaptiveAvgPool2d(1)      # 1만 입력해야 Global으로 동작

        self.excitation = nn.Sequential(
          nn.Linear(in_dim, in_dim//r, bias=False),
          nn.ReLU(),
          nn.Linear(in_dim//r, in_dim, bias=False),
          nn.Sigmoid()
        )


    
    def forward(self, x):      
        batch, channel, _, _, = x.size()
        se = self.Avgpool(x).view(batch, channel)
        se = self.excitation(se).view(batch, channel, 1, 1)
        return x * se.expand_as(x)

####################################################################################


class BottleNeck(nn.Module):
    def __init__(self,in_dim,mid_dim,out_dim,down=False):
        super(BottleNeck,self).__init__()
        self.down=down
        self.relu=nn.ReLU()
        self.se = SEModule(out_dim)

        if self.down:
            self.layer = nn.Sequential(
              conv_block_1(in_dim,mid_dim,nn.ReLU(),stride=2),
              conv_block_3(mid_dim,mid_dim,nn.ReLU(),stride=1),
              conv_block_1(mid_dim,out_dim,nn.Identity(),stride=1),
            )
            
            self.downsample = nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=2)
            
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim,mid_dim,nn.ReLU(),stride=1),
                conv_block_3(mid_dim,mid_dim,nn.ReLU(),stride=1),
                conv_block_1(mid_dim,out_dim,nn.Identity(),stride=1),
            )
            
        self.dim_equalizer = nn.Conv2d(in_dim,out_dim,kernel_size=1)
                  
    def forward(self,x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = self.se(out)
            out = out + downsample
            out = self.relu(out)    # 그냥 nn.ReLU치면 안되고 self.relu로 만들어서 해야 함
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            
            out = self.se(out)
            out = out + x
            out = self.relu(out)
        return out

####################################################################################


class ResNet(nn.Module):
    def __init__(self, base_dim = 16, num_classes=20):
        super(ResNet, self).__init__()
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3,base_dim,7,2,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim*2, base_dim*4, down=False),
        )   
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim*4,base_dim*8,base_dim*16, down=True),
        )

        self.avgpool = nn.AvgPool2d(1,1) 
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Linear(16*16*28*28,num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
                
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc_layer(out)
        
        return F.log_softmax(out)



#####################################################################################
#####################################################################################



""" You can change the model name and implement your model. """
"""
class Classifier(nn.Module):
    def __init__(self, num_classes=20, **kwargs):
        super().__init__()

    def forward(self, x):
        
        return logit
"""     


""" [IMPORTANT]
get_classifier function will be imported in evaluation file.
You should pass all the model configuration arguments in the get_classifier function 
so that we can successfully load your exact model
saved in the submitted model checktpoint file.
"""
def get_classifier(num_classes=20):
    return ResNet(num_classes=num_classes)