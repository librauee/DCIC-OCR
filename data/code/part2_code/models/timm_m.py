import timm 
from torch import nn
import torch 


class SwinForClass_1k(nn.Module):
    def __init__(self,model_name,pretrained,checkpoint_path,num_classes,dropout,inp_channels=3,
                 ):
        super().__init__()
        self.my_model = timm.create_model(model_name,pretrained=True,
                            in_chans=inp_channels)
        
#         if pretrained and checkpoint_path is not None:
#             print('pretrained model is loaded')
#             self.my_model.load_state_dict(torch.load(checkpoint_path))

        # n_features = self.model.head.in_features
        # self.model.head = nn.Linear(n_features, num_classes)
        # self.model.head = nn.Sequential(
        #     # nn.Dropout(config.dropout),
        #     nn.Linear(n_features , 64),
        #     nn.ReLU(),
        #     # nn.Linear(512,64),
        #     # nn.ReLU(),
        #     nn.Linear(64, config.num_class)
        # )

    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.LazyLinear(1024),
            nn.LeakyReLU(),
            nn.LazyLinear(num_classes)
        )

    

    def forward(self, image):
        x = self.my_model(image)
        # print(output.shape)
        # print(self.n_features)
        x = self.dropout(x)
        # x = torch.cat([x,], dim=1)
        output = self.fc(x)
        return output