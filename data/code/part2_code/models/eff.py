from efficientnet_pytorch import EfficientNet
import warnings 
warnings.filterwarnings('ignore')

def efficientNet_classic(name='efficientnet-b1'):
    alist = list(map(lambda x:tuple([x]),alist))
    return EfficientNet.from_pretrained(model_name=name, num_classes=248)
