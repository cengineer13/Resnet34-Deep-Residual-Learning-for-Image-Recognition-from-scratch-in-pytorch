from torch.nn import *


#consist of 2 conv layers  Residual block
class ResidualBlock(Module): 

    def __init__(self, in_chs, out_chs, stride=1, downsample=None) -> None:
        super().__init__()
        
        #block consist of two sequential conv block
        self.conv1 = Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(num_features=out_chs)
        self.conv2 = Conv2d(in_channels=out_chs, out_channels=out_chs, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(num_features=out_chs)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample        
            
    def forward(self, input): 
        identity = input #kirishdan kopiya olib qolamiz keyinchalik 2 - conv blockdan qoshish uchun
        output = self.relu(self.bn1(self.conv1(input))) #1-conv layer
        output = self.bn2(self.conv2(output)) #2-conv layer

        #identity razmerini downsample qilamiz agar output stride=2 bilan kichiklashtirilgan bo'lsa
        #Ya'ni output sizega tenglashtirib keyin qo'sha olamiz. har bir katta block 1-convda stride=2 bolgani uchn razmer outputkichiklasahdi
        if self.downsample is not None: 
            identity = self.downsample(input)
 
        output += identity #2ta convdan chiqqan natijaga undan oldingi identity ni qo'shamiz. 
        output = self.relu(output)
        return output
    

class ResNet34(Module):
    def __init__(self, img_channel, res_block, layer_chs, layers_num, num_classes=1000) -> None:
        super().__init__()
        """
        ResNet-34 model from scratch in pytorch as described in 'Deep Residual Learning for Image Recognition".

        :param img_channel: input image channel 
        :param res_block: Basic Block or residual block which consist of consecutively two conv layers with joining skipped input 
        :param layers_chs: Channel size of each layer (big block )
        :param layers_num: how many residual blocks have in each layer (big block)
        :num_classes: number of classes to be predicted
        """

        self.in_chs = 64

        #1-conv layer 7x7 kernelli blok 
        self.conv1 = Conv2d(img_channel, self.in_chs, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn = BatchNorm2d(self.in_chs)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block=res_block, out_chs=layer_chs[0], num_resblocks=layers_num[0], stride=1)
        self.layer2 = self.make_layer(block=res_block, out_chs=layer_chs[1], num_resblocks=layers_num[1], stride=2)
        self.layer3 = self.make_layer(block=res_block, out_chs=layer_chs[2], num_resblocks=layers_num[2], stride=2)
        self.layer4 = self.make_layer(block=res_block, out_chs=layer_chs[3], num_resblocks=layers_num[3], stride=2)

        self.avg_pool = AdaptiveAvgPool2d((1,1))
        self.fc = Linear(layer_chs[3], num_classes)

    def make_layer(self, block, out_chs, num_resblocks, stride): 
        #downsample - birinchi res blockdan tashqari qolganlarida birinchi convda stride=2 bolgan va bu digani
        #channel kotarilayotganda boshqa 64->128 otayotgan size kichkinalashadi
        downsample = None 
        if stride != 1 or self.in_chs != out_chs:
            downsample = Sequential(
                Conv2d(self.in_chs, out_chs, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_chs))
        
        layers = [] 
        layers.append(block(self.in_chs, out_chs, stride, downsample)) 
        
        self.in_chs = out_chs
        #big blocklarni loop qilamiz. 1 chi layerni downsample orqali yozganiz uchun -1 kamroq loop qilamiz 
        for _ in range(num_resblocks-1): 
            layers.append(block(self.in_chs, out_chs, stride=1, downsample=None)) 
        
        return Sequential(*layers) #list ni torch tensor ga otkizamiz shtobi pytorch tanishi uchun 


    def forward(self, input): 
        output = self.relu(self.bn(self.conv1(input))) #224x224 -> 112 x 112 
        output = self.maxpool(output) #112 x 112 -> 56 x 56 (3,2 maxpool bn)
        output = self.layer1(output) # 56x56
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avg_pool(output)
        output = torch.flatten(output,1) #1-dim dan boshlab tekislaydi. yani 1 dimdan uyogini tekislaydi
        output = self.fc(output)


        return output

if __name__ == "__main__":   
    resnet34 = ResNet34(img_channel=3, res_block=ResidualBlock, layer_chs = [64, 128, 256, 512], layers_num = [3, 4, 6, 3], num_classes=1000)
    input = torch.rand((10,3,224,224))
    out = resnet34(input) 
