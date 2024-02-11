from modalities import * 
from torch import nn
from torchvision import transforms
import torch
import time
from operations import ConvNextBlock, InvBottleNeckBlock, conv3x3, conv1x1

from PIL import Image

operations_call = {
    "ConvNext": ConvNextBlock,
    "InvBottleNeck": InvBottleNeckBlock,
}

class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x

class Network( nn.Module ):

    def __init__(self, genotype, device : str, n_classes = 2):
        super().__init__()
        self.genotype = genotype

        # CHANNEL PER STAGE 
        self.in_channel = [16,32,64,96] 

        # 
        self.blocks = nn.ModuleList()
        self.connections = []

        # STEM CONVOLUTION
        self.stem_conv = conv3x3(3, self.in_channel[0], stride=2)


        # For each stage
        for block_number, block in enumerate(self.genotype):
            cell_1,cell_2,connection_info = block
            cell_1_op = nn.Sequential()
            cell_2_op = nn.Sequential()
            connection_info = get_connection_dictionary(connection_info)
            self.connections.append( connection_info )
            if (one_operation_mode(connection_info)) :
                for i in range(4) : 
                    cell_1_op.add_module(f"{cell_1[0]}-{block_number}-{i}",operations_call[cell_1[0]](self.in_channel[block_number],cell_1[1]))
            else :
                for i in range(2):
                    cell_1_op.add_module(f"{cell_1[0]}-{block_number}-{i}",operations_call[cell_1[0]](self.in_channel[block_number],cell_1[1]))
                    cell_2_op.add_module(f"{cell_2[0]}-{block_number}-{i}",operations_call[cell_2[0]](self.in_channel[block_number],cell_2[1]))
                

            if block_number < ( len(self.genotype) - 1) : 
                downsample = conv3x3(self.in_channel[block_number], self.in_channel[block_number+1], stride = 2)
            else :
                downsample = IdentityModule()
            self.blocks.append(nn.ModuleList([ cell_1_op, cell_2_op, downsample ]) )
                
            
        # BUILD THE NET
        # self.layers = nn.Sequential(*self.layers)
        self.last_conv = conv1x1(self.in_channel[-1], 1280)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, n_classes)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # change device if possible.
        self.device = torch.device(device)
        self.to(self.device)

        self.time_records = {}

    def forward(self, X):
        start_time = time.time()

        # Calcolo del tempo per lo stem_conv
        X = self.stem_conv(X)
        self.conv_output = X
        stem_conv_time = time.time() - start_time

        start_time = time.time()

        # Calcolo del tempo per block_forward
        X = self.block_forward(X)
        blocks_forward_time = time.time() - start_time

        start_time = time.time()

        # Calcolo del tempo per last_conv
        X = self.last_conv(X)
        last_conv_time = time.time() - start_time

        start_time = time.time()

        # Calcolo del tempo per avg_pool e view
        X = self.avg_pool(X).view(-1, 1280)
        avg_pool_view_time = time.time() - start_time

        start_time = time.time()

        # Calcolo del tempo per classifier
        X = self.classifier(X)
        classifier_time = time.time() - start_time

        # Conserva i tempi
        self.time_records = {
            'stem_conv': stem_conv_time,
            'blocks_forward': blocks_forward_time,
            'last_conv': last_conv_time,
            'avg_pool_view': avg_pool_view_time,
            'classification': classifier_time
        }

        return X

    def info_time(self):
        for step, time_taken in self.time_records.items():
            print(f"{step} = {time_taken:.3f} s")


    def block_forward(self, X):
        max_time = 0.0
        max_op = None

        if self.device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        else:
            start_time = time.time()

        for id, (block, connection) in enumerate (zip(self.blocks, self.connections)) :
            cell_1, cell_2, downsample = block

            if self.device == 'cuda':
                start.record()

            if one_operation_mode(connection):
                X = downsample(cell_1(X))

            if two_branch_mode(connection):
                X = downsample(cell_1(X) + cell_2(X))

            if sequential_mode(connection):
                if connection["skip"]:
                    X = downsample(cell_2(cell_1(X)) + X)
                else:
                    X = downsample(cell_2(cell_1(X)))

            if complete_mode(connection):
                if connection["skip"]:
                    out_1 = cell_1(X)
                    X = downsample(cell_2(out_1 + X) + out_1)
                else:
                    X = downsample(cell_2(cell_1(X) + X))

            if self.device == 'cuda':
                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
            else:
                end_time = time.time()
                elapsed_time = (end_time - start_time)

            if elapsed_time > max_time:
                max_time = elapsed_time
                max_op = block
        # Conserva l'operazione con il tempo massimo e il tempo impiegato
        self.bottleneck = max_op
        self.time_bottleneck = max_time
        self.id = id

        return X
    

    def info_bottleneck(self):
        return self.id,self.bottleneck,self.time_bottleneck

                
    def probabilities(self, image : Image.Image ):
        self.eval()
        return torch.sigmoid(self(self.preprocess(image)).data)
        

    def predict(self, image : Image.Image ) :
        self.eval()
        prediction = torch.argmax(self.probabilities(image))
        if ( prediction == 0) : 
            return "NOT PERSON"
        else : 
            return "PERSON"
        
    def preprocess(self, image : Image.Image):
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype( dtype = torch.float ),
            transforms.Resize(128,antialias=True),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] )
        ])
        X = transform(image).unsqueeze(0).to(self.device)
        return X