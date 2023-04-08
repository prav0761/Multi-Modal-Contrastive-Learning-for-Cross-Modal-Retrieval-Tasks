
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

class ResNetSimCLR(nn.Module):
    def __init__(self, model='resnet50', projection_dim=128 ,layers_to_train=['layer4'],encoder_last_layer=None):
        """
        Initializes ResNetSimCLR model.

        Parameters:
        - model: str, the ResNet model to use (default: 'resnet18')
        - projection_dim: int, the dimension of the projection head output (default: 128)
        - layers_to_train: list of str, the names of the layers in the ResNet model to train (default: ['layer4'])
        """
        super(ResNetSimCLR, self).__init__()

        # Instantiate the backbone ResNet model
        if model == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            in_features = 512
        elif model == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            in_features = 2048
        elif model == 'resnet101':
            backbone = models.resnet101(pretrained=True)
            in_features = 2048
        else:
            raise ValueError('Unsupported ResNet model:', model)

        # Freeze the layers that are not specified in layers_to_train
        for name, child in backbone.named_children():
            if name not in layers_to_train:
                for param in child.parameters():
                    param.requires_grad = False

        # Remove last fully-connected layer from the backbone
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.encoder_last_layer=encoder_last_layer
        # Define the transform to be applied to input images
        self.transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        if encoder_last_layer:
            self.fc_layer = nn.Linear(in_features, encoder_last_layer)
            projection_head_input=encoder_last_layer
        else:
            projection_head_input=in_features
        # Add the projection head layers
        self.projection_head = nn.Sequential(
            nn.Linear(projection_head_input, projection_head_input),
            nn.ReLU(),
            nn.Linear(projection_head_input, projection_dim)
        )

    def forward(self, x,device):
        """
        Performs a forward pass through the ResNetSimCLR model.

        Parameters:
        - x: tensor of shape (batch_size, 3, height, width), the input images

        Returns:
        - features: tensor of shape (batch_size, in_features), the features extracted from the backbone
        - projection: tensor of shape (batch_size, projection_dim), the projections of the features
        """
        # Apply the transform to the input images
        x = self.transform(x)
        x=x.to(device)

        # Extract features from the backbone
        #x = x.unsqueeze(0) # Add batch dimension
        x= self.backbone(x)
        # Flatten the features and pass them through the projection head
        features = x.view(x.size(0), -1)
        if self.encoder_last_layer:
            features=self.fc_layer(features)
        projection = self.projection_head(features)

        # Return the features and projections
        return features, projection
    
    
    
class OpenAI_SIMCLR(nn.Module):
    def __init__(self, model='openai-gpt', projection_dim=128,layers_to_train=['h.11'],encoder_last_layer=None):
        """
        A PyTorch module for text encoding using a pre-trained OpenAI GPT model.

        Args:
            model (str): The name or path of the pre-trained OpenAI GPT model to use. Defaults to 'openai-gpt'.
            projection_dim (int): The dimension of the projection head output. Defaults to 128.
            layers_to_train (list of str): The names of the layers to train. Defaults to ['h.11'].
        """
        super(OpenAI_SIMCLR, self).__init__()

        # Load backbone and tokenizer
        self.backbone = OpenAIGPTModel.from_pretrained(model)
        self.config = self.backbone.config
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(model)
        self.encoder_last_layer=encoder_last_layer
        # Set requires_grad for each parameter based on layers_to_train
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in layers_to_train):
                param.requires_grad = True
            else:
                param.requires_grad = False
        if encoder_last_layer:
            self.fc_layer=nn.Linear(self.config.n_embd,encoder_last_layer)
            projection_head_input=encoder_last_layer
        else:
            projection_head_input=self.config.n_embd
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(projection_head_input ,projection_head_input),
            nn.ReLU(),
            nn.Linear(projection_head_input, projection_dim)
        )
        
    def forward(self, texts,device):
        """
        Forward pass of the TextEncoder module.

        Args:
            text (str): The input text to encode.

        Returns:
            A tuple containing the encoded text features and their projections.
        """
        
        # Tokenize input text
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        input_ids = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]
        tokens_tensor = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=0)
        tokens_tensor = tokens_tensor.to(device)
        # Get text features from backbone
        all_hidden_states = self.backbone(tokens_tensor)
        features = torch.mean(all_hidden_states, dim=1)  # Shape: (1, 768)

        if self.encoder_last_layer:
            features=self.fc_layer(features)
        # Pass text features through projection head
        projections = self.projection_head(features)

        return features,projections
