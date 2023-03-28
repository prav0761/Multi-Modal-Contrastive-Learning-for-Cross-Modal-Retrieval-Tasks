
# coding: utf-8

# In[ ]:


# SOURCE(only contrastive loss) -https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
import torch
import torch.nn as nn
import torch.nn.functional as F
from lars import LARS
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.07):
        """
        Constructor for ContrastiveLoss class.
        :param batch_size: the number of pairs of embeddings in each batch
        :param temperature: temperature parameter for the loss function
        """
        super().__init__()
        self.batch_size = batch_size
        
        # Register temperature and negatives_mask as buffers so that they can be saved and loaded along with the model
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        Compute contrastive loss given two batches of embeddings.
        :param emb_i: the first batch of embeddings
        :param emb_j: the second batch of embeddings, where corresponding indices are pairs
        :return: the contrastive loss
        """
        # Normalize the embeddings to unit length
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # Concatenate the normalized embeddings into a single tensor
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute the pairwise similarity matrix between the representations
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Get the positive pairs from the similarity matrix (diagonal elements at positions k,k+batch_size and k+batch_size,k)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Compute the nominator and denominator for the contrastive loss
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        # Compute the partial loss for each pair of embeddings
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        # Compute the average loss for the batch
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss
    
class Optimizer_simclr:
    def __init__(self, optimizer_name, model_parameters, lr, momentum=None, weight_decay=None, eta=None):
        """
        Initializes the optimizer class with optimizer name, model parameters, learning rate, momentum, weight decay,
        and eta (for LARS optimizer)
        """
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eta = eta
        self.optimizer = self.get_optimizer(model_parameters) # initializes the optimizer
        self.scheduler = self.get_scheduler() # initializes the scheduler
        
    def get_optimizer(self, model_parameters):
        """
        Returns the optimizer object based on the optimizer_name specified
        """
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(model_parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            return torch.optim.Adam(model_parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'lars':
            return LARS(model_parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, eeta=self.eta)
    
    def get_scheduler(self):
        """
        Returns the cosine annealing learning rate scheduler
        """
        return CosineAnnealingLR(self.optimizer, T_max=10)


