
# coding: utf-8

# In[ ]:


# SOURCE -https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/(except optimizer_simclr and cosine functions)
import torch
import torch.nn as nn
import torch.nn.functional as F
from lars import LARS
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

class intra_ContrastiveLoss(nn.Module):
    
    
    def __init__(self, device,temperature=0.07):
        """
        Constructor for ContrastiveLoss class.
        :param batch_size: the number of pairs of embeddings in each batch
        :param temperature: temperature parameter for the loss function
        """
        super().__init__()
        self.device=device
        # Register temperature and negatives_mask as buffers so that they can be saved and loaded along with the model
        self.register_buffer("temperature", torch.tensor(temperature))
            
    def forward(self, emb_i, emb_j , batchsize):
        """
        Compute contrastive loss given two batches of embeddings.
        :param emb_i: the first batch of embeddings
        :param emb_j: the second batch of embeddings, where corresponding indices are pairs
        :return: the contrastive loss
        """
        
        
        negatives_mask =  (~torch.eye(batchsize * 2, batchsize * 2, dtype=bool)).float().to(self.device)

        # Normalize the embeddings to unit length
        z_i = F.normalize(emb_i, dim=1).to(self.device)
        z_j = F.normalize(emb_j, dim=1).to(self.device)

        # Concatenate the normalized embeddings into a single tensor
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute the pairwise similarity matrix between the representations
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Get the positive pairs from the similarity matrix (diagonal elements at positions k,k+batch_size and k+batch_size,k)
        sim_ij = torch.diag(similarity_matrix, batchsize)
        sim_ji = torch.diag(similarity_matrix, -batchsize)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Compute the nominator and denominator for the contrastive loss
        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        # Compute the partial loss for each pair of embeddings
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        # Compute the average loss for the batch
        loss = torch.sum(loss_partial) / (2 * batchsize)

        return loss

    
def cosine_sim(im, s):
    # normalize the image and sentence embeddings
    im = F.normalize(im, p=2, dim=1)
    s = F.normalize(s, p=2, dim=1)
    # compute cosine similarity
    return im.mm(s.t())

class inter_ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(inter_ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, image_embd1, image_embd2, caption_embd1, caption_embd2):
        # compute image-sentence score matrix
        scores1 = self.sim(image_embd1, caption_embd2)
        scores2 = self.sim(caption_embd1, image_embd2)

        # compute the diagonal elements
        diagonal1 = scores1.diag().view(image_embd1.size(0), 1)
        diagonal2 = scores2.diag().view(caption_embd1.size(0), 1)

        # expand the diagonal elements to be the same size as scores
        d11 = diagonal1.expand_as(scores1)
        d12 = diagonal1.expand_as(scores2)
        d21 = diagonal2.expand_as(scores1)
        d22 = diagonal2.expand_as(scores2)

        # compute the losses
        # image2-caption1
        cost_im = (self.margin + scores1 - d21).clamp(min=0)
        # caption2-image1
        cost_s = (self.margin + scores2 - d12).clamp(min=0)

        # clear diagonals
        mask1 = torch.eye(scores1.size(0)) > .5
        mask2 = torch.eye(scores2.size(0)) > .5
        I1 = Variable(mask1)
        I2 = Variable(mask2)
        if torch.cuda.is_available():
            I1 = I1.cuda()
            I2 = I2.cuda()
        cost_im = cost_im.masked_fill_(I1, 0)
        cost_s = cost_s.masked_fill_(I2, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() , cost_im.sum()
class finetune_ContrastiveLoss(nn.Module):
    """
    Implements contrastive loss for image-text matching.
    """
    def __init__(self, margin: float = 0, max_violation: bool = False):
        super(finetune_ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, im: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Computes the contrastive loss between image and text features.

        Args:
            im: A tensor of shape (batch_size, feature_dim) representing image features.
            s: A tensor of shape (batch_size, feature_dim) representing text features.

        Returns:
            A scalar representing the contrastive loss between the image and text features.
        """
        # Compute image-sentence score matrix.
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # Compare every diagonal score to scores in its column for caption retrieval.
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # Compare every diagonal score to scores in its row for image retrieval.
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # Clear diagonals.
        mask = torch.eye(scores.size(0)) > .5
        I = mask.to(im.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # Keep the maximum violating negative for each query if self.max_violation is True.
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
    
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


