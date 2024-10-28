import torch
import torch.nn as nn
import time
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init
from tqdm import trange
from utils import calculate_auc
from utils import structured_negative_sampling

# from torchviz import make_dot

class LightSGCNNetwork(nn.Module):
    def __init__(self, device, args, X):
        super().__init__()
        """
        LightSGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        
        # Initialize a linear layer
        self.linear = nn.Linear(in_features=X.shape[1], out_features=self.args.out_features)
        self.to(self.device)

        self.regression_weights = Parameter(torch.Tensor(2 * self.args.out_features, 3))
        self.regression_bias = Parameter(torch.FloatTensor(3))
        init.xavier_normal_(self.regression_weights)
        self.regression_bias.data.fill_(0.0)

        self.X = self.X.to("cuda")

    def forward(self, positive_edges, negative_edges, target, prob_vi):
        """
        Forward pass of the LightSGCN.
        :param X: Input features.
        :return: Processed output.
        """
        # Pass the input through the linear layer
        self.z = self.linear(self.X)
        loss = self.calculate_loss_function(self.z, positive_edges, negative_edges, target, prob_vi)

        return loss, self.z
    
    def calculate_loss_function(self, z, positive_edges, negative_edges, target, prob_vi):
        loss1, _ = self.calculate_edge_loss(z, target, positive_edges, negative_edges)
        loss2 = self.calculate_mse_loss(z, positive_edges, negative_edges, target, prob_vi)
        loss = loss1 + self.args.lamb * loss2
        return loss

    def calculate_edge_loss(self, z, target, positive_edges, negative_edges):
        """
        Calculating the regression loss for all pairs of nodes.
        :param z: Hidden vertex representations.
        :param target: Target vector.
        :return loss_term: Regression loss.
        :return predictions_soft: Predictions for each vertex pair.
        """

        i, j, k = structured_negative_sampling(positive_edges,z.shape[0])
        self.positive_z_i = z[i]
        self.positive_z_j = z[j]
        self.positive_z_k = z[k]

        i, j, k = structured_negative_sampling(negative_edges,z.shape[0])
        self.negative_z_i = z[i]
        self.negative_z_j = z[j]
        self.negative_z_k = z[k]

        pos = torch.cat((self.positive_z_i, self.positive_z_j), 1)
        neg = torch.cat((self.negative_z_i, self.negative_z_j), 1)

        surr_neg_i = torch.cat((self.negative_z_i, self.negative_z_k), 1)
        surr_neg_j = torch.cat((self.negative_z_j, self.negative_z_k), 1)
        surr_pos_i = torch.cat((self.positive_z_i, self.positive_z_k), 1)
        surr_pos_j = torch.cat((self.positive_z_j, self.positive_z_k), 1)

        features = torch.cat((pos, neg, surr_neg_i, surr_neg_j, surr_pos_i, surr_pos_j))
        predictions = torch.mm(features, self.regression_weights) + self.regression_bias
        predictions_soft = F.log_softmax(predictions, dim=1)
        loss_term = F.nll_loss(predictions_soft, target)
        return loss_term, predictions_soft

    def calculate_mse_loss(self, z, positive_edges, negative_edges, target, prob_vi):
        """
        Calculate the loss function using MSE based on the given formula.
        
        :param z: The hidden representations of nodes.
        :param positive_edges: Tensor of positive edges (2, N).
        :param negative_edges: Tensor of negative edges (2, N).
        :param target: Ground truth labels (1 for positive, 0 for negative edges).
        :param prob_vi: Pre-computed prob_vi for each node.
        :return: Calculated loss.
        """
    
        # Initialize loss
        mse_loss = 0
        
        # Calculate predictions for positive edges
        for idx in range(positive_edges.shape[1]):  # Iterate over columns (edges)
            i, j = positive_edges[:, idx]  # Get the i, j node pair
            
            # Cosine similarity between node i and j embeddings
            prob_vivj = torch.cosine_similarity(z[i], z[j], dim=0)
            
            # Predicted label based on the formula
            y_hat = self.args.beta * prob_vivj + (1 - self.args.beta) / 2 * (prob_vi[i] + prob_vi[j])
            
            # Add MSE for this positive edge (target label is 1)
            mse_loss += 0.5 * (1 - y_hat) ** 2
        
        # Calculate predictions for negative edges
        for idx in range(negative_edges.shape[1]):  # Iterate over columns (edges)
            i, j = negative_edges[:, idx]  # Get the i, j node pair
            
            # Cosine similarity between node i and j embeddings
            prob_vivj = torch.cosine_similarity(z[i], z[j], dim=0)
            
            # Predicted label based on the formula
            y_hat = self.args.beta * prob_vivj + (1 - self.args.beta) / 2 * (prob_vi[i] + prob_vi[j])
            
            # Add MSE for this negative edge (target label is 0)
            mse_loss += 0.5 * (0 - y_hat) ** 2
        
        return mse_loss

        # mse_loss = torch.tensor(0.1, requires_grad=True)

        # return mse_loss



class LightSGCNTrainer(object):
    def __init__(self, args, X, prob_vi, positive_edges, negative_edges, test_positive_edges, test_negative_edges, y, edges):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.args = args
        # self.X = X
        # self.prob_vi = prob_vi
        # self.positive_edges = positive_edges
        # self.negative_edges = negative_edges
        # self.test_positive_edges = test_positive_edges
        # self.test_negative_edges = test_negative_edges
        # self.y = y

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.X = torch.tensor(X, device=self.device) if not isinstance(X, torch.Tensor) else X.to(self.device)
        self.prob_vi = torch.tensor(prob_vi, device=self.device) if not isinstance(prob_vi, torch.Tensor) else prob_vi.to(self.device)
        self.positive_edges = torch.tensor(positive_edges, device=self.device) if not isinstance(positive_edges, torch.Tensor) else positive_edges.to(self.device)
        self.negative_edges = torch.tensor(negative_edges, device=self.device) if not isinstance(negative_edges, torch.Tensor) else negative_edges.to(self.device)
        self.test_positive_edges = torch.tensor(test_positive_edges, device=self.device) if not isinstance(test_positive_edges, torch.Tensor) else test_positive_edges.to(self.device)
        self.test_negative_edges = torch.tensor(test_negative_edges, device=self.device) if not isinstance(test_negative_edges, torch.Tensor) else test_negative_edges.to(self.device)
        self.y = torch.tensor(y, device=self.device) if not isinstance(y, torch.Tensor) else y.to(self.device)
        self.edges = edges

        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance"] = [["Epoch", "AUC", "F1", "pos_ratio"]]
        self.logs["training_time"] = [["Epoch", "Seconds"]]

    # def score_model(self, epoch):
    #     """
    #     Score the model on the test set edges in each epoch.
    #     :param epoch: Epoch number.
    #     """
    #     loss, self.train_z = self.model(self.positive_edges, self.negative_edges, self.y, self.prob_vi)
    #     score_positive_edges = torch.from_numpy(np.array(self.test_positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
    #     score_negative_edges = torch.from_numpy(np.array(self.test_negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)
    #     test_positive_z = torch.cat((self.train_z[score_positive_edges[0, :], :], self.train_z[score_positive_edges[1, :], :]), 1)
    #     test_negative_z = torch.cat((self.train_z[score_negative_edges[0, :], :], self.train_z[score_negative_edges[1, :], :]), 1)
    #     scores = torch.mm(torch.cat((test_positive_z, test_negative_z), 0), self.model.regression_weights.to(self.device)) + self.model.regression_bias.to(self.device)
    #     probability_scores = torch.exp(F.softmax(scores, dim=1))
    #     predictions = probability_scores[:, 0]/probability_scores[:, 0:2].sum(1)
    #     predictions = predictions.cpu().detach().numpy()
    #     targets = [0]*len(self.test_positive_edges) + [1]*len(self.test_negative_edges)
    #     auc, f1, pos_ratio = calculate_auc(targets, predictions, self.edges)
    #     self.logs["performance"].append([epoch+1, auc, f1, pos_ratio])

    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Current epoch number.
        """
        # Compute loss and get the latent representation of nodes after training
        loss, self.train_z = self.model(self.positive_edges, self.negative_edges, self.y, self.prob_vi)
        
        # If self.test_positive_edges and self.test_negative_edges are CUDA tensors, move them to CPU
        if isinstance(self.test_positive_edges, torch.Tensor) and self.test_positive_edges.is_cuda:
            self.test_positive_edges = self.test_positive_edges.cpu()

        if isinstance(self.test_negative_edges, torch.Tensor) and self.test_negative_edges.is_cuda:
            self.test_negative_edges = self.test_negative_edges.cpu()

        # Now convert test positive/negative edges to tensors and move them back to the appropriate device (GPU/CPU)
        score_positive_edges = torch.from_numpy(np.array(self.test_positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        score_negative_edges = torch.from_numpy(np.array(self.test_negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        
        # Extract the embeddings of test edges from train_z
        test_positive_z = torch.cat((self.train_z[score_positive_edges[0, :], :], 
                                    self.train_z[score_positive_edges[1, :], :]), 1)
        test_negative_z = torch.cat((self.train_z[score_negative_edges[0, :], :], 
                                    self.train_z[score_negative_edges[1, :], :]), 1)
        
        # Compute the scores using the regression weights and bias from the model
        scores = torch.mm(torch.cat((test_positive_z, test_negative_z), 0), 
                        self.model.regression_weights.to(self.device)) + self.model.regression_bias.to(self.device)
        
        # Calculate probability scores using softmax and exponentiation
        probability_scores = torch.exp(F.softmax(scores, dim=1))
        
        # Compute predictions by normalizing the probabilities
        predictions = probability_scores[:, 0] / probability_scores[:, 0:2].sum(1)
        predictions = predictions.cpu().detach().numpy()  # Convert to NumPy format
        
        # Create target labels (0 for positive edges, 1 for negative edges)
        targets = [0] * len(self.test_positive_edges) + [1] * len(self.test_negative_edges)
        
        # Calculate AUC, F1 score, and the positive edge ratio
        auc, f1, pos_ratio = calculate_auc(targets, predictions, self.edges)
        
        # Log the performance metrics (AUC, F1, pos_ratio) for the current epoch
        self.logs["performance"].append([epoch + 1, auc, f1, pos_ratio])


    def create_and_train_model(self):
        self.model = LightSGCNNetwork(self.device, self.args, self.X).to(self.device)
        # for name, param in self.model.named_parameters():
        #     print(f"Layer: {name} | Shape: {param.shape} | Values: {param[:5]}")
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        
        self.model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")
        for epoch in self.epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, _ = self.model(self.positive_edges, self.negative_edges, self.y, self.prob_vi)

            # make_dot(loss, params=dict(self.model.named_parameters())).render("computational_graph", format="png")
            
            loss.backward()
            self.epochs.set_description("LightSGCN (Loss=%g)" % round(loss.item(), 4))
            self.optimizer.step()
            self.logs["training_time"].append([epoch+1, time.time()-start_time])
            if self.args.test_size > 0:
                self.score_model(epoch)
        
