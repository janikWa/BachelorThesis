import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import levy_stable
import utils.optimreg as optimreg
import os 
import pickle 
from data import sqlitedb, h5pydb
from estimator.stable_estimators import robust_alpha_estimator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FCNet(nn.Module):
    def __init__(self, layer_sizes=[784, 256, 256, 256, 10],
                 activation='relu', weight_init='gaussian', params = []):
        """
        layer_sizes: list of integers
        activation: 'relu', 'tanh', 'sigmoid'
        weight_init: 'gaussian' or 'heavy_tail'
        alpha: alpha for heavy tailes
        sigma: scale parameter 
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.activation_name = activation
        self.weight_init = weight_init
        self.params = params
        
        
        self.init_weights()
        
    def init_weights(self):
        for layer in self.layers:
            if self.weight_init == 'gaussian':

                if self.params == []:
                    mu = 0.0
                    sigma =  1.0 
                else: 
                    mu, sigma = self.params[:2]

                nn.init.normal_(layer.weight, mean=mu, std=sigma / (layer.weight.size(1)**0.5))

            elif self.weight_init == 'heavy_tail':

                if self.params == []: 
                    alpha = 1.5 # tail index
                    beta = 0.5 # skewness
                    gamma = 1.0 # scale param 
                    delta = 0.0 # loc param 
                else: 
                    alpha, beta, gamma, delta = self.params[:4]
         
                scale = gamma * (0.5 / layer.weight.size(1))**(1./alpha)

                weight_np = levy_stable.rvs(alpha, beta, scale=scale, loc=delta, size=layer.weight.shape)
                layer.weight.data = torch.tensor(weight_np, dtype=torch.float32)

            nn.init.constant_(layer.bias, 0.0)
            
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.activation_name == 'relu':
                x = F.relu(x)
            elif self.activation_name == 'tanh':
                x = torch.tanh(x)
            elif self.activation_name == 'sigmoid':
                x = torch.sigmoid(x)
            else:
                raise ValueError("Unknown activation")

        x = self.layers[-1](x)
        return x

def train(model, train_loader, val_loader, optimizer=None, run=0, epochs=5, logging=True, model_name=None, regularizer=None, lambda_reg=0.01, architecture=None):

    if logging and model_name is None: 
        raise Exception("If 'logging' is True, 'model_name' must be defined.")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
  
            if regularizer == "hill": 
                loss += optimreg.hill_regularizer(model, reduction="sum")
            elif regularizer == "hill_weighted": 
                loss += optimreg.hill_regularizer_weighted(model)
            elif regularizer == "parabolic_hill_spec_layers0": 
                loss += optimreg.parabolic_hill_spec_layers(model, 0)
            elif regularizer == "parabolic_hill_spec_layers1": 
                loss += optimreg.parabolic_hill_spec_layers(model, 1)
            elif regularizer == "parabolic_hill_spec_layers2": 
                loss += optimreg.parabolic_hill_spec_layers(model, 2)
            elif regularizer == "parabolic_hill": 
                loss += optimreg.parabolic_hill(model)
            elif regularizer == "xiao": 
                loss += optimreg.weighted_alpha_regularizer(model)
            elif regularizer == "decay": 
                loss += optimreg.decay_weighted_alpha_regularizer(model, epoch+1)
            elif regularizer == "lower": 
                loss += optimreg.lower_threshold_weighted_alpha_regularizer(model)
            elif regularizer == "lasso": 
                loss += optimreg.l1_regularization(model, lambda_reg)
            elif regularizer == "ridge": 
                loss += optimreg.l2_regularization(model, lambda_reg)


            loss.backward()
        
            optimizer.step()
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        
        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total * 100

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


        if logging:
            h5pydb.log_training_data_h5(
                model=model,
                model_name=model_name,
                architecture=architecture,
                weight_init=model.weight_init,
                params=model.params,
                run=run,
                epoch=epoch + 1,
                optimizer=optimizer,
                regularizer=regularizer,
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc
            )



# def train(model, train_loader, optimizer, run=0, epochs=5, logging=True, model_name=None, regularizer=None, lambda_reg=0.01, architecture=None):

#     if logging and model_name is None: 
#         raise Exception("If 'log_weights' is True, 'weights_name' needs to be defined")
    
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()

#     weight_dict = {}    
    
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = criterion(outputs, y)
            
#             if regularizer == "remore":
#                 loss += lambda_reg * optimreg.remore(model)
#             elif regularizer == "smor":
#                 loss += lambda_reg * optimreg.smor(model)
#             elif regularizer == "parom":
#                 loss += lambda_reg * optimreg.parom(model)
#             elif regularizer == "hill": 
#                 loss += optimreg.hill_regularizer(model, reduction="sum")
#             elif regularizer == "hill_weighted": 
#                 loss += optimreg.hill_regularizer_weighted(model)
#             elif regularizer == "xiao": 
#                 loss += optimreg.weighted_alpha_regularizer(model)
#             elif regularizer == "decay": 
#                 loss += optimreg.decay_weighted_alpha_regularizer(model, epoch+1)
#             elif regularizer == "lower": 
#                 loss += optimreg.lower_threshold_weighted_alpha_regularizer(model)
#             elif regularizer == "lasso": 
#                 loss += optimreg.l1_regularization(model, lambda_reg)
            
            
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
            
#             # Accuracy
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == y).sum().item()
#             total += y.size(0)
        
#         epoch_loss = total_loss / len(train_loader)
#         epoch_acc = correct / total * 100
        
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

#         if logging:
#             h5pydb.log_training_data_h5(
#                 model=model,
#                 model_name=model_name,
#                 architecture=  architecture, 
#                 weight_init = model.weight_init, 
#                 params = model.params, 
#                 run = run, 
#                 epoch=epoch + 1,
#                 optimizer=optimizer,
#                 regularizer=regularizer, 
#                 loss= epoch_loss, 
#                 accuracy= epoch_acc
#             )




def save_weights_per_epoch(model):
    epoch_weight_dict = {}
    for i, layer in enumerate(model.layers):
        epoch_weight_dict[f"layer_{i}"] = {
            "weights": layer.weight.detach().cpu().clone(),
            "bias": layer.bias.detach().cpu().clone()
        }
    return epoch_weight_dict

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc*100:.2f}%")
    return acc


