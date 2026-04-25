'''
Enhanced MethodModule class for MLP experiments with different architectures, optimizers, and loss functions
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Enhanced_Method_MLP(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    
    # training metrics tracking
    train_losses = []
    train_accuracies = []
    
    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # Configuration parameters
        self.config = config or {}
        self.architecture = self.config.get('architecture', [784, 512, 256, 128, 10])
        self.optimizer_type = self.config.get('optimizer', 'adam').lower()
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        self.weight_decay = self.config.get('weight_decay', 0.0)
        self.momentum = self.config.get('momentum', 0.9)
        self.batch_norm = self.config.get('batch_norm', False)
        self.activation_type = self.config.get('activation', 'relu').lower()
        
        # Build network layers
        self._build_network()
        
    def _build_network(self):
        """Build the neural network architecture"""
        layers = []
        
        # Input layer to first hidden layer
        for i in range(len(self.architecture) - 1):
            input_size = self.architecture[i]
            output_size = self.architecture[i + 1]
            
            # Linear layer
            layers.append(nn.Linear(input_size, output_size))
            
            # Add batch normalization (except for output layer)
            if self.batch_norm and i < len(self.architecture) - 2:
                layers.append(nn.BatchNorm1d(output_size))
            
            # Add activation function (except for output layer)
            if i < len(self.architecture) - 2:
                if self.activation_type == 'relu':
                    layers.append(nn.ReLU())
                elif self.activation_type == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.01))
                elif self.activation_type == 'elu':
                    layers.append(nn.ELU())
                elif self.activation_type == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())
                
                # Add dropout (except for output layer)
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward propagation"""
        return self.network(x)
    
    def _get_optimizer(self):
        """Get optimizer based on configuration"""
        if self.optimizer_type == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _get_loss_function(self):
        """Get loss function based on configuration"""
        loss_type = self.config.get('loss_function', 'cross_entropy').lower()
        
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_type == 'cross_entropy_weighted':
            # Calculate class weights based on training data distribution
            if hasattr(self, 'data') and 'train' in self.data:
                y_train = np.array(self.data['train']['y'])
                class_counts = np.bincount(y_train)
                class_weights = 1.0 / (class_counts + 1e-6)
                class_weights = class_weights / class_weights.sum() * len(class_counts)
                return nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
            else:
                return nn.CrossEntropyLoss()
        elif loss_type == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            return nn.CrossEntropyLoss()
    
    def fit_model(self, X, y):
        """Training method with configurable optimizer and loss function"""
        optimizer = self._get_optimizer()
        loss_function = self._get_loss_function()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        
        # Clear previous metrics
        self.train_losses = []
        self.train_accuracies = []
        
        # Learning rate scheduler
        scheduler = None
        if self.config.get('use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        print(f"Training with {self.optimizer_type.upper()} optimizer, LR: {self.learning_rate}")
        if self.weight_decay > 0:
            print(f"Weight decay: {self.weight_decay}")
        if self.dropout_rate > 0:
            print(f"Dropout rate: {self.dropout_rate}")
        if self.batch_norm:
            print("Using Batch Normalization")
        
        for epoch in range(self.max_epoch):
            # Convert data to tensors
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            y_true = torch.LongTensor(np.array(y))
            
            # Calculate loss
            train_loss = loss_function(y_pred, y_true)
            
            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Update learning rate if using scheduler
            if scheduler:
                scheduler.step()
            
            # Calculate and store metrics
            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            epoch_accuracy = accuracy_evaluator.evaluate()
            epoch_loss = train_loss.item()
            
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)
            
            # Print progress with configurable interval
            print_interval = 25 if self.max_epoch <= 100 else 100
            if epoch % print_interval == 0 or epoch == self.max_epoch - 1:
                current_lr = optimizer.param_groups[0]['lr']
                progress_pct = (epoch + 1) / self.max_epoch * 100
                print(f'Epoch: {epoch}/{self.max_epoch} ({progress_pct:.1f}%), Accuracy: {epoch_accuracy:.4f}, Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}')
    
    def test(self, X):
        """Testing method"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            return y_pred.max(1)[1]
    
    def run(self):
        """Main run method"""
        print('Enhanced MLP method running...')
        print('--start training...')
        self.fit_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
