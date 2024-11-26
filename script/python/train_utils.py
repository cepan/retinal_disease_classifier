# train_utils.py
import torch
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            # First epoch, set the best loss and save the model
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, initial=True)
        elif val_loss > self.best_loss - self.delta:
            # Validation loss did not improve
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Validation loss improved
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, initial=False):
        if self.verbose:
            if initial:
                print(f'Validation loss set to {val_loss:.6f} at the start. Saving model...')
            else:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)



def train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, epochs):
    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0
        print(f"Epoch [{epoch+1}/{epochs}]")
        start_time = time.time()
              
        for images, labels in train_loader:  
            
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)           
            loss = criterion(outputs, labels) 

            # Backward pass and optimization
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()      
            
            running_loss += loss.item()

        end_time = time.time()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}, Time: {end_time-start_time}")
        
        # Validation phase
        model.eval()  
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  
            for images, labels in validation_loader:  
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(validation_loader)

        print(f'Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy:.2f}%')
        
        # Check for early stopping
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy:.2f}%')

    return avg_test_loss, test_accuracy





