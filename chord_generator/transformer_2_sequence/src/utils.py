import os
from datetime import datetime


class CustomLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def write_log(self, level, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f'{timestamp} - {level} - {message}\n'
        
        with open(self.log_file, 'a') as file:
            file.write(log_message)

    def log_training_info(self, epochs, train_loss, train_accuracy, val_loss, val_accuracy):
        log_message = f'Epochs: {epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}'
        self.write_log('INFO', log_message)
    
    def log_inference_info(self, epochs, test_loss, test_accuracy):
        log_message = f'Inference on Test Data - Epochs: {epochs}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}'
        self.write_log('INFO', log_message)

    def log_model_loading(self, model_filename, message):
        log_message = f'Model Loading - {message} - {model_filename}'
        self.write_log('CRITICAL', log_message)
