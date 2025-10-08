# Import all required libraries for data processing, machine learning, and deep learning
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For data manipulation (if using CSV files)
import matplotlib.pyplot as plt  # For creating visualizations and plots
import seaborn as sns  # For enhanced statistical visualizations
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn.preprocessing import LabelEncoder  # For converting text labels to numerical values
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # For model evaluation
import tensorflow as tf  # Main deep learning framework
from tensorflow import keras  # High-level neural networks API
from tensorflow.keras import layers  # Pre-built neural network layers
from tensorflow.keras.utils import to_categorical  # For converting labels to one-hot encoding
import itertools  # For efficient looping (utility)
import os  # For operating system interactions like file paths

class miRNAClassifier:
    """
    A Convolutional Neural Network (CNN) classifier for miRNA sequences.
    This class handles the entire pipeline from data preprocessing to model evaluation.
    """
    
    def __init__(self, sequence_length=25):
        """
        Initialize the miRNA classifier with configurable parameters.
        
        Args:
            sequence_length (int): Fixed length for all miRNA sequences. 
                                  Sequences longer than this will be truncated,
                                  shorter sequences will be zero-padded.
        """
        self.sequence_length = sequence_length  # Store the fixed sequence length
        self.model = None  # Placeholder for the Keras model (will be built later)
        self.label_encoder = LabelEncoder()  # Encoder for converting string labels to numbers
        self.classes_ = None  # Will store the unique class names after fitting
        self.history = None  # Will store training history for visualization
    
    
    def one_hot_encode_sequence(self, sequence):
        """
        Convert a miRNA sequence string to one-hot encoded numerical representation.
        
        One-hot encoding converts each nucleotide to a 4-element binary vector:
        A -> [1, 0, 0, 0]
        U -> [0, 1, 0, 0] 
        G -> [0, 0, 1, 0]
        C -> [0, 0, 0, 1]
        
        Args:
            sequence (str): miRNA sequence string containing characters A, U, G, C
            
        Returns:
            numpy.ndarray: One-hot encoded sequence of shape (sequence_length, 4)
        """
        # Dictionary mapping each nucleotide to its position in the one-hot vector
        nucleotide_to_index = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        
        # Initialize a zero matrix with dimensions: sequence_length × 4 nucleotides
        encoded = np.zeros((self.sequence_length, 4))
        
        # Iterate through each position in the sequence (up to sequence_length)
        for i, nucleotide in enumerate(sequence[:self.sequence_length]):
            # Check if nucleotide is valid and set the corresponding position to 1
            if nucleotide in nucleotide_to_index:
                encoded[i, nucleotide_to_index[nucleotide]] = 1
        
        return encoded
    
    
    def prepare_data(self, sequences, labels):
        """
        Prepare the miRNA sequence data for training by converting sequences to 
        one-hot encoding and labels to categorical format.
        
        Args:
            sequences (list): List of miRNA sequence strings
            labels (list): List of target gene family labels (strings)
            
        Returns:
            tuple: (X, y) where:
                X (numpy.ndarray): One-hot encoded sequences
                y (numpy.ndarray): Categorical encoded labels (one-hot)
        """
        print("Preparing data...")
        
        # Convert all sequences to one-hot encoding
        X = np.array([self.one_hot_encode_sequence(seq) for seq in sequences])
        
        # Convert string labels to numerical values (0, 1, 2, ...)
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_  # Store the class names
        
        # Convert numerical labels to one-hot encoding for multi-class classification
        y_categorical = to_categorical(y, num_classes=len(self.classes_))
        
        # Print data statistics for verification
        print(f"Data shape: {X.shape}")  # Should be (num_samples, sequence_length, 4)
        print(f"Number of classes: {len(self.classes_)}")
        print(f"Classes: {self.classes_}")
        
        return X, y_categorical
    
    
    def build_model(self, input_shape, num_classes):
        """
        Build a Convolutional Neural Network (CNN) model for sequence classification.
        
        The architecture includes:
        - Three 1D convolutional layers for feature extraction
        - Batch normalization for stable training
        - Max pooling for dimensionality reduction
        - Dropout layers to prevent overfitting
        - Dense layers for final classification
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, 4)
            num_classes (int): Number of target classes to predict
            
        Returns:
            keras.Model: Compiled Keras model ready for training
        """
        print("Building CNN model...")
        
        # Create a sequential model (linear stack of layers)
        model = keras.Sequential([
            # Input layer - defines the expected input shape
            layers.Input(shape=input_shape),
            
            # First 1D convolutional layer - learns local patterns in sequences
            layers.Conv1D(64, kernel_size=5, activation='relu', 
                         padding='same', name='conv1d_1'),
            layers.BatchNormalization(),  # Normalize activations for stable training
            layers.MaxPooling1D(pool_size=2, name='max_pooling1d_1'),  # Reduce sequence length
            layers.Dropout(0.3),  # Randomly disable 30% of neurons to prevent overfitting
            
            # Second convolutional layer - learns more complex patterns
            layers.Conv1D(128, kernel_size=3, activation='relu', 
                         padding='same', name='conv1d_2'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2, name='max_pooling1d_2'),
            layers.Dropout(0.3),
            
            # Third convolutional layer - learns high-level features
            layers.Conv1D(256, kernel_size=3, activation='relu', 
                         padding='same', name='conv1d_3'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(name='global_max_pooling1d'),  # Convert to 1D vector
            layers.Dropout(0.4),
            
            # Fully connected (dense) layers for classification
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer - softmax activation for multi-class classification
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compile the model with optimizer, loss function, and metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Adaptive learning rate
            loss='categorical_crossentropy',  # Suitable for multi-class classification
            metrics=['accuracy']  # Track accuracy during training
        )
        
        print("Model built successfully!")
        model.summary()  # Print model architecture summary
        
        return model
    
    
    def train(self, X, y, test_size=0.2, validation_split=0.2, 
              epochs=100, batch_size=32):
        """
        Train the CNN model on the prepared data.
        
        Args:
            X (numpy.ndarray): Input features (one-hot encoded sequences)
            y (numpy.ndarray): Target labels (one-hot encoded)
            test_size (float): Proportion of data to use for testing (0.0-1.0)
            validation_split (float): Proportion of training data for validation
            epochs (int): Maximum number of training iterations
            batch_size (int): Number of samples per gradient update
            
        Returns:
            tuple: (X_test, y_test, test_accuracy) test data and accuracy
        """
        print("Starting training...")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Build the model with appropriate input shape and number of classes
        self.model = self.build_model(X_train.shape[1:], y_train.shape[1])
        
        # Define callbacks for better training
        callbacks = [
            # Stop training when validation loss stops improving
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            # Reduce learning rate when validation loss plateaus
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
            )
        ]
        
        # Train the model on the training data
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,  # Use part of training data for validation
            callbacks=callbacks,  # Use the defined callbacks
            verbose=1  # Show progress bars
        )
        
        # Evaluate the trained model on the test set
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return X_test, y_test, test_accuracy
    
    
    def plot_training_history(self, save_path=None):
        """
        Plot the training history to visualize model performance over epochs.
        
        Creates two subplots:
        - Left: Training and validation accuracy
        - Right: Training and validation loss
        
        Args:
            save_path (str): Optional path to save the plot image
        """
        if self.history is None:
            print("No training history available!")
            return
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy on first subplot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss on second subplot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()  # Adjust spacing between subplots
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()  # Display the plot
    
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """
        Plot a confusion matrix to visualize classification performance.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): True test labels (one-hot encoded)
            save_path (str): Optional path to save the confusion matrix image
            
        Returns:
            float: Test accuracy calculated from the confusion matrix
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Generate predictions from the model
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
        y_true_classes = np.argmax(y_test, axis=1)  # Convert one-hot to class indices
        
        # Calculate accuracy score
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Create heatmap visualization of confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes_,  # Use actual class names for x-axis
                   yticklabels=self.classes_)  # Use actual class names for y-axis
        plt.title(f'Confusion Matrix\nTest Accuracy: {accuracy:.4f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)  # Rotate x labels for better readability
        plt.yticks(rotation=0)
        
        # Save confusion matrix if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()  # Display the plot
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=self.classes_))
        
        return accuracy
    
    
    def save_model(self, filepath):
        """
        Save the trained model to a file for later use.
        
        Args:
            filepath (str): Path where the model should be saved
        """
        if self.model is None:
            print("No model to save!")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    
    def load_model(self, filepath):
        """
        Load a previously trained model from a file.
        
        Args:
            filepath (str): Path to the saved model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def generate_sample_data(num_samples=1000):
    """
    Generate sample miRNA data for demonstration purposes.
    In real usage, replace this function with your actual data loading code.
    
    Args:
        num_samples (int): Number of sample sequences to generate
        
    Returns:
        tuple: (sequences, labels) containing generated data
    """
    print("Generating sample data...")
    
    # Define possible nucleotides in miRNA sequences
    nucleotides = ['A', 'U', 'G', 'C']
    
    # Define example target gene families for classification
    gene_families = ['Kinase', 'Transcription_Factor', 'Receptor', 
                    'Channel', 'Phosphatase', 'Protease']
    
    sequences = []
    labels = []
    
    # Generate random sequences and labels
    for i in range(num_samples):
        # Create random sequence of length between 20-24 nucleotides
        seq_length = np.random.randint(20, 25)
        sequence = ''.join(np.random.choice(nucleotides, seq_length))
        sequences.append(sequence)
        
        # Assign random gene family label
        label = np.random.choice(gene_families)
        labels.append(label)
    
    return sequences, labels


def main():
    """
    Main function to execute the complete miRNA classification pipeline.
    This function coordinates data preparation, model training, and evaluation.
    """
    print("=== miRNA Classification using CNN ===")
    
    # Create output directory for saving results
    os.makedirs('output', exist_ok=True)
    
    # Initialize the miRNA classifier with desired sequence length
    classifier = miRNAClassifier(sequence_length=25)
    
    # Generate sample data (REPLACE THIS WITH YOUR ACTUAL DATA LOADING)
    sequences, labels = generate_sample_data(2000)
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Number of unique labels: {len(set(labels))}")
    
    # Prepare data: convert sequences to one-hot encoding and labels to categorical
    X, y = classifier.prepare_data(sequences, labels)
    
    # Train the model and get test data for evaluation
    X_test, y_test, test_accuracy = classifier.train(
        X, y, 
        epochs=50,  # Reduced for faster demonstration
        batch_size=32
    )
    
    # Visualize training progress
    classifier.plot_training_history(save_path='output/training_history.png')
    
    # Evaluate model performance with confusion matrix
    final_accuracy = classifier.plot_confusion_matrix(
        X_test, y_test, 
        save_path='output/confusion_matrix.png'
    )
    
    # Save the trained model for future use
    classifier.save_model('output/miRNA_classifier.h5')
    
    # Save accuracy results to a text file
    with open('output/accuracy.txt', 'w') as f:
        f.write(f"Test Accuracy: {final_accuracy:.4f}\n")
    
    # Print completion message with results summary
    print(f"\n=== Training Complete ===")
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    print(f"Model saved: output/miRNA_classifier.h5")
    print(f"Results saved in 'output' directory")


# Standard Python idiom: execute main function only if script is run directly
if __name__ == "__main__":
    main()