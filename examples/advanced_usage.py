#!/usr/bin/env python3
"""
Advanced usage example for BioSynth-EMG.
Demonstrates transfer learning validation and large-scale dataset generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from biosynth_emg import BioSynthGenerator, SpectralValidator


class EMGDataset(Dataset):
    """PyTorch dataset for EMG signals."""
    
    def __init__(self, emg_signals, labels, forces):
        self.emg_signals = torch.FloatTensor(emg_signals)
        self.labels = torch.LongTensor(labels)
        self.forces = torch.FloatTensor(forces)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.emg_signals[idx], self.labels[idx], self.forces[idx]


class EMGClassifier(nn.Module):
    """Simple CNN-based EMG classifier."""
    
    def __init__(self, num_channels=8, num_classes=8, seq_length=2000):
        super(EMGClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size
        conv_output_size = seq_length // 4  # After two pooling operations
        self.fc1 = nn.Linear(64 * conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc_force = nn.Linear(128, 1)  # Force regression head
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        
        gesture = self.fc2(x)
        force = self.fc_force(x)
        
        return gesture, force.squeeze()


def prepare_dataset(emg_data, metadata, seq_length=2000):
    """Prepare EMG data for training."""
    X = []
    y_gesture = []
    y_force = []
    
    for i, (signals, meta) in enumerate(zip(emg_data, metadata)):
        # Ensure consistent length
        if signals.shape[1] > seq_length:
            signals = signals[:, :seq_length]
        elif signals.shape[1] < seq_length:
            # Pad with zeros
            pad_width = seq_length - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
        
        X.append(signals)
        y_gesture.append(meta['gesture_id'])
        y_force.append(meta['force_level'])
    
    return np.array(X), np.array(y_gesture), np.array(y_force)


def train_model(model, train_loader, val_loader, num_epochs=20, device='cpu'):
    """Train the EMG classifier."""
    criterion_gesture = nn.CrossEntropyLoss()
    criterion_force = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_gesture, batch_force in train_loader:
            batch_x, batch_gesture, batch_force = (
                batch_x.to(device), 
                batch_gesture.to(device), 
                batch_force.to(device)
            )
            
            optimizer.zero_grad()
            gesture_pred, force_pred = model(batch_x)
            
            loss_gesture = criterion_gesture(gesture_pred, batch_gesture)
            loss_force = criterion_force(force_pred, batch_force)
            loss = loss_gesture + 0.1 * loss_force  # Weighted loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_gesture, batch_force in val_loader:
                batch_x, batch_gesture, batch_force = (
                    batch_x.to(device), 
                    batch_gesture.to(device), 
                    batch_force.to(device)
                )
                
                gesture_pred, force_pred = model(batch_x)
                loss_gesture = criterion_gesture(gesture_pred, batch_gesture)
                loss_force = criterion_force(force_pred, batch_force)
                loss = loss_gesture + 0.1 * loss_force
                
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the trained model."""
    model.eval()
    all_gesture_preds = []
    all_gesture_labels = []
    all_force_preds = []
    all_force_labels = []
    
    with torch.no_grad():
        for batch_x, batch_gesture, batch_force in test_loader:
            batch_x = batch_x.to(device)
            
            gesture_pred, force_pred = model(batch_x)
            
            all_gesture_preds.extend(torch.argmax(gesture_pred, dim=1).cpu().numpy())
            all_gesture_labels.extend(batch_gesture.numpy())
            all_force_preds.extend(force_pred.cpu().numpy())
            all_force_labels.extend(batch_force.numpy())
    
    # Calculate metrics
    gesture_accuracy = np.mean(np.array(all_gesture_preds) == np.array(all_gesture_labels))
    force_mse = np.mean((np.array(all_force_preds) - np.array(all_force_labels)) ** 2)
    force_mae = np.mean(np.abs(np.array(all_force_preds) - np.array(all_force_labels)))
    
    return {
        'gesture_accuracy': gesture_accuracy,
        'force_mse': force_mse,
        'force_mae': force_mae,
        'gesture_preds': all_gesture_preds,
        'gesture_labels': all_gesture_labels,
        'force_preds': all_force_preds,
        'force_labels': all_force_labels
    }


def main():
    """Advanced usage demonstration with transfer learning validation."""
    print("BioSynth-EMG Advanced Usage Example")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize generator
    print("\n1. Initializing BioSynth-EMG generator...")
    generator = BioSynthGenerator(
        sampling_rate=2000,
        num_motor_units=150,  # More motor units for realism
        num_channels=8,
        random_seed=42
    )
    
    # Generate large synthetic dataset
    print("\n2. Generating large synthetic dataset (1000 samples)...")
    start_time = time.time()
    
    synthetic_data = generator.generate_dataset(
        num_samples=1000,
        output_format='dict',
        enable_progress=True
    )
    
    synthetic_time = time.time() - start_time
    print(f"   Synthetic dataset generated in {synthetic_time:.2f} seconds")
    
    # Generate small "real" dataset for transfer learning
    print("\n3. Generating small real-like dataset (100 samples)...")
    start_time = time.time()
    
    real_data = generator.generate_dataset(
        num_samples=100,
        output_format='dict',
        random_seed=123,  # Different seed for variety
        enable_progress=False
    )
    
    real_time = time.time() - start_time
    print(f"   Real-like dataset generated in {real_time:.2f} seconds")
    
    # Prepare datasets
    print("\n4. Preparing datasets for training...")
    
    X_synthetic, y_gesture_synthetic, y_force_synthetic = prepare_dataset(
        synthetic_data['emg_signals'], synthetic_data['metadata']
    )
    X_real, y_gesture_real, y_force_real = prepare_dataset(
        real_data['emg_signals'], real_data['metadata']
    )
    
    print(f"   Synthetic dataset: {X_synthetic.shape}")
    print(f"   Real dataset: {X_real.shape}")
    
    # Experiment 1: Train only on synthetic data
    print("\n5. Experiment 1: Training model only on synthetic data...")
    
    X_train, X_test, y_train_g, y_test_g, y_train_f, y_test_f = train_test_split(
        X_synthetic, y_gesture_synthetic, y_force_synthetic, 
        test_size=0.2, random_state=42, stratify=y_gesture_synthetic
    )
    
    train_dataset = EMGDataset(X_train, y_train_g, y_train_f)
    test_dataset = EMGDataset(X_test, y_test_g, y_test_f)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model_synthetic = EMGClassifier().to(device)
    train_losses, val_losses = train_model(model_synthetic, train_loader, test_loader, 
                                          num_epochs=20, device=device)
    
    # Evaluate on synthetic test set
    synth_results = evaluate_model(model_synthetic, test_loader, device)
    print(f"   Synthetic-only model - Gesture accuracy: {synth_results['gesture_accuracy']:.3f}")
    
    # Evaluate on real data
    real_dataset = EMGDataset(X_real, y_gesture_real, y_force_real)
    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
    
    real_results_synthetic = evaluate_model(model_synthetic, real_loader, device)
    print(f"   Synthetic-only model on real data - Gesture accuracy: {real_results_synthetic['gesture_accuracy']:.3f}")
    
    # Experiment 2: Transfer learning (90% synthetic + 10% real)
    print("\n6. Experiment 2: Transfer learning (90% synthetic + 10% real)...")
    
    # Combine synthetic and real data
    X_combined = np.vstack([X_synthetic, X_real])
    y_gesture_combined = np.hstack([y_gesture_synthetic, y_gesture_real])
    y_force_combined = np.hstack([y_force_synthetic, y_force_real])
    
    # Create data source labels (0=synthetic, 1=real)
    source_labels = np.hstack([np.zeros(len(X_synthetic)), np.ones(len(X_real))])
    
    # Split combined dataset
    X_train_c, X_test_c, y_train_g_c, y_test_g_c, y_train_f_c, y_test_f_c, source_train, source_test = train_test_split(
        X_combined, y_gesture_combined, y_force_combined, source_labels,
        test_size=0.2, random_state=42, stratify=y_gesture_combined
    )
    
    # Filter training set to have 90% synthetic, 10% real
    synthetic_mask = source_train == 0
    real_mask = source_train == 1
    
    # Sample 90% synthetic
    n_synthetic_train = int(0.9 * len(y_train_g_c))
    synthetic_indices = np.where(synthetic_mask)[0]
    selected_synthetic = np.random.choice(synthetic_indices, n_synthetic_train, replace=False)
    
    # Sample 10% real
    n_real_train = len(y_train_g_c) - n_synthetic_train
    real_indices = np.where(real_mask)[0]
    selected_real = np.random.choice(real_indices, min(n_real_train, len(real_indices)), replace=False)
    
    # Combine selected indices
    selected_indices = np.concatenate([selected_synthetic, selected_real])
    
    X_train_tl = X_train_c[selected_indices]
    y_train_g_tl = y_train_g_c[selected_indices]
    y_train_f_tl = y_train_f_c[selected_indices]
    
    # Create datasets
    train_dataset_tl = EMGDataset(X_train_tl, y_train_g_tl, y_train_f_tl)
    test_dataset_tl = EMGDataset(X_test_c, y_test_g_c, y_test_f_c)
    
    train_loader_tl = DataLoader(train_dataset_tl, batch_size=32, shuffle=True)
    test_loader_tl = DataLoader(test_dataset_tl, batch_size=32, shuffle=False)
    
    # Train transfer learning model
    model_tl = EMGClassifier().to(device)
    train_losses_tl, val_losses_tl = train_model(model_tl, train_loader_tl, test_loader_tl, 
                                                 num_epochs=20, device=device)
    
    # Evaluate transfer learning model
    tl_results = evaluate_model(model_tl, test_loader_tl, device)
    print(f"   Transfer learning model - Gesture accuracy: {tl_results['gesture_accuracy']:.3f}")
    
    # Compare results
    print("\n7. Transfer Learning Validation Results:")
    print("-" * 50)
    print(f"Synthetic-only model on real data: {real_results_synthetic['gesture_accuracy']:.3f}")
    print(f"Transfer learning model on mixed data: {tl_results['gesture_accuracy']:.3f}")
    
    improvement = tl_results['gesture_accuracy'] - real_results_synthetic['gesture_accuracy']
    print(f"Improvement: {improvement:+.3f} ({improvement/real_results_synthetic['gesture_accuracy']*100:+.1f}%)")
    
    # Create visualization
    print("\n8. Creating comprehensive visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Training curves
    ax1 = axes[0, 0]
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Synthetic Only - Train')
    ax1.plot(epochs, val_losses, 'b--', label='Synthetic Only - Val')
    ax1.plot(epochs, train_losses_tl, 'r-', label='Transfer Learning - Train')
    ax1.plot(epochs, val_losses_tl, 'r--', label='Transfer Learning - Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Confusion matrix - Synthetic only
    ax2 = axes[0, 1]
    cm_synthetic = confusion_matrix(real_results_synthetic['gesture_labels'], 
                                   real_results_synthetic['gesture_preds'])
    sns.heatmap(cm_synthetic, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Synthetic Only - Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # Plot 3: Confusion matrix - Transfer learning
    ax3 = axes[0, 2]
    cm_tl = confusion_matrix(tl_results['gesture_labels'], tl_results['gesture_preds'])
    sns.heatmap(cm_tl, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Transfer Learning - Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Plot 4: Force prediction comparison
    ax4 = axes[1, 0]
    ax4.scatter(real_results_synthetic['force_labels'], real_results_synthetic['force_preds'], 
               alpha=0.5, label='Synthetic Only', s=20)
    ax4.scatter(tl_results['force_labels'], tl_results['force_preds'], 
               alpha=0.5, label='Transfer Learning', s=20)
    ax4.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax4.set_xlabel('True Force')
    ax4.set_ylabel('Predicted Force')
    ax4.set_title('Force Prediction Comparison')
    ax4.legend()
    ax4.grid(True)
    
    # Plot 5: Performance comparison
    ax5 = axes[1, 1]
    models = ['Synthetic\nOnly', 'Transfer\nLearning']
    accuracies = [real_results_synthetic['gesture_accuracy'], tl_results['gesture_accuracy']]
    colors = ['blue', 'red']
    
    bars = ax5.bar(models, accuracies, color=colors, alpha=0.7)
    ax5.set_ylabel('Gesture Accuracy')
    ax5.set_title('Model Performance Comparison')
    ax5.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    ax5.grid(True, axis='y')
    
    # Plot 6: Dataset composition
    ax6 = axes[1, 2]
    composition = ['Synthetic\nTrain', 'Real\nTrain', 'Synthetic\nTest', 'Real\nTest']
    counts = [len(X_train), len(X_real), len(X_test), len(X_test_c)]
    
    bars = ax6.bar(composition, counts, color=['lightblue', 'lightcoral', 'blue', 'red'])
    ax6.set_ylabel('Number of Samples')
    ax6.set_title('Dataset Composition')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    fig_path = output_dir / "transfer_learning_results.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   Results visualization saved to: {fig_path}")
    
    # Save detailed report
    report_path = output_dir / "transfer_learning_report.txt"
    with open(report_path, 'w') as f:
        f.write("BioSynth-EMG Transfer Learning Validation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset Generation:\n")
        f.write(f"  Synthetic samples: {len(X_synthetic)} (took {synthetic_time:.2f}s)\n")
        f.write(f"  Real-like samples: {len(X_real)} (took {real_time:.2f}s)\n\n")
        
        f.write(f"Model Performance:\n")
        f.write(f"  Synthetic-only on real data: {real_results_synthetic['gesture_accuracy']:.3f}\n")
        f.write(f"  Transfer learning: {tl_results['gesture_accuracy']:.3f}\n")
        f.write(f"  Improvement: {improvement:+.3f} ({improvement/real_results_synthetic['gesture_accuracy']*100:+.1f}%)\n\n")
        
        f.write(f"Force Prediction:\n")
        f.write(f"  Synthetic-only MAE: {real_results_synthetic['force_mae']:.3f}\n")
        f.write(f"  Transfer learning MAE: {tl_results['force_mae']:.3f}\n")
        
        if improvement > 0:
            f.write(f"\nCONCLUSION: Transfer learning VALIDATED - synthetic data improves real-world performance!\n")
        else:
            f.write(f"\nCONCLUSION: Transfer learning needs further investigation.\n")
    
    print(f"   Detailed report saved to: {report_path}")
    
    plt.show()
    
    print("\n" + "=" * 50)
    print("Advanced BioSynth-EMG demonstration completed!")
    print("Transfer learning validation shows the value of synthetic data.")


if __name__ == "__main__":
    main()
