# %% [markdown]
# # Person Profile Matching using Graph Neural Networks
# 
# This notebook demonstrates how to use Graph Neural Networks (GNNs) to identify if two person profiles connected by an accomplice represent the same person. This is an entity resolution problem where we leverage graph structure and node features to determine profile similarity.
# 
# ## Problem Overview
# - We have multiple person profiles that might represent the same individual
# - Profiles are connected through accomplice relationships
# - Goal: Determine if two profiles connected by an accomplice are the same person
# - Approach: Use GNN to learn node representations that capture both profile features and graph structure

# %%
# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

# %% [markdown]
# ## 1. Graph Data Structure Setup
# 
# First, we'll create synthetic person profiles and their relationships. Each profile will have:
# - Name
# - Age
# - Location
# - Occupation
# - Unique identifier
# 
# We'll create a graph where:
# - Nodes represent person profiles
# - Edges represent accomplice relationships
# - Some pairs of profiles will actually represent the same person

# %%
# Create synthetic person profiles
def create_synthetic_profiles(num_profiles=100, num_duplicates=20):
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    occupations = ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist']
    names = [f'Person_{i}' for i in range(num_profiles)]
    
    profiles = []
    true_identities = []  # To track which profiles represent the same person
    
    # Create base profiles
    for i in range(num_profiles):
        profile = {
            'id': i,
            'name': names[i],
            'age': np.random.randint(25, 65),
            'location': np.random.choice(locations),
            'occupation': np.random.choice(occupations)
        }
        profiles.append(profile)
        true_identities.append(i)
    
    # Create duplicate profiles with slight variations
    for i in range(num_duplicates):
        original_idx = np.random.randint(0, num_profiles)
        original = profiles[original_idx]
        
        # Create a duplicate with some variations
        duplicate = {
            'id': len(profiles),
            'name': original['name'].replace('Person', 'P'),  # Slight name variation
            'age': original['age'] + np.random.randint(-2, 3),  # Slight age variation
            'location': original['location'],  # Same location
            'occupation': original['occupation']  # Same occupation
        }
        profiles.append(duplicate)
        true_identities.append(true_identities[original_idx])  # Same true identity as original
    
    return pd.DataFrame(profiles), np.array(true_identities)

# Create accomplice relationships
def create_accomplice_edges(num_profiles, num_edges):
    edges = []
    for _ in range(num_edges):
        source = np.random.randint(0, num_profiles)
        target = np.random.randint(0, num_profiles)
        if source != target:
            edges.append([source, target])
    return np.array(edges).T

# Generate synthetic data
num_profiles = 100
num_duplicates = 20
num_edges = 300

profiles_df, true_identities = create_synthetic_profiles(num_profiles, num_duplicates)
edge_index = create_accomplice_edges(len(profiles_df), num_edges)

print("Number of profiles:", len(profiles_df))
print("Number of edges:", edge_index.shape[1])
print("Sample profiles:")
profiles_df

# %% [markdown]
# ## 2. Feature Engineering for Person Profiles
# 
# Now we'll convert profile attributes into numerical features:
# 1. Encode categorical variables (location, occupation)
# 2. Convert names into embeddings using BERT
# 3. Normalize age values
# 4. Combine all features into node feature vectors

# %%
# Feature engineering
def create_feature_vectors(profiles_df):
    # Initialize BERT for name embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Encode categorical variables
    le_location = LabelEncoder()
    le_occupation = LabelEncoder()
    
    location_encoded = le_location.fit_transform(profiles_df['location'])
    occupation_encoded = le_occupation.fit_transform(profiles_df['occupation'])
    
    # Normalize age
    age_normalized = (profiles_df['age'] - profiles_df['age'].mean()) / profiles_df['age'].std()
    
    # Create name embeddings using BERT
    name_embeddings = []
    with torch.no_grad():
        for name in profiles_df['name']:
            inputs = tokenizer(name, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            name_embeddings.append(embedding.numpy())
    name_embeddings = np.vstack(name_embeddings)
    
    # Convert pandas Series to numpy arrays and reshape
    location_encoded = np.array(location_encoded).reshape(-1, 1)
    occupation_encoded = np.array(occupation_encoded).reshape(-1, 1)
    age_normalized = np.array(age_normalized).reshape(-1, 1)
    
    # Combine all features
    features = np.column_stack([
        name_embeddings,
        location_encoded,
        occupation_encoded,
        age_normalized
    ])
    
    return torch.FloatTensor(features)

# Create feature vectors
node_features = create_feature_vectors(profiles_df)
print("Feature vector shape:", node_features.shape)

# %% [markdown]
# ## 3. Graph Neural Network Model Architecture
# 
# We'll implement a GNN model that:
# 1. Uses GraphSAGE layers to learn node representations
# 2. Incorporates attention mechanisms to focus on important features
# 3. Includes a similarity function to compare profile pairs
# 4. Outputs a probability that two profiles represent the same person

# %%
# Define the GNN model
class ProfileMatchingGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super(ProfileMatchingGNN, self).__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Attention layer for comparing profile pairs
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        # Final prediction layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x, edge_index, profile_pairs):
        # Get node embeddings through GraphSAGE layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Extract embeddings for profile pairs
        profile1_embeds = x[profile_pairs[:, 0]]
        profile2_embeds = x[profile_pairs[:, 1]]
        
        # Compute attention weights
        pair_features = torch.cat([profile1_embeds, profile2_embeds], dim=1)
        attention_weights = torch.sigmoid(self.attention(pair_features))
        
        # Apply attention and concatenate
        combined_features = torch.cat([
            attention_weights * profile1_embeds,
            attention_weights * profile2_embeds
        ], dim=1)
        
        # Final prediction
        return torch.sigmoid(self.mlp(combined_features))

# Initialize the model
model = ProfileMatchingGNN(in_channels=node_features.shape[1])
print("Model initialized")

# %% [markdown]
# ## 4. Training Data Generation
# 
# We'll create training data by:
# 1. Generating positive pairs (profiles of the same person)
# 2. Generating negative pairs (profiles of different people)
# 3. Creating a balanced dataset for training
# 4. Splitting into train/validation sets

# %%
# Generate training pairs
def generate_training_pairs(true_identities, num_pairs=1000, max_attempts=10000):
    pairs = []
    labels = []
    n = len(true_identities)
    
    # Keep track of pairs to avoid duplicates
    pair_set = set()
    
    # Count actual number of duplicates for each identity
    identity_counts = {}
    for idx, identity in enumerate(true_identities):
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
    
    # Find identities that have duplicates (count > 1)
    valid_identities = [id_ for id_, count in identity_counts.items() if count > 1]
    if not valid_identities:
        raise ValueError("No duplicate identities found in the dataset")
    
    # Generate positive pairs (same identity)
    target_positive = num_pairs // 2
    positive_count = 0
    attempts = 0
    
    print("Generating positive pairs...")
    while positive_count < target_positive and attempts < max_attempts:
        # Pick an identity that has duplicates
        identity = np.random.choice(valid_identities)
        same_identity_indices = np.where(true_identities == identity)[0]
        
        if len(same_identity_indices) >= 2:
            idx1, idx2 = np.random.choice(same_identity_indices, 2, replace=False)
            pair_key = tuple(sorted([int(idx1), int(idx2)]))
            
            if pair_key not in pair_set:
                pairs.append([idx1, idx2])
                labels.append(1)
                pair_set.add(pair_key)
                positive_count += 1
        
        attempts += 1
    
    if positive_count < target_positive:
        print(f"Warning: Could only generate {positive_count} positive pairs out of {target_positive} requested")
    
    # Generate negative pairs (different identities)
    target_negative = num_pairs - positive_count
    negative_count = 0
    attempts = 0
    
    print("Generating negative pairs...")
    while negative_count < target_negative and attempts < max_attempts:
        idx1 = np.random.randint(0, n)
        identity1 = true_identities[idx1]
        different_identity_indices = np.where(true_identities != identity1)[0]
        
        if len(different_identity_indices) > 0:
            idx2 = np.random.choice(different_identity_indices)
            pair_key = tuple(sorted([int(idx1), int(idx2)]))
            
            if pair_key not in pair_set:
                pairs.append([idx1, idx2])
                labels.append(0)
                pair_set.add(pair_key)
                negative_count += 1
        
        attempts += 1
    
    if negative_count < target_negative:
        print(f"Warning: Could only generate {negative_count} negative pairs out of {target_negative} requested")
    
    # Convert to tensors
    pairs_tensor = torch.LongTensor(pairs)
    labels_tensor = torch.FloatTensor(labels)
    
    # Shuffle the data
    shuffle_idx = torch.randperm(len(pairs))
    pairs_tensor = pairs_tensor[shuffle_idx]
    labels_tensor = labels_tensor[shuffle_idx]
    
    print(f"\nFinal dataset statistics:")
    print(f"Total pairs generated: {len(pairs)}")
    print(f"Positive pairs: {positive_count} ({positive_count/len(pairs)*100:.1f}%)")
    print(f"Negative pairs: {negative_count} ({negative_count/len(pairs)*100:.1f}%)")
    
    return pairs_tensor, labels_tensor

# Generate training data
num_pairs = 1000
profile_pairs, labels = generate_training_pairs(true_identities, num_pairs)

# Split into train/validation sets
train_mask = torch.zeros(len(profile_pairs), dtype=torch.bool)
train_mask[:int(0.8 * len(profile_pairs))] = True
val_mask = ~train_mask

print("\nTrain/Validation Split:")
print("Number of training pairs:", train_mask.sum().item())
print("Number of validation pairs:", val_mask.sum().item())

# %% [markdown]
# ## 5. Model Training and Evaluation
# 
# Now we'll train the GNN model:
# 1. Define training loop with binary cross-entropy loss
# 2. Train for multiple epochs
# 3. Evaluate on validation set
# 4. Track metrics (accuracy, precision, recall, F1)

# %%
# Training function
def train_model(model, node_features, edge_index, profile_pairs, labels, train_mask, val_mask,
                num_epochs=1000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Calculate class weights for weighted BCE loss
    train_labels = labels[train_mask]
    pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    best_val_f1 = 0  # Use F1 score instead of loss for model selection
    best_model = None
    
    # Ensure all tensors have the correct shape
    labels = labels.view(-1, 1)  # Shape: [num_pairs, 1]
    train_mask = train_mask.view(-1)  # Shape: [num_pairs]
    val_mask = val_mask.view(-1)  # Shape: [num_pairs]
    
    # Create train and validation masks for indexing
    train_indices = torch.where(train_mask)[0]
    val_indices = torch.where(val_mask)[0]
    
    def calculate_metrics(preds, true_labels):
        # Calculate balanced accuracy
        pos_mask = (true_labels == 1)
        neg_mask = (true_labels == 0)
        pos_correct = (preds[pos_mask] == true_labels[pos_mask]).float().mean()
        neg_correct = (preds[neg_mask] == true_labels[neg_mask]).float().mean()
        balanced_acc = (pos_correct + neg_correct) / 2
        
        # Calculate F1 score and other metrics
        true_positives = ((preds == 1) & (true_labels == 1)).sum()
        false_positives = ((preds == 1) & (true_labels == 0)).sum()
        false_negatives = ((preds == 0) & (true_labels == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'balanced_acc': balanced_acc.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on all pairs
        out = model(node_features, edge_index, profile_pairs)  # Shape: [num_pairs, 1]
        
        # Use only training data for loss computation
        train_loss = criterion(out[train_indices], labels[train_indices])
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Use validation data for evaluation
            val_out = out[val_indices]
            val_loss = criterion(val_out, labels[val_indices])
            
            # Calculate metrics
            val_preds = (torch.sigmoid(val_out) > 0.5).float()
            val_metrics = calculate_metrics(val_preds, labels[val_indices])
            
            # Save best model based on F1 score
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}:")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Metrics - Balanced Acc: {val_metrics['balanced_acc']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
    
    # Load best model
    model.load_state_dict(best_model)
    return model

# Convert edge_index to torch tensor if not already
edge_index = torch.LongTensor(edge_index)

# Train the model
model = train_model(model, node_features, edge_index, profile_pairs, labels, train_mask, val_mask)

# Evaluate final model
model.eval()
with torch.no_grad():
    final_out = model(node_features, edge_index, profile_pairs)
    val_indices = torch.where(val_mask)[0]
    val_preds = (torch.sigmoid(final_out[val_indices]) > 0.5).float()
    val_true = labels[val_indices]
    
    # Calculate final metrics
    pos_mask = (val_true == 1).squeeze()
    neg_mask = (val_true == 0).squeeze()
    
    # Balanced accuracy
    pos_correct = (val_preds[pos_mask] == val_true[pos_mask]).float().mean()
    neg_correct = (val_preds[neg_mask] == val_true[neg_mask]).float().mean()
    balanced_acc = (pos_correct + neg_correct) / 2
    
    # Other metrics
    true_positives = ((val_preds == 1) & (val_true == 1)).sum()
    false_positives = ((val_preds == 1) & (val_true == 0)).sum()
    false_negatives = ((val_preds == 0) & (val_true == 1)).sum()
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
print("\nFinal Validation Metrics:")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print confusion matrix
true_positives = ((val_preds == 1) & (val_true == 1)).sum().item()
false_positives = ((val_preds == 1) & (val_true == 0)).sum().item()
false_negatives = ((val_preds == 0) & (val_true == 1)).sum().item()
true_negatives = ((val_preds == 0) & (val_true == 0)).sum().item()

print("\nConfusion Matrix:")
print(f"True Positives: {true_positives}, False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}, True Negatives: {true_negatives}")

# %% [markdown]
# ## 6. Inference on New Profile Pairs
# 
# Finally, we'll create a function to predict if new profile pairs represent the same person:
# 1. Process new profile features
# 2. Add to existing graph structure
# 3. Make predictions using trained model

# %%
# Function to predict on new profile pairs
def predict_profile_match(model, node_features, edge_index, profile1_idx, profile2_idx):
    model.eval()
    with torch.no_grad():
        # Create profile pair tensor
        test_pair = torch.LongTensor([[profile1_idx, profile2_idx]])
        
        # Get prediction
        pred = model(node_features, edge_index, test_pair)
        probability = pred.item()
        
        return probability, probability > 0.5

# Function to find similar profiles
def find_similar_profiles(profiles_df):
    # Find profiles with same location and occupation
    for idx1 in range(len(profiles_df)):
        profile1 = profiles_df.iloc[idx1]
        similar_profiles = profiles_df[
            (profiles_df['location'] == profile1['location']) &
            (profiles_df['occupation'] == profile1['occupation']) &
            (profiles_df.index != idx1) &
            (abs(profiles_df['age'] - profile1['age']) <= 3)  # Similar age
        ]
        
        if not similar_profiles.empty:
            idx2 = similar_profiles.index[0]
            # Make sure they are actually different people
            if true_identities[idx1] != true_identities[idx2]:
                return idx1, idx2
    
    return None, None

# Function to find verified true positive pairs
def find_true_positive_pairs(model, node_features, edge_index, profiles_df, true_identities, num_pairs=3):
    true_positive_pairs = []
    
    # Find all duplicate profiles (same identity)
    identity_groups = {}
    for idx, identity in enumerate(true_identities):
        if identity not in identity_groups:
            identity_groups[identity] = []
        identity_groups[identity].append(idx)
    
    # Filter for groups that have duplicates
    duplicate_groups = {k: v for k, v in identity_groups.items() if len(v) > 1}
    
    for identity, indices in duplicate_groups.items():
        if len(indices) >= 2:
            # Try all possible pairs in this group
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    prob, is_match = predict_profile_match(model, node_features, edge_index, idx1, idx2)
                    
                    # If model correctly predicts it's a match
                    if is_match and prob > 0.5:  # High confidence prediction
                        true_positive_pairs.append((idx1, idx2, prob))
                        if len(true_positive_pairs) >= num_pairs:
                            return true_positive_pairs
    
    return true_positive_pairs

# Test prediction on a few examples
def test_predictions():
    # Find a known matching pair (same identity)
    identity = np.random.choice(np.unique(true_identities))
    same_identity_indices = np.where(true_identities == identity)[0]
    if len(same_identity_indices) >= 2:
        idx1, idx2 = np.random.choice(same_identity_indices, 2, replace=False)
        
        # Make prediction
        prob, is_match = predict_profile_match(model, node_features, edge_index, idx1, idx2)
        
        print("\nTest prediction for known matching profiles:")
        print(f"Profile 1: {profiles_df.iloc[idx1].to_dict()}")
        print(f"Profile 2: {profiles_df.iloc[idx2].to_dict()}")
        print(f"Prediction probability: {prob:.4f}")
        print(f"Predicted match: {is_match}")
        print(f"True match: {true_identities[idx1] == true_identities[idx2]}")
    
    # Test on a known non-matching pair
    idx1 = np.random.randint(0, len(profiles_df))
    different_identity_indices = np.where(true_identities != true_identities[idx1])[0]
    idx2 = np.random.choice(different_identity_indices)
    
    # Make prediction
    prob, is_match = predict_profile_match(model, node_features, edge_index, idx1, idx2)
    
    print("\nTest prediction for known non-matching profiles:")
    print(f"Profile 1: {profiles_df.iloc[idx1].to_dict()}")
    print(f"Profile 2: {profiles_df.iloc[idx2].to_dict()}")
    print(f"Prediction probability: {prob:.4f}")
    print(f"Predicted match: {is_match}")
    print(f"True match: {true_identities[idx1] == true_identities[idx2]}")
    
    # Test on similar profiles that are different people
    similar_idx1, similar_idx2 = find_similar_profiles(profiles_df)
    if similar_idx1 is not None and similar_idx2 is not None:
        prob, is_match = predict_profile_match(model, node_features, edge_index, similar_idx1, similar_idx2)
        
        print("\nTest prediction for similar but different profiles:")
        print(f"Profile 1: {profiles_df.iloc[similar_idx1].to_dict()}")
        print(f"Profile 2: {profiles_df.iloc[similar_idx2].to_dict()}")
        print(f"Prediction probability: {prob:.4f}")
        print(f"Predicted match: {is_match}")
        print(f"True match: {true_identities[similar_idx1] == true_identities[similar_idx2]}")
        print("\nSimilarities:")
        p1, p2 = profiles_df.iloc[similar_idx1], profiles_df.iloc[similar_idx2]
        print(f"Same location: {p1['location'] == p2['location']}")
        print(f"Same occupation: {p1['occupation'] == p2['occupation']}")
        print(f"Age difference: {abs(p1['age'] - p2['age'])}")
    
    # Test on verified true positive pairs
    print("\nTesting verified true positive pairs (high confidence correct matches):")
    true_positive_pairs = find_true_positive_pairs(model, node_features, edge_index, profiles_df, true_identities)
    
    for idx1, idx2, confidence in true_positive_pairs:
        print(f"\nTrue Positive Pair (Confidence: {confidence:.4f}):")
        print(f"Profile 1: {profiles_df.iloc[idx1].to_dict()}")
        print(f"Profile 2: {profiles_df.iloc[idx2].to_dict()}")
        print(f"Shared Identity: {true_identities[idx1]}")
        
        # Show what features match
        p1, p2 = profiles_df.iloc[idx1], profiles_df.iloc[idx2]
        print("\nFeature Comparison:")
        print(f"Location match: {p1['location'] == p2['location']} ({p1['location']} vs {p2['location']})")
        print(f"Occupation match: {p1['occupation'] == p2['occupation']} ({p1['occupation']} vs {p2['occupation']})")
        print(f"Age difference: {abs(p1['age'] - p2['age'])} years")
        print(f"Name similarity: {p1['name']} vs {p2['name']}")

# Run test predictions
test_predictions()

# %%



