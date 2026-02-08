# Import libraries - Save to S3
import pandas as pd
from tqdm import tqdm
import os
from os.path import isfile, join
import boto3
import tempfile

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
# More memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
import numpy as np
import torch.nn.functional as F

import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Set GPU 0 usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set parameters (optimized for better performance)
model_name = 'xlm-roberta-base'  # Using larger model for better performance xlm-roberta-base
EPOCHS = 10  # Increased epochs
BATCH_SIZE = 8  # for large
ACCUMULATE_GRAD_BATCHES = 4  # for large
# BATCH_SIZE = 8  # Adjusted batch size for A100
# ACCUMULATE_GRAD_BATCHES = 4  # Effective batch size of 8*4=32
MAX_LENGTH = 256  # Increased sequence length
LEARNING_RATE = 1e-5  # Adjusted learning rate for large
# LEARNING_RATE = 1e-5  # Adjusted learning rate
WEIGHT_DECAY = 0.001  # L2 regularization
WARMUP_STEPS = 1000  # for large
# WARMUP_STEPS = 1000  # More warmup steps
LOSS_TYPE = 'bce'  # Loss function type: 'bce' or 'focal'
MAX_EXPLANATION_LENGTH = 128  # Maximum explanation text length

# S3 Configuration
ENDPOINT = "https://s3.pagoda.liris.cnrs.fr"
S3_BUCKET = "your_s3_bucket_name"  # Your S3 bucket name
S3_PREFIX = "your_s3_prefix"  # S3 prefix for storing models
AWS_ACCESS_KEY_ID = "your_aws_access_key_id"
AWS_SECRET_ACCESS_KEY = "your_aws_secret_access_key"

# S3 functionality implementation
def upload_to_s3(local_file_path, bucket_name, s3_key):
    """
    Upload file to S3 bucket
    
    Args:
        local_file_path: Local file path
        bucket_name: S3 bucket name
        s3_key: Object key in S3 (target path)
    
    Returns:
        bool: Whether upload was successful
        str: S3 URI if successful
    """
    try:
        # Create S3 resource
        s3 = boto3.resource(
            's3',
            endpoint_url=ENDPOINT,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Get bucket
        bucket = s3.Bucket(bucket_name)
        
        # Upload file
        print(f"Uploading model to {ENDPOINT}/{bucket_name}/{s3_key}...")
        bucket.upload_file(local_file_path, s3_key)
        print(f"Successfully uploaded model to {ENDPOINT}/{bucket_name}/{s3_key}")
        
        # Build complete S3 URI
        s3_uri = f"{ENDPOINT}/{bucket_name}/{s3_key}"
        return True, s3_uri
    except Exception as e:
        print(f"Failed to upload to S3: {str(e)}")
        return False, None

def download_from_s3(bucket_name, s3_key, local_file_path):
    """
    Download file from S3
    
    Args:
        bucket_name: S3 bucket name
        s3_key: S3 object key
        local_file_path: Local save path
    
    Returns:
        bool: Whether download was successful
    """
    try:
        # Create S3 resource
        s3 = boto3.resource(
            's3',
            endpoint_url=ENDPOINT,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download file
        print(f"Downloading file from {ENDPOINT}/{bucket_name}/{s3_key}...")
        s3.Bucket(bucket_name).download_file(s3_key, local_file_path)
        print(f"Successfully downloaded file from {ENDPOINT}/{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        print(f"Failed to download from S3: {str(e)}")
        return False

# Create S3 model save callback
class S3ModelCheckpoint(pl.Callback):
    """Custom callback for saving models to S3 instead of local disk"""
    
    def __init__(self, bucket_name, s3_prefix, model_name, monitor='val_f1_micro', mode='max'):
        super().__init__()
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.model_name = model_name
        self.monitor = monitor
        self.mode = mode
        self.best_model_score = None
        self.best_model_path = None
        self.temp_dir = tempfile.mkdtemp()
        
        # Set comparison function based on mode
        self.compare = lambda x, y: x > y if self.mode == 'max' else x < y

    def on_validation_end(self, trainer, pl_module):
        # Get current metric
        current_score = trainer.callback_metrics.get(self.monitor)
        
        if current_score is None:
            return
        
        # Convert tensor to Python scalar if needed
        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()
            
        # Check if this is the best model
        if self.best_model_score is None or self.compare(current_score, self.best_model_score):
            # Update best score
            self.best_model_score = current_score
            
            # Create temporary file to save model
            filename = f"{self.model_name.split('/')[-1]}_{EPOCHS}_{LOSS_TYPE}_1e_explanation_enhanced.ckpt"
            temp_path = os.path.join(self.temp_dir, filename)
            
            # Save model to temporary file
            trainer.save_checkpoint(temp_path)
            
            # Build S3 key
            s3_key = f"{self.s3_prefix}/{filename}"
            
            # Upload to S3
            success, s3_uri = upload_to_s3(temp_path, self.bucket_name, s3_key)
            
            if success:
                # Save best model path
                self.best_model_path = s3_uri
                print(f"New best model saved to: {s3_uri} (score: {current_score:.4f})")
                
                # Delete temporary file
                os.remove(temp_path)
            else:
                print("Unable to upload best model to S3")
    
    def on_train_end(self, trainer, pl_module):
        # Clean up temporary directory
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

# Useful settings
torch.manual_seed(42)  # Changed seed for better reproducibility
# Check if CUDA is available before setting device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")
    # Free up cache at start
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print(f"CUDA not available, using CPU")

# Data handling functions remain the same
def make_dataframe(data_type='train'):
    # MAKE TXT DATAFRAME
    if data_type == 'train':
        input_folder = 'your_train_articles_folder/'
        labels_fn = 'your_train_labels_file.txt'
    elif data_type == 'dev':
        input_folder = 'your_dev_articles_folder/'
        labels_fn = 'your_dev_labels_file.txt'
    else:
        raise ValueError("data_type must be 'train' or 'dev'")
    
    # Text data processing section remains unchanged
    text = []
    skipped_files = 0
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD = fil.split('.')[0]
        try:
            with open(input_folder+fil, 'r', encoding='utf-8') as f:
                content = f.read()
            lines = list(enumerate(content.splitlines(), 1))
            text.extend([(iD,) + line for line in lines])
        except UnicodeDecodeError as e:
            skipped_files += 1
            print(f"File decoding failed: {fil} - Skipping this file")
            continue
    print(f"Total skipped {skipped_files} files that couldn't be decoded")

    df_text = pd.DataFrame(text, columns=['id','line','text'])
    df_text.line = df_text.line.apply(int)
    df_text = df_text[df_text.text.str.strip().str.len() > 0].copy()
    df_text = df_text.set_index(['id','line'])

    # Modified label file processing section
    try:
        # Check if file exists
        if not os.path.exists(labels_fn):
            print(f"Label file does not exist: {labels_fn}")
            raise FileNotFoundError(f"Label file does not exist: {labels_fn}")
            
        # Manually parse label file
        labels_data = []
        with open(labels_fn, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split('\t')
                if len(parts) < 2:  # Try splitting by space
                    parts = line.split()
                    
                if len(parts) >= 2:
                    id_val = parts[0]
                    try:
                        line_val = int(parts[1])
                        # If third column exists, it's the label value; otherwise empty string
                        label_val = parts[2] if len(parts) >= 3 else ''
                        labels_data.append([id_val, line_val, label_val])
                    except ValueError:
                        print(f"Skipping line {line_num} with invalid line number: {line}")
                else:
                    print(f"Skipping incorrectly formatted line {line_num}: {line}")
        
        # Convert to DataFrame
        if labels_data:
            labels = pd.DataFrame(labels_data, columns=['id', 'line', 'labels'])
            # Set line column as integer type
            labels['line'] = labels['line'].astype(int)
            # Set index
            labels = labels.set_index(['id','line'])
            
            # JOIN - Use right join to ensure all text is included
            df = df_text.join(labels, how='left')[['text','labels']]
            # Fill empty string for rows without labels
            df['labels'] = df['labels'].fillna('')
        else:
            raise ValueError("Label file is empty after parsing")
                
    except Exception as e:
        print(f"Error processing label file: {str(e)}")
        print("Creating dataframe without labels...")
        # If label file cannot be processed, create dataframe without labels
        df = df_text.copy()
        df['labels'] = ''
        df = df[['text', 'labels']]
        
    # Add text cleaning step
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    return df.reset_index()
    
# Load label classes
def load_label_classes():
    classes_file = "techniques.txt"
    
    # If file doesn't exist, use example label list
    if not os.path.exists(classes_file):
        print(f"Warning: Cannot find label file {classes_file}, using example label list")
        # Example labels, replace with actual label list if file doesn't exist
        return ["Appeal_to_Authority", "Appeal_to_Fear", "Appeal_to_Prejudice", 
                "Bandwagon", "Black_and_White_Fallacy", "Causal_Oversimplification",
                "Doubt", "Exaggeration", "Flag_Waving", "Loaded_Language",
                "Name_Calling", "Reductio_ad_Hitlerum", "Repetition", "Slogans",
                "Straw_Man", "Thought_Terminating_Cliches", "Whataboutism"]
    
    # File exists, read normally
    labels_name = []
    with open(classes_file, "r") as f:
        for line in f.readlines():
            labels_name.append(line.rstrip())
    labels_name.sort() 
    return labels_name

# Load explanation data
def load_explanations_data(explanations_file):
    """
    Load LLM-generated explanation data
    
    Args:
        explanations_file: Path to explanation data file, TSV format
    
    Returns:
        dict: Dictionary of explanation data with (id, text) as keys
    """
    explanations = {}
    try:
        df = pd.read_csv(explanations_file, sep='\t')
        for _, row in df.iterrows():
            # Use (id, text) tuple as key
            key = (row['id'], row['text'])
            explanations[key] = row['analysis']
    except Exception as e:
        print(f"Error loading explanation data: {str(e)}")
        explanations = {}
    
    print(f"Successfully loaded {len(explanations)} explanation entries")
    return explanations

# Modified - Create multi-label dataset with explanation data
def load_multi_label_data_with_explanations(data_type='train', explanations=None):
    df = make_dataframe(data_type=data_type)
    
    # Add data cleaning step - remove overly long text
    max_text_len = 1000
    df = df[df['text'].str.len() <= max_text_len].copy()
    
    all_idxs = df["id"].to_numpy()
    all_lines = df["line"].to_numpy()
    all_data = df["text"].to_numpy()
    
    # Load all labels
    labels_name = load_label_classes()
    num_labels = len(labels_name)
    
    # Create multi-label matrix - each sample corresponds to a vector, each element in vector corresponds to whether a label exists
    multi_labels = []
    for label_str in df['labels'].fillna('').values:
        label_vec = np.zeros(num_labels, dtype=np.float32)
        if label_str:
            labels = label_str.split(',')
            for label in labels:
                if label in labels_name:
                    label_idx = labels_name.index(label)
                    label_vec[label_idx] = 1.0
        multi_labels.append(label_vec)
    
    # Add explanation data
    explanation_texts = []
    if explanations:
        for idx, text in zip(all_idxs, all_data):
            key = (idx, text)
            if key in explanations:
                explanation_texts.append(explanations[key])
            else:
                explanation_texts.append("")  # Use empty string for missing explanations
    else:
        explanation_texts = [""] * len(all_data)  # Use empty strings for all
        
    all_idxs_array = np.array(all_idxs, dtype=object)  # Use object dtype to handle string IDs
    all_lines_array = np.array(all_lines, dtype=np.int32)
    
    return all_idxs_array, all_lines_array, all_data, torch.tensor(np.array(multi_labels)), explanation_texts, labels_name

# Multi-label classification dataset with explanations
class MultiLabelClassificationDataWithExplanations(torch.utils.data.Dataset):
    def __init__(self, tokenizer=None, max_length=MAX_LENGTH, 
                 max_explanation_length=MAX_EXPLANATION_LENGTH, 
                 data_tuple=None, data_type='train'):
        
        self.max_length = max_length
        self.max_explanation_length = max_explanation_length
        
        if data_tuple is None:
            self.idxs, self.lines, X, self.y, self.explanations, self.label_names = load_multi_label_data_with_explanations(data_type)
        else:
            self.idxs, self.lines, X, self.y, self.explanations, self.label_names = data_tuple
            
        self.tokenized = False
        
        if tokenizer is not None:
            self.tokenized = True
            # Text tokenization processing
            batch_size = 256  # Increase batch size
            self.input_ids = []
            self.attention_mask = []
            
            # Combine text and explanation into single input
            combined_texts = []
            for i in range(len(X)):
                # Only combine when explanation is not empty
                if self.explanations[i].strip():
                    combined_texts.append(f"{X[i]} [SEP] {self.explanations[i]}")
                else:
                    combined_texts.append(X[i])
            
            for i in range(0, len(combined_texts), batch_size):
                batch = combined_texts[i:i+batch_size]
                tokenized = tokenizer(
                    batch, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                self.input_ids.append(tokenized["input_ids"])
                self.attention_mask.append(tokenized["attention_mask"])
                
                # Clean cache
                if i % 5000 == 0 and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
            self.input_ids = torch.cat(self.input_ids, dim=0)
            self.attention_mask = torch.cat(self.attention_mask, dim=0)
        else:
            self.X = X
            self.explanations = self.explanations

    def __getitem__(self, index):
        sample = self.input_ids[index]
        mask = self.attention_mask[index]
        a = self.idxs[index]
        b = torch.tensor(self.lines[index], dtype=torch.int64)
        label = torch.squeeze(self.y[index])
        return sample, mask, label, a, b

    def __len__(self):
        if self.tokenized:
            return self.input_ids.shape[0]
        else:
            return len(self.X)

# Add focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Add epsilon to prevent extreme values
        epsilon = 1e-7
        targets_stable = torch.clamp(targets.float(), min=epsilon, max=1-epsilon)
        
        # Calculate basic BCE loss
        if self.pos_weight is not None:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets_stable, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets_stable, reduction='none'
            )
        
        # Calculate prediction probability to adjust focal loss
        pt = torch.exp(-BCE_loss)  # pt = p if y=1, pt = 1-p if y=0
        
        # Focal loss calculation
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        # Return result based on reduction parameter
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Modified - Multi-label classification model integrating explanation data
class MultiLabelClassifierWithExplanations(pl.LightningModule):
    def __init__(self, plm, num_labels, class_weights=None, learning_rate=LEARNING_RATE, 
                 warmup_steps=WARMUP_STEPS, loss_type='bce', focal_gamma=2.0, focal_alpha=1.0):
        
        super().__init__()
        self.plm = plm
        self.num_labels = num_labels
        
        # Enable gradient checkpointing
        if hasattr(self.plm, "gradient_checkpointing_enable"):
            self.plm.gradient_checkpointing_enable()
            
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.loss_type = loss_type
        
        # Choose loss function
        if loss_type == 'focal':
            # Use focal loss
            self.criterion = FocalLoss(
                alpha=focal_alpha, 
                gamma=focal_gamma, 
                pos_weight=class_weights
            )
        else:  # Default to BCE
            # Use standard BCEWithLogitsLoss
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
          
        # Add evaluation metrics
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
    
    def forward(self, samples, masks):
        x = self.plm(samples, masks)
        logits = x.logits  # Multi-label output

        # Add safety check to limit extreme values
        return torch.clamp(logits, min=-10.0, max=10.0)

    def training_step(self, batch, batch_idx):
        batch_ids, batch_mask, labels, _, _ = batch
        preds = self(samples=batch_ids, masks=batch_mask)
        
        # Ensure labels and preds shapes match
        if labels.shape != preds.shape:
            if len(labels.shape) < len(preds.shape):
                labels = labels.view(*preds.shape)
            else:
                preds = preds.view(*labels.shape)
        
        # Stabilization processing
        epsilon = 1e-7
        labels_stable = torch.clamp(labels.float(), min=epsilon, max=1-epsilon)
        
        # Calculate loss
        loss = self.criterion(preds, labels_stable)
        
        # Handle NaN loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss occurred in batch {batch_idx}!")
            return {"loss": torch.tensor(1e-5, requires_grad=True, device=loss.device)}
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Safely collect predictions
        with torch.no_grad():
            pred_probs = torch.sigmoid(preds)
            # Filter out possible NaN values
            valid_mask = ~torch.isnan(pred_probs)
            if valid_mask.all():
                self.train_preds.extend(pred_probs.detach().cpu().numpy())
                self.train_labels.extend(labels.detach().cpu().numpy())
        
        # Clean cache more frequently
        if batch_idx % 50 == 0 and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        return {"loss": loss}
        
    def validation_step(self, batch, batch_idx):
        batch_ids, batch_mask, labels, _, _ = batch
        
        # Validate input tensors
        if torch.isnan(batch_ids).any() or torch.isnan(batch_mask).any() or torch.isnan(labels).any():
            print(f"Warning: Validation batch {batch_idx} input data contains NaN, skipping this batch")
            return {"val_loss": torch.tensor(0.0, device=self.device)}
        
        preds = self(samples=batch_ids, masks=batch_mask)
        
        # Ensure labels and preds shapes match
        if labels.shape != preds.shape:
            if len(labels.shape) < len(preds.shape):
                labels = labels.view(*preds.shape)
            else:
                preds = preds.view(*labels.shape)
        
        # Add epsilon to prevent extreme values
        epsilon = 1e-7
        labels_stable = torch.clamp(labels.float(), min=epsilon, max=1-epsilon)
        
        # Calculate loss
        loss = self.criterion(preds, labels_stable)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss occurred in validation batch {batch_idx}!")
            return {"val_loss": torch.tensor(0.0, device=self.device)}
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        # Multi-label evaluation
        with torch.no_grad():
            pred_probs = torch.sigmoid(preds)
            pred_labels = (pred_probs > 0.5).float()  # confidence
            
            # Calculate sample-level accuracy (exact match)
            exact_match = (pred_labels == labels).all(dim=1).float().mean()
            self.log('val_exact_match', exact_match, prog_bar=True)
            
            # Safely collect predictions
            valid_mask = ~torch.isnan(pred_probs)
            if valid_mask.all():
                self.val_preds.extend(pred_probs.detach().cpu().numpy())
                self.val_labels.extend(labels.detach().cpu().numpy())
        
        return {"val_loss": loss}

    def on_train_epoch_end(self):
        # Calculate multi-label metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Convert to binary classification labels
        pred_labels = (np.array(self.train_preds) > 0.5).astype(int)
        true_labels = np.array(self.train_labels)
        
        # Calculate metrics - micro-average and macro-average
        precision_micro = precision_score(true_labels, pred_labels, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels, pred_labels, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
        
        precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        
        # Log metrics
        self.log('train_precision_micro', precision_micro)
        self.log('train_recall_micro', recall_micro)
        self.log('train_f1_micro', f1_micro)
        self.log('train_precision_macro', precision_macro)
        self.log('train_recall_macro', recall_macro)
        self.log('train_f1_macro', f1_macro)
        
        # Clear collected predictions
        self.train_preds = []
        self.train_labels = []
        
    def on_validation_epoch_end(self):
        # Calculate multi-label metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Convert to binary classification labels
        pred_labels = (np.array(self.val_preds) > 0.5).astype(int)
        true_labels = np.array(self.val_labels)
        
        # Calculate metrics - micro-average and macro-average
        precision_micro = precision_score(true_labels, pred_labels, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels, pred_labels, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
        
        precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        
        # Log metrics
        self.log('val_precision_micro', precision_micro, prog_bar=True)
        self.log('val_recall_micro', recall_micro, prog_bar=True)
        self.log('val_f1_micro', f1_micro, prog_bar=True)
        self.log('val_precision_macro', precision_macro)
        self.log('val_recall_macro', recall_macro)
        self.log('val_f1_macro', f1_macro)
        
        
        # Calculate metrics for each label separately
        if self.val_preds and len(self.val_preds) > 0:
            for i in range(self.num_labels):
                label_f1 = f1_score(true_labels[:, i], pred_labels[:, i], zero_division=0)
                # Only log non-zero F1 values
                if label_f1 > 0:
                    self.log(f'val_f1_label_{i}', label_f1)
        
        # Clear collected predictions
        self.val_preds = []
        self.val_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=WEIGHT_DECAY
        )
        
        # Use Lightning's method to estimate total steps
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_f1_micro",
            },
        }

# Calculate multi-label class weights
def compute_multi_label_class_weights(labels, neg_pos_ratio=3.0, max_weight=30.0):
    """
    Calculate class weights for multi-label classification with weight upper limit
    
    Parameters:
    labels (torch.Tensor): Binary label tensor of shape [samples, classes]
    neg_pos_ratio (float): Negative-positive sample ratio coefficient
    max_weight (float): Weight upper limit
    
    Returns:
    torch.Tensor: Weight for each class
    """
    # Ensure labels are binary
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Labels should be binary (0 or 1)")
    
    # Calculate positive sample count for each label
    pos_counts = labels.sum(dim=0)
    # Total samples
    total_samples = labels.shape[0]
    # Calculate negative sample count
    neg_counts = total_samples - pos_counts
    
    # Calculate weight for each label
    weights = []
    for i in range(labels.shape[1]):
        if pos_counts[i] > 0:
            # Positive sample weight = negative samples / positive samples * ratio
            weight = min((neg_counts[i] / pos_counts[i]) * neg_pos_ratio, max_weight)
            weights.append(weight)
        else:
            # If no positive samples, set to default weight
            weights.append(1.0)
    
    # Create weight tensor and ensure it's on same device as input
    return torch.tensor(weights, dtype=torch.float32, device=labels.device)

# Main training function: Train explanation-enhanced multi-label classification model
def train_explanation_enhanced_model(explanations_file=None, loss_type=LOSS_TYPE, 
                                    focal_gamma=2.0, focal_alpha=1.0):
    print("Loading label classes...")
    labels_name = load_label_classes()
    num_labels = len(labels_name)
    print(f"Total of {num_labels} label classes")
    
    print(f"Using loss function: {loss_type}")
    if loss_type == 'focal':
        print(f"Focal Loss parameters: gamma={focal_gamma}, alpha={focal_alpha}")
    
    # Load explanation data
    explanations = None
    if explanations_file:
        print(f"Loading explanation data file: {explanations_file}")
        explanations = load_explanations_data(explanations_file)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Create temporary directory for logs
    temp_log_dir = tempfile.mkdtemp(prefix="lightning_logs_")
    print(f"Created temporary log directory: {temp_log_dir}")
    
    # Load training data (with explanations)
    print("Loading training data...")
    idxs, lines, X, y, explanations_train, _ = load_multi_label_data_with_explanations(
        'train', explanations=explanations
    )
    
    # Load validation data (with explanations)
    print("Loading validation data...")
    dev_idxs, dev_lines, dev_X, dev_y, explanations_dev, _ = load_multi_label_data_with_explanations(
        'dev', explanations=explanations
    )
    print(f"Validation data loading complete, contains {len(dev_X)} samples")
    
    # Calculate class weights
    y_train = y.numpy()
    pos_counts = y_train.sum(axis=0)
    total_samples = y_train.shape[0]
    neg_counts = total_samples - pos_counts
    
    # Output positive sample count and proportion for each label
    print("Positive sample statistics for each label:")
    for i, label in enumerate(labels_name):
        if pos_counts[i] > 0:
            print(f"Label {i}: {label} - Positive samples: {pos_counts[i]}, Proportion: {pos_counts[i]/total_samples:.4f}")
    
    # Calculate multi-label class weights
    class_weights = compute_multi_label_class_weights(y)
    class_weights = class_weights.to(device)
    print(f"Class weight calculation complete")
    
    # Create training and validation sets
    dataset_train = MultiLabelClassificationDataWithExplanations(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_explanation_length=MAX_EXPLANATION_LENGTH,
        data_tuple=(idxs, lines, X, y, explanations_train, labels_name),
        data_type='train'
    )
    
    dataset_val = MultiLabelClassificationDataWithExplanations(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_explanation_length=MAX_EXPLANATION_LENGTH,
        data_tuple=(dev_idxs, dev_lines, dev_X, dev_y, explanations_dev, labels_name),
        data_type='dev'
    )
    
    # Create data loaders
    train_sampler = RandomSampler(dataset_train)
    train_loader = DataLoader(
        dataset_train, 
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE * 2,
        num_workers=2,
        pin_memory=True
    )
    
    # Create multi-label classification model
    config = AutoConfig.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config
    )
    
    model = MultiLabelClassifierWithExplanations(
        classification_model,
        num_labels=num_labels,
        class_weights=class_weights,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha
    )
    
    # Custom S3 model save callback
    s3_checkpoint_callback = S3ModelCheckpoint(
        bucket_name=S3_BUCKET,
        s3_prefix=f"{S3_PREFIX}/explanation_enhanced",
        model_name=model_name,
        monitor='val_f1_micro',
        mode='max'
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_f1_micro',
        patience=3,
        verbose=True,
        mode='max'
    )
    
    # Check if TensorBoard is available
    try:
        # Try importing tensorboard or tensorboardX
        try:
            import tensorboard
            has_tensorboard = True
        except ImportError:
            try:
                import tensorboardX
                has_tensorboard = True
            except ImportError:
                has_tensorboard = False
        
        if has_tensorboard:
            # TensorBoard logger
            logger = TensorBoardLogger(
                save_dir=temp_log_dir,
                name='explanation_enhanced_model'
            )
        else:
            print("Warning: tensorboard or tensorboardX not found, using CSV logger instead")
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger(
                save_dir=temp_log_dir,
                name='explanation_enhanced_model'
            )
    except Exception as e:
        print(f"Failed to create logger: {str(e)}, will use no-logging mode")
        logger = None
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[s3_checkpoint_callback, early_stop_callback],
        precision="32",  # Changed to 32-bit full precision instead of 16-bit mixed precision
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=0.5,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        logger=logger
    )
    
    # Train model
    try:
        model = model.to(device)
        trainer.fit(model, train_loader, val_loader)
        
        # Get best model path on S3
        best_model_path = s3_checkpoint_callback.best_model_path
        best_model_score = s3_checkpoint_callback.best_model_score
        
        if best_model_path:
            print(f"Best model saved on S3: {best_model_path}")
            print(f"Best F1 micro-average score: {best_model_score:.4f}")
            
            # Record evaluation metrics
            eval_results = {
                'model_type': 'explanation_enhanced',
                'train_samples': len(dataset_train),
                'dev_samples': len(dataset_val),
                'best_val_f1_micro': best_model_score,
                's3_model_path': best_model_path
            }
            
            # Save evaluation results to S3
            eval_df = pd.DataFrame([eval_results])
            
            # Create temporary CSV file
            temp_results_path = os.path.join(tempfile.mkdtemp(), "explanation_enhanced_evaluation_results.csv")
            eval_df.to_csv(temp_results_path, index=False)
            
            # Upload to S3
            s3_key = f"{S3_PREFIX}/explanation_enhanced/evaluation_results.csv"
            success, s3_uri = upload_to_s3(temp_results_path, S3_BUCKET, s3_key)
            
            if success:
                print(f"Evaluation results saved to: {s3_uri}")
                # Delete temporary file
                os.remove(temp_results_path)
        else:
            # If no best model, try saving final model to S3
            print(f"No best model, attempting to save final model...")
            # Create temporary file
            temp_final_path = os.path.join(tempfile.mkdtemp(), "explanation_enhanced_model_final.ckpt")
            trainer.save_checkpoint(temp_final_path)
            
            # Upload to S3
            s3_key = f"{S3_PREFIX}/explanation_enhanced/explanation_enhanced_model_final.ckpt"
            success, s3_uri = upload_to_s3(temp_final_path, S3_BUCKET, s3_key)
            
            if success:
                print(f"Final model saved to: {s3_uri}")
                # Delete temporary file
                os.remove(temp_final_path)
    except Exception as e:
        print(f"Training failed: {str(e)}")
        print("Attempting to save partial model to S3...")
        try:
            # Create temporary file to save partial model
            temp_partial_path = os.path.join(tempfile.mkdtemp(), "explanation_enhanced_model_partial.ckpt")
            trainer.save_checkpoint(temp_partial_path)
            
            # Upload to S3
            s3_key = f"{S3_PREFIX}/explanation_enhanced/explanation_enhanced_model_partial.ckpt"
            success, s3_uri = upload_to_s3(temp_partial_path, S3_BUCKET, s3_key)
            
            if success:
                print(f"Partial model saved to: {s3_uri}")
                # Delete temporary file
                os.remove(temp_partial_path)
            else:
                print("Unable to upload partial model to S3")
        except Exception as save_err:
            print(f"Failed to save partial model: {str(save_err)}")
    finally:
        # Clean up GPU memory
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # Release model memory
        del model
        del classification_model
        del train_loader
        del val_loader
        del dataset_train
        del dataset_val
        import gc
        gc.collect()
    
    # Clean up temporary log directory
    try:
        import shutil
        shutil.rmtree(temp_log_dir)
    except:
        pass
    
    print("\nExplanation-enhanced multi-label classification model training complete and saved to S3!")

# Prediction function - Use model for prediction
def predict_with_explanation_model(texts, explanations=None, s3_model_uri=None, 
                                tokenizer_name=model_name, threshold=0.5):
    """
    Use explanation-enhanced multi-label model for prediction
    
    Args:
        texts (list): List of texts to predict
        explanations (list, optional): List of text explanations
        s3_model_uri (str): Model URI on S3
        tokenizer_name (str): Tokenizer name
        threshold (float): Prediction threshold, default 0.5
    
    Returns:
        list: List of prediction results
    """
    # Download model
    local_model_path = os.path.join(tempfile.mkdtemp(), "model.ckpt")
    # Parse S3 URI
    s3_parts = s3_model_uri.replace(ENDPOINT, '').strip('/').split('/', 1)
    bucket_name = s3_parts[0]
    s3_key = s3_parts[1]
    
    success = download_from_s3(bucket_name, s3_key, local_model_path)
    if not success:
        raise Exception(f"Unable to download model from S3: {s3_model_uri}")
    
    # Load model
    model = MultiLabelClassifierWithExplanations.load_from_checkpoint(local_model_path)
    model.eval()
    model = model.to(device)
    
    # Load label names
    labels_name = load_label_classes()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process text and explanations
    combined_texts = []
    for i, text in enumerate(texts):
        # Combine text and explanation
        if explanations and i < len(explanations) and explanations[i].strip():
            combined_texts.append(f"{text} [SEP] {explanations[i]}")
        else:
            combined_texts.append(text)
    
    # Encode text
    inputs = tokenizer(
        combined_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.sigmoid(logits)
    
    # Convert to prediction results
    probs_numpy = probs.cpu().numpy()
    pred_labels = (probs_numpy > threshold).astype(int)
    
    # Prepare results
    results = []
    for i, text in enumerate(texts):
        text_result = {
            "text": text,
            "explanation": explanations[i] if explanations and i < len(explanations) else None,
            "probabilities": {labels_name[j]: float(probs_numpy[i, j]) for j in range(len(labels_name))},
            "predicted_labels": [labels_name[j] for j in range(len(labels_name)) if pred_labels[i, j] == 1]
        }
        results.append(text_result)
    
    # Clean up
    if os.path.exists(local_model_path):
        os.remove(local_model_path)
    del model
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    return results

# Main execution function
if __name__ == "__main__":
    # Detect if running in Jupyter environment
    try:
        # IPython-specific variable, exists if running in Jupyter
        ipy_str = str(type(get_ipython()))
        is_jupyter = 'zmqshell' in ipy_str
    except:
        is_jupyter = False
        
    if is_jupyter:
        # Running in Jupyter, use default parameters
        print("Running in Jupyter environment, using default parameters")
        print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Training epochs: {EPOCHS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Model name: {model_name}")
        print(f"Max sequence length: {MAX_LENGTH}")
        print(f"S3 bucket: {S3_BUCKET}")
        print(f"S3 prefix: {S3_PREFIX}")
        
        # Set explanation data file path
        explanations_file = "your_explanations_file.tsv"
        print(f"Explanation data file: {explanations_file}")
        print("Starting explanation-enhanced multi-label classification model training...")
        
        # Set loss function type
        loss_type = LOSS_TYPE
        focal_gamma = 2.0
        focal_alpha = 1.0
        
        print(f"Using loss function: {loss_type}")
        if loss_type == 'focal':
            print(f"Focal Loss parameters: gamma={focal_gamma}, alpha={focal_alpha}")
        
        # Execute training
        train_explanation_enhanced_model(
            explanations_file=explanations_file,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha
        )
    else:
        # Running in regular Python environment, use argparse
        import argparse
        
        # Create command line argument parser
        parser = argparse.ArgumentParser(description='Train explanation-enhanced multi-label classification model for propaganda technique detection and save to S3')
        parser.add_argument('--gpu', type=int, help='GPU ID to use', default=0)
        parser.add_argument('--batch_size', type=int, help='Batch size', default=BATCH_SIZE)
        parser.add_argument('--epochs', type=int, help='Training epochs', default=EPOCHS)
        parser.add_argument('--learning_rate', type=float, help='Learning rate', default=LEARNING_RATE)
        parser.add_argument('--model', type=str, help='Model name', default=model_name)
        parser.add_argument('--max_length', type=int, help='Maximum sequence length', default=MAX_LENGTH)
        parser.add_argument('--s3_bucket', type=str, help='S3 bucket name', default=S3_BUCKET)
        parser.add_argument('--s3_prefix', type=str, help='S3 prefix path', default=S3_PREFIX)
        parser.add_argument('--explanations_file', type=str, help='Explanation data file path', required=True)
        parser.add_argument('--loss_type', type=str, choices=['bce', 'focal'], 
                           help='Loss function type: bce (binary cross entropy) or focal (focal loss)', default=LOSS_TYPE)
        parser.add_argument('--focal_gamma', type=float, help='Gamma parameter for Focal Loss', default=2.0)
        parser.add_argument('--focal_alpha', type=float, help='Alpha parameter for Focal Loss', default=1.0)
        
        # Parse command line arguments
        args = parser.parse_args()
        
        # Update global parameters
        if args.gpu != 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"Using GPU {args.gpu}")
        
        if args.batch_size != BATCH_SIZE:
            globals()['BATCH_SIZE'] = args.batch_size
            print(f"Batch size set to {args.batch_size}")
        
        if args.epochs != EPOCHS:
            globals()['EPOCHS'] = args.epochs
            print(f"Training epochs set to {args.epochs}")
            
        if args.learning_rate != LEARNING_RATE:
            globals()['LEARNING_RATE'] = args.learning_rate
            print(f"Learning rate set to {args.learning_rate}")
            
        if args.model != model_name:
            globals()['model_name'] = args.model
            print(f"Model name set to {args.model}")
            
        if args.max_length != MAX_LENGTH:
            globals()['MAX_LENGTH'] = args.max_length
            print(f"Maximum sequence length set to {args.max_length}")
            
        if args.s3_bucket != S3_BUCKET:
            globals()['S3_BUCKET'] = args.s3_bucket
            print(f"S3 bucket set to {args.s3_bucket}")
            
        if args.s3_prefix != S3_PREFIX:
            globals()['S3_PREFIX'] = args.s3_prefix
            print(f"S3 prefix set to {args.s3_prefix}")
                    
        # Print loss function information
        print(f"Using loss function: {args.loss_type}")
        if args.loss_type == 'focal':
            print(f"Focal Loss parameters: gamma={args.focal_gamma}, alpha={args.focal_alpha}")
        
        # Execute training
        train_explanation_enhanced_model(
            explanations_file=args.explanations_file,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha
        )
