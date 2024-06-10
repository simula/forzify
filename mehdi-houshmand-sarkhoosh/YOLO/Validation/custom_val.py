import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/mehdihou/D1/YOLO-Train/Train/classification_02/train/weights/best.pt')  # Replace with your model path
model.eval()  # Set the model to evaluation mode

# Set up the validation dataset and DataLoader
# Replace 'validation_dataset' with your actual dataset
validation_dataset = '/home/mehdihou/D1/YOLO-Train/Train/datasets/config.yaml/val'

val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)  # Set your batch size

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize confusion matrix
num_classes = len(validation_dataset.classes)  # Number of classes
confusion_matrix = torch.zeros(num_classes, num_classes)

# Inference and calculate metrics
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        # Update confusion matrix
        for t, p in zip(targets.view(-1), predictions.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

# Calculate precision, recall, and F1 score
tp = confusion_matrix.diag()
fp = confusion_matrix.sum(dim=0) - tp
fn = confusion_matrix.sum(dim=1) - tp
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

# Handle division by zero in precision and recall
precision[precision != precision] = 0  # Replaces NaN values with 0
recall[recall != recall] = 0

# Calculate overall accuracy
accuracy = tp.sum().item() / confusion_matrix.sum().item()

# Print metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision.mean().item()}')
print(f'Recall: {recall.mean().item()}')
print(f'F1 Score: {f1.mean().item()}')
