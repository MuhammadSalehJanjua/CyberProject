import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from detector import EnhancedDetector

def evaluate_model():
    detector = EnhancedDetector()
    y_true = []
    y_pred = []
    misclassified = []
    
    # Test directory structure: test_data/real/... and test_data/fake/...
    for label in ['real', 'fake']:
        folder = os.path.join('test_data', label)
        for img_file in os.listdir(folder):
            path = os.path.join(folder, img_file)
            pred, conf = detector.predict(path)
            
            y_true.append(0 if label == 'real' else 1)
            y_pred.append(0 if pred == 'Real' else 1)
            
            if pred.lower() != label:
                misclassified.append(path)

    # Generate reports
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0,1], ['Real', 'Fake'])
    plt.yticks([0,1], ['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Save misclassified examples
    with open('misclassified.txt', 'w') as f:
        f.write('\n'.join(misclassified))

if __name__ == '__main__':
    evaluate_model()