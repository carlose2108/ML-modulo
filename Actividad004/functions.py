import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def plot_class_report(y_hat, y_hat1):
    """" Se grafica las principales métricas de evaluac en el algoritmo de clasificación
         multiclases como: Precision, Recall, F1-Score
         
    """
   
    
    #Guardo de manera temporal las métricas
    tmp_precision = precision_score(y_hat, y_hat1, average=None)
    tmp_recall = recall_score(y_hat, y_hat1, average=None)
    tmp_f1score = f1_score(y_hat, y_hat1, average=None)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.barh(['No Moroso', 'Moroso'], [tmp_f1score[0], tmp_f1score[1]])
    plt.title('f1-score', size=16)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.barh(['No Moroso', 'Moroso'], [tmp_precision[0], tmp_precision[1]])
    plt.title('precision', size=16)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.barh(['No Moroso', 'Moroso'], [tmp_recall[0], tmp_recall[1]])
    plt.title('recall', size=16)