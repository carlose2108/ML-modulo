from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def class_report(y_hat, y_hat1):
    """ Función para generar las métricas:
            F1-Score
            Precision
            Recall
    """
    
    f1score = f1_score(y_hat, y_hat1, average=None)
    precision = precision_score(y_hat, y_hat1, average=None)
    recall = recall_score(y_hat, y_hat1, average=None)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.barh(['Benigno', 'Maligno'], [f1score[0], f1score[1]])
    plt.title("F1-score")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 2)
    plt.barh(['Benigno', 'Maligno'], [precision[0], precision[1]])
    plt.title("Precision")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 3)
    plt.barh(['Benigno', 'Maligno'], [recall[0], recall[1]])
    plt.title("Recall")

def overlap_percent(df):
    
    """Creación de Dataframe para expresar los porcentajes
       de no separabilidad de los atributos.
    """
    
    name_atributo = []
    percent_atributo = []
    
    for index, value in enumerate (df):
        h1, _ = np.histogram(df[df['diagnosis'] == 0][value], bins=100)
        h2, _ = np.histogram(df[df['diagnosis'] == 1][value], bins=100)
        get_minima = np.minimum(h1, h2)
        intersection = np.true_divide(np.sum(get_minima), np.sum(h2))
        name_atributo.append(value)
        percent_atributo.append(intersection)
    tmp_df = pd.DataFrame({'atributo':name_atributo, 'percent':percent_atributo}).sort_values(by='percent', ascending=False).reset_index()
    return tmp_df