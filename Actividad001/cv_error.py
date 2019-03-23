# Entrega el error via CV para metodos de regresion lineal con y sin penalización, además de los coeficientes

def cv_error(x_train,y_train,k, method = 'OLS', alpha = 1):
    from sklearn.model_selection import KFold
    import sklearn.linear_model as lm
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    Xm = x_train.as_matrix()
    ym = y_train.as_matrix()
    kf = KFold(n_splits = k)
    rmse_cv = 0
    coef_v = []
    
    #OLS
    if(method == 'OLS'):
        for train, val in kf.split(Xm):
            linreg = lm.LinearRegression(fit_intercept=False)
            linreg.fit(Xm[train],ym[train])
            coef_v.append(linreg.coef_)
            yhat_val = linreg.predict(Xm[val])
            rmse_fold = np.mean(np.power(yhat_val-ym[val], 2))
            rmse_cv +=rmse_fold
        coef_T = pd.DataFrame([[fold[x] for x in range(len(x_train.columns.tolist()))] for fold in coef_v])
        coef_T.columns = x_train.columns.tolist()
        return coef_T, rmse_cv
    #Ridge
    elif(method == 'ridge'):
        for train, val in kf.split(Xm):
            ridgereg = lm.Ridge(alpha = alpha, fit_intercept=False)
            ridgereg.fit(Xm[train],ym[train])
            coef_v.append(ridgereg.coef_)
            yhat_val = ridgereg.predict(Xm[val])
            rmse_fold = np.mean(np.power(yhat_val-ym[val], 2))
            rmse_cv +=rmse_fold
        coef_T = pd.DataFrame([[fold[x] for x in range(len(x_train.columns.tolist()))] for fold in coef_v])
        coef_T.columns = x_train.columns.tolist()
        return coef_T, rmse_cv
    #Lasso
    elif(method == 'lasso'):
        for train, val in kf.split(Xm):
            lassoreg = lm.Lasso(alpha = alpha, fit_intercept=False)
            lassoreg.fit(Xm[train],ym[train])
            coef_v.append(lassoreg.coef_)
            yhat_val = lassoreg.predict(Xm[val])
            rmse_fold = np.mean(np.power(yhat_val-ym[val], 2))
            rmse_cv +=rmse_fold
        coef_T = pd.DataFrame([[fold[x] for x in range(len(x_train.columns.tolist()))] for fold in coef_v])
        coef_T.columns = x_train.columns.tolist()
        return coef_T, rmse_cv
    # Elastic Net
    elif(method == 'elastic net'):
        for train, val in kf.split(Xm):
            elasticreg = lm.ElasticNet(alpha = alpha, fit_intercept=False)
            elasticreg.fit(Xm[train],ym[train])
            coef_v.append(elasticreg.coef_)
            yhat_val = elasticreg.predict(Xm[val])
            rmse_fold = np.mean(np.power(yhat_val-ym[val], 2))
            rmse_cv +=rmse_fold
        coef_T = pd.DataFrame([[fold[x] for x in range(len(x_train.columns.tolist()))] for fold in coef_v])
        coef_T.columns = x_train.columns.tolist()
        return coef_T, rmse_cv
        
def early_stop(Xtrain, ytrain, alphas, tolerancia = 0.1, metodo = 'OLS'):
    import pandas as pd
    import numpy as np
    import sklearn.linear_model as lm
#    from cv_error import cv_error
    cv_alphas = []
    coefs_model = []
    cv_err_model = []
    if(metodo == 'OLS'):
        model = lm.LinearRegression(fit_intercept = False)
    elif(metodo == 'Ridge'):
        model = lm.Ridge(fit_intercept=False)
    elif(metodo == 'Lasso'):
        model = lm.Lasso(fit_intercept=False)
    elif(metodo == 'ElasticNet'):
        model = lm.ElasticNet(fit_intercept=False)

    tol = tolerancia # umbral de tolerancia, 0.1 parece un buen valor para empezar
    print(alphas)
    for a in alphas:
        model.set_params(alpha = a)
        model.fit(Xtrain, ytrain)
        coefs_model.append(model.coef_)
        dummy,cv_err_estimates = cv_error(Xtrain, ytrain, k = 10, method = metodo, alpha = a)
        cv_err_model.append(np.mean(cv_err_estimates)) # OJO: estamos guardando la media del error de cv para cada alpha
        cv_alphas.append(a)
        if(len(cv_err_model)>=2):
            diff_error = cv_err_model[-1] - cv_err_model[-2]
            if(diff_error > tol): # Si el error comienza a hacerse muy grande y supera a la tolerancia
                break # Terminar el ciclo
    return model, cv_err_model, cv_alphas