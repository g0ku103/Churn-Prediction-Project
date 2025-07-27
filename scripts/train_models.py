import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,roc_curve
import matplotlib.pyplot as plt

#Load preprocessed data
X_train=pd.read_csv('data/X_train.csv')
X_test=pd.read_csv('data/X_test.csv')
y_train=pd.read_csv('data/y_train.csv')
y_test=pd.read_csv('data/y_test.csv')

#initializign models
lr=LogisticRegression(random_state=42)
rf=RandomForestClassifier(random_state=42)

#Training Models
lr.fit(X_train,y_train.values.ravel())
rf.fit(X_train,y_train.values.ravel())

#prediction
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

#Evaluation model
print('Logistic Regression')
print(f'Accuracy:{accuracy_score(y_test,lr_pred):.4f}')
print(f'F1 Score : {f1_score(y_test,lr_pred):.4f}')
print(f'ROC-AUC : {roc_auc_score(y_test,lr.predict_proba(X_test)[:,1]):.4f}')

print("\nRandom Forest:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, rf_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")

#save model
with open('models/logistic_regression.pkl','wb') as f:
    pickle.dump(lr,f)
with open('models/random_forest.pkl','wb') as f:
    pickle.dump(rf,f)

# Plot ROC curves
plt.figure(figsize=(8, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]):.2f})', color='#1f77b4')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.2f})', color='#ff7f0e')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('docs/roc_curve.png')
plt.show()