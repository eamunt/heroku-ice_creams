# Nạp các gói thư viện cần thiết
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/Ice_cream_R.csv")
X = df.iloc[:,[1,3,4]]
y = df.ice_cream

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 390)

#Cân bằng dữ liệu bằng SMOTE
smt = SMOTE(random_state=102)
X_train, y_train = smt.fit_resample(X_train, y_train)

# Xây dựng mô hình với giải thuật KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
# Dự đoán nhãn tập kiểm tra
y_pred = model.predict(X_test)

filename = 'knn_model_icecream.pkl'
pickle.dump(model, open(filename, 'wb'))