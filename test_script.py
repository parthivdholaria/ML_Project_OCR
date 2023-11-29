import pandas as pd
from PIL import Image
import numpy as np
import os
from skimage.feature import hog
from skimage import feature
from sklearn.metrics import accuracy_score
import joblib

def compute_edges(image, sigma=1.0):
    
    edge_image = feature.canny(image, sigma=sigma)
    return edge_image

def extract_hog_features(image):
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features


# f = open("demo-testing.txt", 'r')
img_names = os.listdir("testing")
test_path = [f"testing/{file_name}" for file_name in img_names]
print(test_path)

# class_folders = [folder for folder in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, folder))]

# classwise_imgs_paths = []
# for image_dir in class_folders:
#     p = [os.path.join(test_path, image_dir, filename) for filename in os.listdir(os.path.join(test_path, image_dir)) if filename.endswith(('.jpg', '.png'))]
#     classwise_imgs_paths.append(p)

data_test = []
data_test_norm = []
data_edge_test = []
tru_labels = []
# for c in classwise_imgs_paths:
for image_p in test_path:
#         print(image_p)
    img = Image.open(image_p)
    img = img.convert('L')
    img = img.resize((38, 38), Image.Resampling.LANCZOS)
    img_arr = np.array(img).flatten()
    img_arr_norm = np.array(img).flatten() / 255.0
    edge_arr = compute_edges(np.array(img))
    edge_arr_pil = Image.fromarray(edge_arr)
    edge_arr_pil = edge_arr_pil.resize((38, 38), Image.Resampling.LANCZOS)
    edge_arr = np.array(edge_arr_pil).flatten()
    # cl = image_p.split("/")[-2]
    # tru_labels.append(cl)
    data_test.append(img_arr)
    data_test_norm.append(img_arr_norm)        
    data_edge_test.append(edge_arr)

pixel_cols = [f'pixel_{i}' for i in range(np.array(data_test).shape[1])]
edge_cols = [f'edge_{i}' for i in range(np.array(data_edge_test).shape[1])]

data_df_test = pd.DataFrame(data_test, columns=pixel_cols)
data_df_norm_test = pd.DataFrame(data_test_norm, columns=pixel_cols)
test_df_edge = pd.DataFrame(data_edge_test, columns=edge_cols)
labels_df_test = pd.DataFrame(tru_labels)
test_df_edge = test_df_edge.replace({True: 1, False: 0})

hog_features_test = np.array([extract_hog_features(image.reshape(38, 38)) for image in data_df_test.values])
hog_cols = [f'hog_{i}' for i in range(hog_features_test.shape[1])]

hog_features_df_test = pd.DataFrame(hog_features_test, columns=hog_cols)

# with open('pca.pkl', 'rb') as file:
#     pca = pickle.load(file)
pca = joblib.load('pca_1350.pkl')

test_df_norm_w_hog_features_and_lbp = pd.concat([data_df_norm_test, hog_features_df_test, test_df_edge], axis=1)
pca_result_tst = pca.transform(test_df_norm_w_hog_features_and_lbp.values)
pca_df_tst = pd.DataFrame(pca_result_tst, columns=[f"pca_{i}" for i in range(pca_result_tst.shape[1])])

# with open('meta_model.pkl', 'rb') as file:
#     meta_model = joblib.load(file)

meta_model = joblib.load('meta_model.pkl')

# with open('lrcf.pkl') as file:
#     lrcf = pickle.load(file)
lrcf = joblib.load('lrcf.pkl')

# with open('svm_clm.pkl') as file:
#     svm_clf = pickle.load(file)
svm_clf = joblib.load('svm.pkl')

# with open('rf_clf.pkl') as file:
#     rf_clf = pickle.load(file)
rf_clf = joblib.load('rf.pkl')

# with open('knn.pkl') as file:
#     knn = pickle.load(file)
knn = joblib.load('knn.pkl')

# with open('mclr.pkl') as file:
#     model = pickle.load(file)

alpha_map = {'A': 10, 'B': 11, 'C': 12, 'D': 13,
 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28,
 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33,
 'Y': 34, 'Z': 35, '0': 0, '1': 1, '2': 2,
 '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
 '8': 8, '9': 9}

big_df_w_pca_feats_test = pd.concat([test_df_norm_w_hog_features_and_lbp, pca_df_tst], axis=1)
# test_predictions = model.predict(test_df_norm_w_hog_features_and_lbp.values)
# answers = np.argmax(test_predictions, axis=1)
# mapped_labels_df = pd.DataFrame()
# mapped_labels_df["label"] = labels_df_test[0].map(alpha_map)

# print(mapped_labels_df)

test_svm_predictions = svm_clf.predict_proba(test_df_norm_w_hog_features_and_lbp)
test_rf_predictions = rf_clf.predict_proba(big_df_w_pca_feats_test)
test_knn_predictions = knn.predict_proba(big_df_w_pca_feats_test)
test_lrcf_predictions = lrcf.predict_proba(test_df_norm_w_hog_features_and_lbp)
# test_mclr_predictions = model.predict(test_df_norm_w_hog_features_and_lbp)

# stacking_data_test = np.column_stack((test_svm_predictions, test_rf_predictions, test_knn_predictions, test_lrcf_predictions))
# final_predictions = meta_model.predict(stacking_data_test)

final_predictions = svm_clf.predict(test_df_norm_w_hog_features_and_lbp)

# final_predictions = lrcf.predict(test_df_norm_w_hog_features_and_lbp)

alpha_map_reverse = {value: key for key, value in alpha_map.items()}
final_predictions = [alpha_map_reverse[value] for value in final_predictions]
print(final_predictions)