# This file is here because one time I couldn't run jupyter notebook.
# I find it easier to just run this file though, so I never moved back to jupyter
from util_methods import *
from sklearn.linear_model import RidgeCV
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler






fold = 1
IMAGERY_PATH = r'/data/esa_10'



def ridge_cv(fold, dhs_vect_path):
    try:
        train_vector = np.load(os.path.join(dhs_vect_path, f"train_fold_{fold}.npy"))
        test_vector = np.load(os.path.join(dhs_vect_path, f"test_fold_{fold}.npy"))
    except:
        try:
            train_vector = np.load(os.path.join(dhs_vect_path, f"train_raw_{fold}.npy"))
            test_vector = np.load(os.path.join(dhs_vect_path, f"test_raw_{fold}.npy"))
        except:
            train_vector = np.load(os.path.join(dhs_vect_path, f"train_finetuned_{fold}.npy"))
            test_vector = np.load(os.path.join(dhs_vect_path, f"test_finetuned_{fold}.npy"))
    print(train_vector.shape, test_vector.shape)
    
    train_X = train_vector[:, :1024]
    train_y = train_vector[:, 1024]
    test_X = test_vector[:, :1024]
    test_y = test_vector[:, 1024]

    alphas = np.logspace(-6, 6, 26)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    # ridge.fit(np.concatenate([train_X, test_X]), np.concatenate([train_y, test_y]))
    ridge.fit(train_X, train_y)
    # print("Best alpha:", ridge.alpha_)
    print("Training score:", ridge.score(train_X, train_y))
    print("Test score:", ridge.score(test_X, test_y))

    train_l1 = np.mean(np.abs(ridge.predict(train_X).clip(0, 1) - train_y))
    test_l1 = np.mean(np.abs(ridge.predict(test_X).clip(0, 1) - test_y))
    print("Training L1:", train_l1)
    print("Test L1:", test_l1)
    return train_l1, test_l1


# In[ ]:
def evaluate(paths, name):
    train_losses = []
    test_losses = []
    
    print("\n\n================================")
    print(name, "\n")
    for fold, dhs_vect_path in enumerate(paths, start=2):
        print("Evaluating fold", fold)
        train_l1, test_l1 = ridge_cv(fold, dhs_vect_path)
        train_losses.append(train_l1)
        test_losses.append(test_l1)
        print("--------------------\n")

    print(name)
    print("Mean MAE:", np.mean(test_losses))
    print("Std MAE:", np.std(test_losses) / np.sqrt(len(paths)))    


# evaluate([
    "/data/output/a682cd00-4",
    "/data/output/b75a58b9-8",
    "/data/output/e57bcec7-8",
    "/data/output/e6de728a-0",
    "/data/output/567b27af-b"
], "SatMAE-L Raw")

evaluate([
    "/data/output/c20b97b6-d",
    "/data/output/c20b97b6-d",
    "/data/output/c20b97b6-d",
    "/data/output/c20b97b6-d",
    "/data/output/c20b97b6-d",
    # "/data/output/63eb1677-0",
    # "/data/output/63eb1677-0",
    # "/data/output/63eb1677-0",
    # "/data/output/63eb1677-0",
    # "/data/output/63eb1677-0",
], "SatMAE-S Raw")




# In[ ]:


# DHS_VECT_PATHS = [
#     "/data/output/e3a06201-e",
#     "/data/output/e3a06201-e",
#     "/data/output/e3a06201-e",
#     "/data/output/e3a06201-e",
#     "/data/output/e3a06201-e"
# ]    

evaluate([
    "/data/output/44525ebc-8",
    "/data/output/44525ebc-8",
    "/data/output/44525ebc-8",
    "/data/output/44525ebc-8",
    "/data/output/44525ebc-8",
], "SatMAE-L Finetuned")

evaluate([
    "/data/output/7011e81f-c",
    "/data/output/7011e81f-c",
    "/data/output/7011e81f-c",
    "/data/output/7011e81f-c",
    "/data/output/7011e81f-c",
], "SatMAE-S Finetuned")

evaluate([
    "/data/output/4848602d-9",
], "SatMAE-LT Raw")

evaluate([
    "/data/output/0fe4a385-6",
], "SatMAE-LT Finetuned")

evaluate([
    "/data/output/6b0d2db0-6",
    
], "SatMAE-ST Raw")

evaluate([
    "/data/output/077ab8b6-5",
    
], "SatMAE-ST Finetuned")




