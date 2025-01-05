# %% [markdown]
# ## Imports
# ---

# %%
import torch

torch.cuda.is_available()

# %%
from model.model import CNet2D
from model.dataloader import NearlabDatasetLoader, NinaproDatasetLoader
from model.utils import list_files
import scipy.io
import numpy as np
import pandas as pd

# %%
path_nearlab = "/Users/dennisschielke/Desktop/Uni/Bachelor_Thesis/src/data/nearlab/8features/person1"
version="GLVQ"
file_paths_nearlab = list_files(path_nearlab, "csv")

data = NearlabDatasetLoader(file_paths_nearlab[:2], file_paths_nearlab[2:])
X_train, y_train, X_test, y_test = data.load_data(split_method="repetition_wise")

# %%
path_ninapro = "/Users/dennisschielke/Desktop/Uni/Bachelor_Thesis/src/data/ninapro/DB2/person1"
version="GMLVQ"
file_paths_ninapro = list_files(path_ninapro, "mat")

# %% [markdown]
# # Evaluation
# ---

# %% [markdown]
# # What i want to evaluate
# 
# 1. Accuracy overview GLVQ, GMLVQ and Softmax
#     - Also keep into account which model is being used
#     - Compare to standard Machine learning methods
# 2. Compare different number of prototypes
#     - Run the algorithm for GLVQ and GMLVQ for different numbers of the prototypes and plot the difference between number and accuracy
# 3. Something with few-shot learning
# 4. Compare Confusion matrices to see where each model has its trouble
# 5. Graph of the accuracy per subject
# 6. Plot of the loss function how it decreases over each epoch
# 7. Plot of the model accracy
# 8. Table with each Layer with the following values:
#     1. Average Accuracy
#     2. Std
#     3. Median
#     4. Highest and lowest value

# %%
model = ["GLVQ", "GMLVQ"]
model_with_soft = ["Softmax", "GLVQ", "GMLVQ"]
path_results = "/Users/dennisschielke/Desktop/Uni/Bachelor_Thesis/src/results"
path_data = "/Users/dennisschielke/Desktop/Uni/Bachelor_Thesis/src/data"
prototypes_per_class = [1, 2, 3, 4, 7]  

# %% [markdown]
# # Simple Repetition Evaluation over all subjects

# %%
for i in range(0,12):
    for m in model_with_soft:
        data = NearlabDatasetLoader(file_paths_nearlab[:2], file_paths_nearlab[2:])
        X_train, y_train, X_test, y_test = data.load_data(split_method="repetition_wise")
        current_model = CNet2D(version=version, epochs=50, batch_size=32)
        history = current_model.fit(X_train, y_train)
        current_model.save_history_csv(history, f"{path_results}/person{i+1}_{m}")
        current_model.save_model_state(f"{path_results}/person{i+1}_{m}")

# %% [markdown]
# # Run each subject as train and 1 for test for all models with all combinations of subjects

# %%
for i in range(0, 12):
    file_paths_nearlab = list_files(path_data + "/person" + str(i), "csv")
    nearlab = NearlabDatasetLoader(file_paths_nearlab[:i] + file_paths_nearlab[i+1:], file_paths_nearlab[i])
    X_train, y_train, X_test, y_test = nearlab.load_data(split_method="file_split")
    for m in model_with_soft:
        current_model = CNet2D(version=m, epochs=50, batch_size=32)
        history = current_model.fit(X_train, y_train)
        current_model.save_history_csv(history, f"{path_results}/person{i+1}_{m}")
        current_model.save_model_state(f"{path_results}/person{i+1}_{m}")

# %% [markdown]
# # Run each hand orientation comb from each subject

# %%
for i in range(1,12):
    for j in range(0, 3):
        file_paths_nearlab = list_files(path_data + "/person" + str(i), "csv")
        nearlab = NearlabDatasetLoader(file_paths_nearlab[:j] + file_paths_nearlab[j+1:], file_paths_nearlab[j])
        X_train, y_train, X_test, y_test = nearlab.load_data()
        for m in model_with_soft:
            current_model = CNet2D(version=m, epochs=50, batch_size=32)
            history = current_model.fit(X_train, y_train)
            current_model.save_history_csv(history, f"{path_results}/person{i+1}_{m}")
            current_model.save_model_state(f"{path_results}/person{i+1}_{m}")


# %% [markdown]
# # Playing around with GLVQ num of prototypes

# %%
for i in range(1,12):
    for prot in prototypes_per_class:
        file_paths_nearlab = list_files(path_data + "/person" + str(i), "csv")
        nearlab = NearlabDatasetLoader(file_paths_nearlab[:2], file_paths_nearlab[2:])
        X_train, y_train, X_test, y_test = nearlab.load_data()
        for m in model:
            current_model = CNet2D(version=m, epochs=10)
            current_model.save_history_csv(history, f"{path_results}/person{i+1}_{m}")
            current_model.save_model_state(f"{path_results}/person{i+1}_{m}")

# %% [markdown]
# ## FSL Part

# %%
nearlab = NearlabDatasetLoader(file_paths_nearlab[:2], file_paths_nearlab[2:])
X_train, y_train, X_test, y_test = nearlab.load_data()
mask_train = y_train != 7
X_train_no8 = X_train[mask_train]
y_train_no8 = y_train[mask_train]

print(len(X_train_no8))
print(len(X_train))

mask_class8 = y_test == 7
X_class8 = X_test[mask_class8]
y_class8 = y_test[mask_class8]

# Train model on the first 7 classes
model = CNet2D(version="GLVQ", num_classes=7, epochs=1)
model.fit(X_train_no8, y_train_no8)


# %%
prototypes = model.classifier.get_prototypes()
print(prototypes.shape)

num_classes = model.classifier.get_num_classes()
print(num_classes)

num_prototypes = model.classifier.get_num_prototypes()
print(num_prototypes)

prototype_labels = model.classifier.get_prototype_labels()
print(prototype_labels)



# %%
model.add_new_class(X_class8, y_class8[0])

# %%
prototypes = model.classifier.get_prototypes()
print(prototypes.shape)

num_classes = model.classifier.get_num_classes()
print(num_classes)

num_prototypes = model.classifier.get_num_prototypes()
print(num_prototypes)

prototype_labels = model.classifier.get_prototype_labels()
print(prototype_labels)



# %%
model.optimize_new_prototypes(X_class8, y_class8, epochs=20)

# Evaluate on all classes
results = model.evaluate_model(X_test, y_test)

# %% [markdown]
# # RESULT - PART
# ---

# %%
model = ["Softmax", "GLVQ", "GMLVQ"]

path_results = "E:/Dennis_Bachelor/Bachelor_Thesis-main/src/results/basicEval"
path_data = "E:/Dennis_Bachelor/Bachelor_Thesis-main/src/data/nearlab/8features"

for i in range(1, 12):
    file_paths_nearlab = list_files(path_data + "/person" + str(i), "csv")
    nearlab = NearlabDatasetLoader(file_paths_nearlab[:2], file_paths_nearlab[2:])
    X_train, y_train, X_test, y_test = nearlab.load_data()
    for m in model:
        model_state_path = path_results + "/person" + str(i) + "_" + m + "/model_state.pth"

        current_model = CNet2D(version=m, epochs=50, batch_size=32)
        
        current_model.load_model_state(model_state_path)
        evaluation = current_model.evaluate_model(X_test, y_test)



