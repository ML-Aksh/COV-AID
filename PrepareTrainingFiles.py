import numpy as np
import os
from IPython.display import display, HTML
import pandas as pd
from sklearn.model_selection import KFold


def make_csvs(copy_directory, split_1=81, split_2=131, split_3=211, num_folds_input=5, num_repeats=5):
    csv_storage_path = os.path.join(copy_directory, "PreparedCSV")
    try:
        os.mkdir(csv_storage_path)
    except:
        print("Error in making prepared CSV directory")

    for j in range(num_repeats):
        X_Covid = np.array(list(range(1, split_1)))
        X_Healthy = np.array(list(range(split_1, split_2)))
        X_Others = np.array(list(range(split_2, split_3)))

        num_folds = num_folds_input

        kf_Covid = KFold(n_splits=num_folds, shuffle=True)
        kf_Healthy = KFold(n_splits=num_folds, shuffle=True)
        kf_Others = KFold(n_splits=num_folds, shuffle=True)

        kf_Covid.get_n_splits(X_Covid)
        kf_Healthy.get_n_splits(X_Healthy)
        kf_Others.get_n_splits(X_Others)

        Covid_indices = []
        Healthy_indices = []
        Others_indices = []

        for train_index, test_index in kf_Covid.split(X_Covid):
            Covid_indices.append([X_Covid[train_index], X_Covid[test_index]])

        for train_index, test_index in kf_Healthy.split(X_Healthy):
            Healthy_indices.append([X_Healthy[train_index], X_Healthy[test_index]])

        for train_index, test_index in kf_Covid.split(X_Others):
            Others_indices.append([X_Others[train_index], X_Others[test_index]])

        train_image_indices = []
        test_image_indices = []

        for i in range(len(Covid_indices)):
            Covid_images = Covid_indices[i]
            Healthy_images = Healthy_indices[i]
            Others_images = Others_indices[i]

            Covid_images_train = Covid_images[0]
            Covid_images_test = Covid_images[1]

            Healthy_images_train = Healthy_images[0]
            Healthy_images_test = Healthy_images[1]

            Others_images_train = Others_images[0]
            Others_images_test = Others_images[1]

            # print(Covid_images_train)

            train_images = np.concatenate((Covid_images_train, Healthy_images_train, Others_images_train))
            test_images = np.concatenate((Covid_images_test, Healthy_images_test, Others_images_test))

            train_image_indices.append(train_images)
            test_image_indices.append(test_images)

        outline_df = pd.read_csv(os.path.join(copy_directory, 'Outline.csv'))

        for i in range(len(train_image_indices)):
            training_ids = train_image_indices[i]
            testing_ids = test_image_indices[i]
            train_df = outline_df[list(map(lambda x: x in training_ids, list(outline_df['Patient'])))]
            test_df = outline_df[list(map(lambda x: x in testing_ids, list(outline_df['Patient'])))]

            print(f"Train {train_df.shape}")
            print(f"Test: {test_df.shape}")

            current_index = num_folds * j + i
            train_df.to_csv(f"{csv_storage_path}/{current_index}-train.csv")
            test_df.to_csv(f"{csv_storage_path}/{current_index}-test.csv")

copy_directory = r"/ifs/loni/faculty/dduncan/agarg/Updated Dataset/CleanedData/"
make_csvs(copy_directory)