import os
import numpy as np
import pandas as pd
import re

def extract_label(filename):
    """
    Extract class label from filename.
    Example:
        walk1_1.csv → 1
        sit4_0.csv → 0
    """
    match = re.search(r'_([0-9]+)\.csv$', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Label not found in {filename}")


def create_dataset(
        folder_path,
        window_size=50,
        step_size=50,     # use 25 for overlapping windows
        output_file="dataset.npz"
    ):

    X_list = []
    y_list = []

    # Process each CSV file
    for file in os.listdir(folder_path):

        if file.endswith(".csv"):

            file_path = os.path.join(folder_path, file)

            print("Processing:", file)

            # ---- Read CSV ----
            df = pd.read_csv(file_path)

            data = df.values

            # ---- Extract label ----
            label = extract_label(file)

            # ---- Create sliding windows ----
            num_rows = data.shape[0]

            for start in range(
                    0,
                    num_rows - window_size + 1,
                    step_size):

                end = start + window_size

                window = data[start:end]

                X_list.append(window)
                y_list.append(label)

    # Convert to arrays
    X = np.array(X_list)
    y = np.array(y_list)

    print("\nDataset Created")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Save dataset
    np.savez(output_file, X=X, y=y)

    print("Saved:", output_file)

    return X, y




X, y = create_dataset("D:/sensordata")


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_reshaped = X.reshape(-1, X.shape[2])
X_scaled = scaler.fit_transform(X_reshaped)

X = X_scaled.reshape(X.shape)