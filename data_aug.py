import numpy as np

def augment_with_noise(
        X,
        y,
        sensor_feature_count,
        mfcc_feature_count=13,
        sensor_noise_std=0.02,
        mfcc_noise_std=0.08,
        augmentation_factor=1
    ):
    """
    Add different noise to sensor and MFCC data.

    Parameters
    ----------
    X : array
        shape = (samples, 50, features)

    y : array
        labels

    sensor_feature_count : int
        Number of sensor columns

    mfcc_feature_count : int
        Default = 13

    augmentation_factor : int
        Number of noisy copies to generate
    """

    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(augmentation_factor):

        X_noisy = np.copy(X)

        # Split features
        sensor_part = X_noisy[:, :, :sensor_feature_count]
        mfcc_part = X_noisy[:, :, sensor_feature_count:
                                      sensor_feature_count + mfcc_feature_count]

        # ---- Sensor Noise ----
        sensor_noise = np.random.normal(
            0,
            sensor_noise_std,
            sensor_part.shape
        )

        sensor_part += sensor_noise

        # ---- MFCC Noise ----
        mfcc_noise = np.random.normal(
            0,
            mfcc_noise_std,
            mfcc_part.shape
        )

        mfcc_part += mfcc_noise

        # Merge back
        X_noisy[:, :, :sensor_feature_count] = sensor_part
        X_noisy[:, :, sensor_feature_count:
                      sensor_feature_count + mfcc_feature_count] = mfcc_part

        X_aug_list.append(X_noisy)
        y_aug_list.append(y)

    # Combine original + augmented
    X_final = np.vstack(X_aug_list)
    y_final = np.hstack(y_aug_list)

    print("Augmented Dataset Shape:")
    print("X:", X_final.shape)
    print("y:", y_final.shape)

    return X_final, y_final


data = np.load("dataset.npz")

X = data["X"]
y = data["y"]

sensor_feature_count = 20 
mfcc_feature_count = 13

X_aug, y_aug = augment_with_noise(
    X,
    y,
    sensor_feature_count=sensor_feature_count,
    mfcc_feature_count=mfcc_feature_count,
    augmentation_factor=2
)

np.savez("dataset_augmented.npz", X=X_aug, y=y_aug)