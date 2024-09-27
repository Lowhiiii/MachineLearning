from typing import Tuple, Dict
import pandas as pd
import json
from network_ad.config import DATA_PATH, BASE_FEATURES, TEST_RATIO, RAW_DATA_FILE, NUMERICAL_FEATURES, \
    CATEGORICAL_FEATURES, PREPROCESSED_DATA_PATH
import sys
sys.path.append("../..")

def load_raw_data() -> pd.DataFrame:
    """
    Load train data or test data
    :return df: The raw data
    """
    df = pd.read_csv(DATA_PATH / RAW_DATA_FILE)
    for col in BASE_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the data")

    return df[BASE_FEATURES]


def compute_ip_network_and_host_features(data: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each flow (line) compute the following features:
    - NETWORK_IPV4_SRC_ADDR: The network part of the source IP address
    - NETWORK_IPV4_DST_ADDR: The network part of the destination IP address
    - HOST_IPV4_SRC_ADDR: The host part of the source IP address
    - HOST_IPV4_DST_ADDR: The host part of the destination IP address
    :param data:
    :return:
    """

    # Ensure the input dataframe has the necessary columns
    required_columns = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input dataframe must contain the following columns: {required_columns}")

    # Extract the network and host parts of the source IP address
    data["NETWORK_IPV4_SRC_ADDR"] = data["IPV4_SRC_ADDR"].str.rsplit(".", n=1).str[0]
    data["HOST_IPV4_SRC_ADDR"] = data["IPV4_SRC_ADDR"].str.rsplit(".", n=1).str[1]

    # Extract the network and host parts of the destination IP address
    data["NETWORK_IPV4_DST_ADDR"] = data["IPV4_DST_ADDR"].str.rsplit(".", n=1).str[0]
    data["HOST_IPV4_DST_ADDR"] = data["IPV4_DST_ADDR"].str.rsplit(".", n=1).str[1]

    #Remove the original IP address columns
    data.drop(columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"], inplace=True)

    return data


def clip_ports (data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Standard ports keep the same (0-1023)
    registered ports (1024-49151) are clipped to 1025
    dynamic ports (49152-65535) are clipped to 49152

    :param data:
    :return: data with clipped ports
    """

    data["CLIP_L4_SRC_PORT"] = data["L4_SRC_PORT"].apply(lambda x: 1025 if 1024 <= x <= 49151 else 49152 if 49152 <= x <= 65535 else x)
    data["CLIP_L4_DST_PORT"] = data["L4_DST_PORT"].apply(lambda x: 1025 if 1024 <= x <= 49151 else 49152 if 49152 <= x <= 65535 else x)
    return data


def convert_categorical_features_to_string(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the categorical features to string type
    :param data: The input dataframe
    :return: The dataframe with categorical features converted to string type
    """
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(str)
    return data




def compute_and_save_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute and save the statistics for the numerical and categorical features
    :param data: The input dataframe
    :param save_path: Path to save the statistics
    """
    # Compute the mean and standard deviation for each numerical feature
    numerical_stats = data[NUMERICAL_FEATURES].agg(['mean', 'std', 'min', 'max']).transpose()
    # Save the distinc values of categorical features to json to fit one hot encoding
    categorical_stats = {}
    for col in CATEGORICAL_FEATURES:
        categorical_stats[col] = data[col].unique().tolist()
    return numerical_stats, categorical_stats




def train_test_split(data: pd.DataFrame = None, test_ratio: float = 0.2) -> (pd.DataFrame, pd.DataFrame):
    """
    Split the data into training and testing sets
    :param data: The input dataframe containing network flow data
    :param test_ratio: The ratio of the data to be used for testing
    :return: A tuple of two dataframes: (train_data, test_data)
    """
    #Random shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Calculate the number of rows for the test set
    test_rows = int(len(data) * test_ratio)

    # Split the data into training and testing sets
    test_data = data.iloc[:test_rows]
    train_data = data.iloc[test_rows:]

    return train_data, test_data


def save_data(train_data: pd.DataFrame,
              test_data: pd.DataFrame,
              numerical_stats: pd.DataFrame,
              categorical_stats: Dict):
    """
    Save the training and testing data to CSV files
    :param train_data: The training data
    :param test_data: The testing data
    """
    if not PREPROCESSED_DATA_PATH.exists():
        PREPROCESSED_DATA_PATH.mkdir(parents=True)
    train_data.to_csv(PREPROCESSED_DATA_PATH / "train.csv", index=False)
    test_data.to_csv(PREPROCESSED_DATA_PATH / "test.csv", index=False)

    numerical_stats.to_csv(PREPROCESSED_DATA_PATH / "numerical_stats.csv")

    #Save the categorical stats to json
    with open(PREPROCESSED_DATA_PATH / "categorical_stats.json", "w") as f:
        json.dump(categorical_stats, f)



if __name__ == "__main__":

    print("1. Loading Raw Data...")
    df  = load_raw_data()

    print("2. Computing IP Network and Host Features...")
    df = compute_ip_network_and_host_features(df)

    print("3. Clipping Ports...")
    df = clip_ports(df)

    print("4. Converting Categorical Features to String...")
    df = convert_categorical_features_to_string(df)

    print("5. Computing and Saving Statistics...")
    numerical_stats, categorical_stats = compute_and_save_statistics(df)

    print(f"6. Train test splitting with test ratio of {TEST_RATIO}...")
    train_data, test_data = train_test_split(df, TEST_RATIO)


    print("7. Saving Data...")
    save_data(train_data, test_data , numerical_stats, categorical_stats)

    print("Done!")
