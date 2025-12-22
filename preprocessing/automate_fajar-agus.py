import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def remove_outliers_iqr(df, columns):
    """
    Menghapus outlier menggunakan metode IQR
    """
    df_clean = df.copy()

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean = df_clean[
            (df_clean[col] >= lower_bound) &
            (df_clean[col] <= upper_bound)
        ]

    return df_clean


def preprocess_data(input_path, output_dir):
    """
    Fungsi utama preprocessing dataset BMW
    """

    # ======================
    # 1. Data Loading
    # ======================
    df = pd.read_csv(input_path)

    # ======================
    # 2. Data Cleaning
    # ======================
    df = df.drop_duplicates()

    median_engine_size = df.loc[df["engineSize"] > 0, "engineSize"].median()
    df.loc[df["engineSize"] == 0, "engineSize"] = median_engine_size

    num_cols = ["price", "mileage", "tax", "mpg", "engineSize"]
    df = remove_outliers_iqr(df, num_cols)

    # ======================
    # 3. Encoding
    # ======================
    df_encoded = pd.get_dummies(
        df,
        columns=["model", "transmission", "fuelType"],
        drop_first=True
    )

    # ======================
    # 4. NORMALISASI NAMA KOLOM (WAJIB)
    # ======================
    df_encoded.columns = (
        df_encoded.columns
        .str.replace(r"_\s+", "_", regex=True)   # "_ 2" → "_2"
        .str.replace(r"\s+", "_", regex=True)    # spasi → "_"
        .str.replace("-", "_", regex=False)      # "-" → "_"
        .str.lower()                             # lowercase
    )

    # ======================
    # 5. Split Feature & Target
    # ======================
    X = df_encoded.drop(columns="price")
    y = df_encoded["price"]

    # ======================
    # 6. Scaling
    # ======================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["price"] = y.values

    # ======================
    # 7. Save Dataset
    # ======================
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "bmw_preprocessed.csv"
    )

    processed_df.to_csv(output_path, index=False)

    print(f"[INFO] Dataset preprocessing berhasil disimpan di: {output_path}")
    print(f"[INFO] Total fitur: {processed_df.shape[1] - 1}")

    return processed_df


# ======================
# Entry Point
# ======================
if __name__ == "__main__":
    INPUT_PATH = "../bmw_raw/bmw.csv"
    OUTPUT_DIR = "bmw_preprocessing"

    preprocess_data(INPUT_PATH, OUTPUT_DIR)
