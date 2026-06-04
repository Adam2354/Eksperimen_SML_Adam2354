import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data():

    # ==================================================
    # Menentukan lokasi file secara otomatis
    # ==================================================
    current_dir = Path(__file__).resolve().parent

    input_path = current_dir.parent / "dataset_raw" / "telco_churn.csv"

    output_path = current_dir / "telco_churn_preprocessing.csv"

    print("=" * 60)
    print("MEMULAI PREPROCESSING DATA")
    print("=" * 60)

    print("\nLokasi Dataset:")
    print(input_path)

    # ==================================================
    # Load Dataset
    # ==================================================
    print("\n[1] Loading Dataset...")

    df = pd.read_csv(input_path)

    print(f"Shape Awal : {df.shape}")

    # ==================================================
    # Drop customerID
    # ==================================================
    print("\n[2] Menghapus customerID...")

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # ==================================================
    # Konversi TotalCharges
    # ==================================================
    print("\n[3] Konversi TotalCharges...")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # ==================================================
    # Missing Value
    # ==================================================
    print("\n[4] Menangani Missing Value...")

    print("Missing sebelum:")
    print(df.isnull().sum())

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    print("\nMissing sesudah:")
    print(df.isnull().sum())

    # ==================================================
    # Hapus Duplikat
    # ==================================================
    print("\n[5] Menghapus Duplikat...")

    duplicate_count = df.duplicated().sum()

    print(f"Jumlah duplikat: {duplicate_count}")

    df.drop_duplicates(inplace=True)

    # ==================================================
    # Encoding
    # ==================================================
    print("\n[6] Encoding Fitur Kategorikal...")

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    encoder = LabelEncoder()

    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])

    print(f"Jumlah kolom kategorikal: {len(categorical_columns)}")

    # ==================================================
    # Scaling
    # ==================================================
    print("\n[7] Feature Scaling...")

    scaler = StandardScaler()

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df[numerical_columns] = scaler.fit_transform(
        df[numerical_columns]
    )

    # ==================================================
    # Simpan Dataset
    # ==================================================
    print("\n[8] Menyimpan Dataset...")

    df.to_csv(
        output_path,
        index=False
    )

    print(f"\nDataset berhasil disimpan:")
    print(output_path)

    print(f"\nShape Akhir : {df.shape}")

    return df


if __name__ == "__main__":

    processed_df = preprocess_data()

    print("\nPreview Dataset:")
    print(processed_df.head())