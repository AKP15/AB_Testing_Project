import numpy as np
import pandas as pd

class InspectData:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def check_data(self) -> pd.DataFrame:
        return self.data.head(5)

    def check_info(self) -> pd.DataFrame:
        return self.data.info()

    def check_shape(self) -> pd.DataFrame:
        return self.data.shape

    def check_dtype(self) -> pd.DataFrame:
        return self.data.dtypes

    def check_missing(self) -> pd.DataFrame:
        return self.data.isnull().sum()

    def change_col(self) -> pd.DataFrame:
        self.data.columns = (
                self.data.columns
                .str.strip()
                .str.lower()
                .str.replace(r'\s+', '_', regex=True)
                )
        return self.data

    def missing_percent(self) -> pd.DataFrame:
        return self.data.isnull().mean()*100

    def check_duplicate(self)-> pd.DataFrame:
        return self.data.duplicated().sum()
