import pandas as pd
class IngestData:

    def __init__(self,data_path) -> None:
        self.data_path = data_path;

                    
    def open_csv(self):
        return pd.read_csv(self.data_path,encoding="latin1")

"""
     -> None type hint indicates it returns nothing.
    
    get_data method. It takes a file path as a string and 
    returns a Pandas DataFrame.

"""



