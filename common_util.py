import pandas as pd
import torch

def convert_ptlist_to_excel(ptfile_path):
    list_data = torch.load(ptfile_path)
    df = pd.DataFrame(list_data, columns=['z_coord'])

    excel_path = ptfile_path[:-4] + ".xlsx"
    df.to_excel(excel_path, index=False)
