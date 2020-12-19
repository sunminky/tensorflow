import os
import tarfile
import urllib.request
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): #데이터 받아오기
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    if os.path.exists(HOUSING_PATH+"/housing.tgz"):
        return

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):   #파일에서 데이터를 읽어서 반환
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)