import pandas as pd
from pandas.core.frame import DataFrame
pd.set_option('display.max_rows', 1000)

import json
import os
import shutil
import requests
from datetime import datetime
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth


  

class Crawler():

  def __init__(self):
    self.parks  =  ["WaltDisneyWorldMagicKingdom",
                    "WaltDisneyWorldEpcot",
                    "WaltDisneyWorldHollywoodStudios",
                    "WaltDisneyWorldAnimalKingdom",
                    "UniversalIslandsOfAdventure" ,
                    "UniversalStudiosFlorida"]

    self.df_parks_waittime = pd.DataFrame()
    self.upload_file = ""
    self.folder_id ="---"

    self.file_datetime = datetime.now().strftime("%Y-%m-%d-%H%M%S")

  
  def get_parks_waittime(self):

    for park in self.parks:
      print(park)

      url_waittime = f"https://api.themeparks.wiki/preview/parks/{park}/waittime"
      r = requests.get(url_waittime)
      d = json.loads(r.text)
      df = pd.json_normalize(d)
      df = df[~(df.status.isna()) & ~(df["meta.type"] == 'RESTAURANT') & ~(df['waitTime'].isna()) ] 
      df['park'] = park
      df["data_carga"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      columns = ["park","name","waitTime","status","active","lastUpdate","meta.longitude","meta.latitude"]
      df = df.filter(columns)
      self.df_parks_waittime = self.df_parks_waittime.append(df)

    return self 
  
  def save_dataframe(self):

   
    print("date and time:",self.file_datetime)	

    self.upload_file = f"../data/{self.file_datetime}.csv"
    self.df_parks_waittime.to_csv(self.upload_file, index=False)
    
    return self

  def gdrive_upload(self):

    gauth = GoogleAuth()

    gauth.LocalWebserverAuth()	
    drive = GoogleDrive(gauth)

    gfile = drive.CreateFile({'parents': [{'id': self.folder_id}], 'title': f"{self.file_datetime}.csv" })

    gfile.SetContentFile(self.upload_file)
    gfile.Upload() # Upload the file.

    return self

  def delete_local_files(self):    
    os.remove(self.upload_file)
