import os
import sqlite3
import pymysql
import sqlalchemy
import praw
import pandas as pd
from datetime import datetime
#分隔島
import pymysql
import sqlalchemy
import requests
import calendar
import random
import time
import torch
import pandas as pd
import numpy as np
from fake_useragent import UserAgent
from webdriver_manager.chrome import ChromeDriverManager#自動下載驅動
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
#確認線
#title preprocess
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel
import tempfile, shutil, glob, pathlib, stat

import time, tempfile, shutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options



#Reddit建議user_agent格式"平台:專案名稱:版本字號 (by /u/你的Reddit帳號)"
reddit = praw.Reddit(
    client_id = "YOUR_ID",
    client_secret = "YOUR_SECRET",
    user_agent = "YOUR_AGENT"
)

#爬蟲(twse每日資料更新)
chrome = Options()
#chrome.add_experimental_option("detach", True)
#1
chrome.add_argument(f"--user-agent={UserAgent().random}")

#chrome.add_experimental_option("excludeSwitches",["enable-automation"])，過氣寫法

chrome.add_experimental_option("useAutomationExtension", False)
chrome.add_argument("--start-maximized")
#2
chrome.add_argument("--disable-blink-features=AutomationControlled")
chrome.add_argument("--no-first-run")
chrome.add_argument("--no-default-browser-check")
chrome.add_argument("--remote-debugging-port=0")   # 避免 9222 埠衝突，爬蟲多開避免衝突可能會用到
chrome.add_argument("--disable-dev-shm-usage")
#chrome.add_argument("--no-sandbox") #需要權限才要加上
chrome.add_argument("--incognito")
#chrome.add_argument("--user-agent=Mozilla/5.0")

driver = webdriver.Chrome(options = chrome)
driver.execute_cdp_cmd("Network.enable",{})
driver.get('https://www.twse.com.tw/zh/trading/historical/bwibbu-day.html')
driver.delete_all_cookies()

time.sleep(0.3)

today = datetime.today().strftime("%Y-%m-%d")
year, month, day = map(int, today.split('-'))

year_val = WebDriverWait(driver,5).until(
        EC.presence_of_element_located((By.XPATH,f'//select[@id = "label0"]/option[contains(@value, "{year}")]'))
        )
year_val.click()

month_val = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, f'//select[@name = "mm"]/option[contains(@value, "{month}")]'))
        )
month_val.click()

day_val  = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH,f'//select[@name = "dd"]/option[contains(@value, "{day}")]'))
                    )
day_val.click()


collected = {}
WebDriverWait(driver, 5).until(
                    EC.any_of(
                        EC.visibility_of_element_located((By.XPATH,'//div[@class = "message" and contains(text(),"抱歉")]')),
                        EC.visibility_of_element_located((By.XPATH, '//*[@id="reports"]/div[2]/div[2]'))
                        )
                    )
                
                #若只使用driver.find_element，當沒有"抱歉"字樣，而是表格時，會錯誤跳到NoSuchElementException，會變成提取不了表格
no_data = driver.find_elements(By.XPATH,'//div[@class = "message" and contains(text(),"抱歉")]')
if no_data:
    print("無資料")
#continue重新回到日期選擇，直到有資料才執行else
else:
    option = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, '//div[@class = "per-page"]//select/option[@value = "-1"]'))
                        )
    option.click()

    table = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="reports"]/div[2]/div[2]'))
                        )
    data = table.text
if data:
    lines = data.strip().split("\n")
    headers = lines[0].split()
    rows = [line.split() for line in lines[1:]]

    df = pd.DataFrame(rows, columns = headers)
    numeric_cols = ["收盤價","殖利率(%)","本益比","股價淨值比"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors = "coerce")#errors = "coerce"，即便有NAN仍然保留並，繼續順利執行
    df = df.dropna(subset = ["收盤價","殖利率(%)","本益比","股價淨值比"])

    filtered_df = df[(df["本益比"] > 10) & (df["本益比" ]< 40)] 
    collected[f"{year}-{month:02d}-{day:02d}"] = filtered_df

if collected:
    df_twse = pd.concat(
        [df.assign(
        年 = int(collected_data.split("-")[0]),
        月 = int(collected_data.split("-")[1]),
        日 = int(collected_data.split("-")[2]),
        日期 = pd.to_datetime(collected_data))
        for collected_data, df in collected.items()],
        ignore_index = True
        )
    titles = []
    for post in reddit.subreddit("worldnews").hot(limit = 10):
        try:
            print(post.title)
            titles.append(post.title)
        except UnicodeEncodeError:
            print(post.title.encode('urf-8', errors = 'replace').decode())

#[today]*len(titles)才能每個標題都能對應到日期
    df_reddit = pd.DataFrame({
        "年":[year]*len(titles),
        "月":[month]*len(titles),
        "日":[day]*len(titles),
        "日期":[today]*len(titles),
        "標題":titles
        })

    print(df_reddit)
driver.quit()
df_reddit["日期"] = pd.to_datetime(df_reddit["日期"] )
merge_df = pd.merge(df_twse, df_reddit, on = ["年","月","日","日期"], how = "left")

group_news = merge_df.groupby("日期")["標題"].apply(lambda titles:" ".join(titles)).reset_index()
full_text = group_news["標題"].values[0]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

inputs = tokenizer(full_text, return_tensors = "pt", truncation = True, max_length = 512)
with torch.no_grad():
    outputs = model(**inputs)
#取出我只輸入的一段文本，跟這段文本的第一個token，然後:就是整個向量(768維度)。
cls_embedding = outputs.last_hidden_state[0,0, :].numpy()

sid = SentimentIntensityAnalyzer()
group_news["情緒分數"] = group_news["標題"].apply(lambda x: sid.polarity_scores(x)["compound"]) 
score = group_news["情緒分數"].iloc[0]
bert_df = pd.DataFrame([cls_embedding])
bert_df["日期"] = group_news["日期"].values[0]
bert_df["平均情緒分數"] = score
train_df = pd.merge(df_twse, bert_df, on = "日期", how = "left")


#SQLlite
conn = sqlite3.connect("full_data.db")
try:
    existing_dates = pd.read_sql('SELECT DISTINCT 日期 FROM full_data',conn)
except:
    existing_dates = pd.DataFrame(columns = ["日期"])

#Pandas偏好datetime[ns]，所以全部轉Pandas要求的格式
existing_dates["日期"] = pd.to_datetime(existing_dates["日期"])
train_df["日期"] = pd.to_datetime(train_df["日期"])

if not existing_dates.empty:
    new_data = train_df[~train_df["日期"].isin(existing_dates["日期"])]
else:
    new_data = train_df

if not new_data.empty:
    new_data.to_sql("full_data", conn, if_exists = 'append', index = False)
else:
    print("有重複")

train_df.head()


'''
#excel老方法
file_path = "reddit_titles.xlsx"
if os.path.exists(file_path):
    df_old = pd.read_excel(file_path)
    df_all = pd.concat([df_old,df_today],ignore_index = True)
else:
    df_all = df_today

df_all.to_excel(file_path, index = False)
'''
#TF-idf作法
'''
#title preprocess
def preprocess(text):
    text = str(text).lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    stop_words.update(string.ascii_lowercase)
    
    filtered = [w for w in tokens 
                if w not in stop_words and not re.fullmatch(r"[\d\.\%\-,]+", w)]

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    lemmatized = [lemmatizer.lemmatize(w, pos = 'v') for w in lemmatized]

    return " ".join(lemmatized)

group_news["轉換標題"] = group_news["標題"].apply(preprocess)
tifdf_vector = TfidfVectorizer(min_df = 1, max_df = 1.0, max_features = 200, ngram_range = (1,2))
final_tfidf = tifdf_vector.fit_transform(group_news["轉換標題"])
tifdf_df = pd.DataFrame(
    final_tfidf.toarray(),
    columns = tifdf_vector.get_feature_names_out()
)
tifdf_df["日期"] = group_news["日期"].values
train_df = pd.merge(df_twse, tifdf_df, on="日期", how="left")
'''


'''
#MYSQL
username = "tommy0000"
password = "tommycurry30"
db_name = "tommy0000$default"

#f"mysql+pymyspl://{username}:{password}@{host}:{port}/database"
connection_str = f"mysql+pymysql://{username}:{password}@tommy0000.mysql.pythonanywhere-services.com/{db_name}"
engine = sqlalchemy.create_engine(connection_str)
merge_df.to_sql("reddit_data", engine,if_exists = "append", index = False)
print("已新增資料至SQL")
'''
