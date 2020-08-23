from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import time
import re
import pandas as pd
import nltk
import os


def get_videos(channel_list):
    title_list = []
    view_num_list = []
    url_list = []
    channel_name = channel.split('/')[-2]
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get(channel)
    # 한번 스크롤 하고 멈출 시간 설정
    body = driver.find_element_by_tag_name('body')
    # body태그를 선택하여 body에 넣음
    while True:
        last_height = driver.execute_script('return document.documentElement.scrollHeight')
        # 현재 화면의 길이를 리턴 받아 last_height에 넣음
        for i in range(10):
            body.send_keys(Keys.END)
            # body 본문에 END키를 입력(스크롤내림)
        #             time.sleep(0.5)
        new_height = driver.execute_script('return document.documentElement.scrollHeight')
        if new_height == last_height:
            break;
    page = driver.page_source
    soup = BeautifulSoup(page, 'lxml')
    all_videos = soup.find_all(id='dismissable')
    for video in all_videos:
        title = video.find(id='video-title')
        if len(title.text.strip()) > 0:
            title_list.append(title.text)
        # 공백을 제거하고 글자수가 0보다 크면 append
    view_num_regexp = re.compile(r'조회수')
    for video in all_videos:
        view_num = video.find('span', {'class': 'style-scope ytd-grid-video-renderer'})
        if view_num_regexp.search(view_num.text):
            # view_num.text 에 '조회수' 문자열이 있으면 True
            view_num_list.append(view_num.text)
    a = soup.find_all(id='video-title')
    lst = []
    for item in a:
        lst.append(item.attrs['href'])
    for link in lst:
        url_link = 'https://www.youtube.com' + link
        url_list.append(url_link)
    driver.quit()
    dict_youtube = {'title': title_list, 'view_num': view_num_list, 'url': url_list}
    df = pd.DataFrame(dict_youtube)
    for i, row in df.iterrows():
        row['view_num'] = row['view_num'].strip('조회수')
        if '천' in row['view_num']:
            row['view_num'] = float(row['view_num'].strip('천')) * 1000
        elif '만' in row['view_num']:
            row['view_num'] = float(row['view_num'].strip('만')) * 10000
        else:
            row['view_num'] = float(row['view_num'])
    df = df.sort_values(by='view_num', ascending=False)

    return channel_name, df

def tokenization(df):
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                 'it', "it's", 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 "that'll", 'these', 'those', 'am',
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the',
                 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'off', 'over', 'under', 'again', 'further',
                 'at', 'by', 'for', 'with', 'about', 'during', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                 'how', 'all', 'any', 'both', 'each', 'few',
                 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
                 "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                 'haven', "haven't", 'isn', "isn't", 'ma',
                 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'youtube', ]

    text = df[['title']].copy()
    text['tokenized_title'] = text.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)
    text['tokenized_title'] = text['tokenized_title'].apply(
        lambda x: [word.lower() for word in x if word.lower() not in (stopwords)])
    text['tokenized_title'] = text['tokenized_title'].apply(
        lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])

    # 단어 인덱싱을 위한 단어 딕셔너리
    detokenized_doc = []
    for i, row in text.iterrows():
        t = ' '.join(row['tokenized_title'])
        detokenized_doc.append(t)
    text['detokenized_title'] = detokenized_doc
    text = text.drop(['title'], axis=1)
    new_df = pd.concat([df, text], axis=1)

    return new_df
