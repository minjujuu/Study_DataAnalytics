import nltk
import pickle
from nltk.corpus import stopwords
import re
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from os import path

nltk.download('all')

with open('C:\\Users\\aaami\\Desktop\\2022_1\\데이터애널리틱스\\과제\\5주차 개인과제\\Word_cloud_v2\XR_patents_abstract.txt', 'r', encoding='utf8') as f:
    content = f.read()
    
# customized_stopwords = ["apparatus", "using", "providing", "method", "methods", "system", "systems", "based",
# "enviorment", "generating", "device", "game", "games", "gaming", "Gaming machine", "machine", "devices", "play", "game program",
#  "program", "playing", "player", "wagering", "processing", "casino", "medium", "control", "electronic", "feature", "user", "symbol", "symbols",
# "card", "information", "virtual", "storage", "first", "second", "include", "includes",
# "one", "may", "plurality", "object", "character", "input", "display", "provided",
# "unit", "position", "least", "operation", "players", "slide", "side", "associated",
# "direction", "session", "terminal", "gameplay", "storing", "interactive", "controlling", "embodiment", "set", "outcome"]

# String function인 replace를 사용하거나 re를 사용
# cleaned_content = content.replace('!', '').replace(',','').replace('.','').replace('“','').replace('”','').replace('\n','').replace('’','')
cleaned_content = re.sub(r'[^\.\?\!\w\d\s]','',content) # 문장단위로 끊기
#print(cleaned_content)

cleaned_content = cleaned_content.lower()

word_tokens = nltk.word_tokenize(cleaned_content)
#print(word_tokens)

tokens_pos = nltk.pos_tag(word_tokens)
#print(tokens_pos)

# 명사는 NN을 포함하고 있음을 알 수 있음
NN_words = []
for word, pos in tokens_pos:
    if 'NN' in pos:
        NN_words.append(word)
#print(NN_words)

# nltk에서 제공되는 WordNetLemmatizer을 이용
# ex) 명사의 경우는 보통 복수 -> 단수 형태로 변형
wlem = nltk.WordNetLemmatizer()
lemmatized_words = []
for word in NN_words:
    new_word = wlem.lemmatize(word)
    lemmatized_words.append(new_word)

#print(lemmatized_words)

stopwords_list = stopwords.words('english') #nltk에서 제공하는 불용어사전 이용
#print('stopwords: ', stopwords_list)
unique_NN_words = set(lemmatized_words)
final_NN_words = lemmatized_words

# 불용어 제거
for word in unique_NN_words:
    if word in stopwords_list:
        while word in final_NN_words: final_NN_words.remove(word)
            
# unique_NN_words1 = set(final_NN_words)
# for word in unique_NN_words1:
#     if word in customized_stopwords:
#         while word in final_NN_words: final_NN_words.remove(word)

# print(final_NN_words)
from collections import Counter
c = Counter(final_NN_words) # input type should be a list of words (or tokens)
#print(c)
k = 100
#print(c.most_common(k)) # 빈도수 기준 상위 k개 단어 출력

print(list(c))
noun_text = ''
for word in final_NN_words:
    noun_text = noun_text +' '+word

    
wordcloud = WordCloud(max_font_size=60, relative_scaling=.5).generate(noun_text) # generate() 는 하나의 string value를 입력 받음
wordcloud.to_file("C:\\Users\\aaami\\Desktop\\2022_1\\데이터애널리틱스\\과제\\5주차 개인과제\\Word_cloud_v2/wordcloud_v2_XR_patents_abstract.png")
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
