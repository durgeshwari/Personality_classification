
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import tweepy
import sys
import os
import nltk 
import re
import numpy as np
import string
from unidecode import unidecode
import csv
from itertools import islice
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')

print("Hello")


ckey='YvQZ4wcEhYo4F4Lj8PQqj500d'
csecret='nDMvzbvQxt2zW9GZOalnOikeDcwGwPwvi2USOD5phJCmw5e9wD'
atoken='994417184322433025-anaAEWzZErK5h8CGTxEgJMPFwfRoMNA'
asecret='M2RwOR7CaWHMO7d83iWuKhHmEtgyaomobBNt9bsJhSF3i'

auth=tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api=tweepy.API(auth)

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def preproc(s):
	#s=emoji_pattern.sub(r'', s) # no emoji
	s= unidecode(s)
	POSTagger=preprocess(s)
	#print(POSTagger)

	tweet=' '.join(POSTagger)
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(tweet)
	#filtered_sentence = [w for w in word_tokens if not w in stop_words]
    
	filtered_sentence = []
	for w in POSTagger:
	    if w not in stop_words:
	        filtered_sentence.append(w)
            
	#print(word_tokens)
	#print(filtered_sentence)
    
	stemmed_sentence=[]
	stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    
	for w in filtered_sentence:
		stemmed_sentence.append(stemmer2.stem(w))
	#print(stemmed_sentence)

	temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation) 
	preProcessed=temp.split(" ")
	final=[]
    
	for i in preProcessed:
		if i not in final:
			if i.isdigit():
				pass
			else:
				if 'http' not in i:
					final.append(i)
	temp1=' '.join(c for c in final)
	#print(preProcessed)
	return temp1

def getTweets(user):
	csvFile = open('user.csv', 'a', newline='')
	csvWriter = csv.writer(csvFile)
	try:
		for i in range(0,4):
			tweets=api.user_timeline(screen_name = user, count = 1000, include_rts=True, page=i)
			for status in tweets:
				tw=preproc(status.text)
				if tw.find(" ") == -1:
					tw="blank"
				csvWriter.writerow([tw])
	except tweepy.TweepError:
		print("Failed to run the command on that user, Skipping...")
	csvFile.close()



username=input("Please Enter Twitter Account handle: ")
getTweets(username)

with open('user.csv','rt') as f:
	csvReader=csv.reader(f)
	tweetList=[rows[0] for rows in csvReader]
os.remove('user.csv')

with open('CSV_Data/newfrequency300.csv','rt') as f:
	csvReader=csv.reader(f)
	mydict={rows[1]: int(rows[0]) for rows in csvReader}

vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(tweetList).toarray()
print("x->",x)
df=pd.DataFrame(x)


model_IE = pickle.load(open("Pickle_Data/BNIEFinal.sav", 'rb'))
model_SN = pickle.load(open("Pickle_Data/BNSNFinal.sav", 'rb'))
model_TF = pickle.load(open('Pickle_Data/BNTFFinal.sav', 'rb'))
model_PJ = pickle.load(open('Pickle_Data/BNPJFinal.sav', 'rb'))

answer=[]

print(df)

IE=model_IE.predict(df)
SN=model_SN.predict(df)
TF=model_TF.predict(df)
PJ=model_PJ.predict(df)
print("ie",IE)

pred_perc=[]

b = Counter(IE)
 
print("b->",b)

value=b.most_common(1)
b=dict(b)

pred_perc.append((value[0][1]*100)//(b.get(1.0,0)+b.get(0.0,0)))
print('value',value)

#print(value)
if value[0][0] == 1.0:
	answer.append("I")
else:
	answer.append("E")

b = Counter(SN)

value=b.most_common(1)
#print(value)


b=dict(b)
pred_perc.append((value[0][1]*100)//(b.get(1.0,0)+b.get(0.0,0)))


if value[0][0] == 1.0:
	answer.append("S")
else:
	answer.append("N")

b = Counter(TF)
value=b.most_common(1)
#print(value)


b=dict(b)
pred_perc.append((value[0][1]*100)//(b.get(1.0,0)+b.get(0.0,0)))


if value[0][0] == 1:
	answer.append("T")
else:
	answer.append("F")

b = Counter(PJ)
value=b.most_common(1)
#print(value)

b=dict(b)

pred_perc.append((value[0][1]*100)//(b.get(1.0,0)+b.get(0.0,0)))

if value[0][0] == 1:
	answer.append("P")
else:
	answer.append("J")
mbti="".join(answer)

# Classifying Personality's ==========================================>

print('===============================================================================>')

print('User Name - ' + username + '\n')

print('Type of Personality \n')


char_per=dict()
opp_char ={'I':'E','E':'I','S':'N','N':'S',
'T':'F','F':'T','J':'P','P':'J'}
for i in range(4):
    char_per[answer[i]]=pred_perc[i]
    char_per[opp_char[answer[i]]]=100-pred_perc[i]

print(char_per)

all_pers = ['ENFJ','ISTJ','INFJ','INTJ','ISTP','ESFJ','INFP','ESFP','ENFP','ESTP','ESTJ','ENTJ','INTP','ISFJ','ENTP','ISFP']

pers_perc={}
for word in all_pers:
    perc=sum([char_per[el] for el in word])//4
    pers_perc[word]=perc
print(pers_perc)

k=Counter(pers_perc)
max3=k.most_common(3)

for i in max3:
  print(i[0]," : ",i[1]," \n")

if mbti == 'ENFJ':
	str1 = '" The Giver "'
	print(mbti +' - '+ str1)
	print("\nThe Giver :-")
	print(" A Protagonist (ENFJ) is a person with the Extraverted, Intuitive, Feeling,\n and Judging personality traits. These warm, forthright types love \n helping others, and they tend to have strong ideas and values. They back their perspective \n with the creative energy to achieve their goals.")
	print("Famous personalities whose traits you resemble are: \n Barack Obama,Oprah Winfrey , John Cusack, Ben Affleck, Malala Yousafzai.")


elif mbti == 'ISTJ':
	str1 = '" The Inspector "'
	print(mbti +' - '+ str1)
	print("\nThe Inspector :-")
	print("A Logistician (ISTJ) is someone with the Introverted, Observant,\n Thinking, and Judging personality traits. These people tend to be reserved yet willful,\n with a rational outlook on life. They compose their actions carefully and carry \n them out with methodical purpose.")
	print("Famous personalities whose traits you resemble are: \n George H.W. Bush, Hermione Granger(Harry Potter), George Washington.")

elif mbti == 'INFJ':
	str1 = '" The Counselor "'
	print(mbti +' - '+ str1)
	print("\nThe Counselor :-")
	print(" An Advocate (INFJ) is someone with the Introverted, Intuitive, \n Feeling, and Judging personality traits. They tend to approach life with \n deep thoughtfulness and imagination. Their inner vision, personal values, \n and a quiet, principled version of humanism guide them in all things.")
	print("Famous personalities whose traits you resemble are: \n Nelson Mandela, Mother Teresa, Martin Luther King.")

elif mbti == 'INTJ':
	str1 = '" The Mastermind "'
	print(mbti +' - '+ str1)
	print("\nThe Mastermind :-")
	print(" An Architect (INTJ) is a person with the Introverted, Intuitive, Thinking,\n and Judging personality traits. \n These thoughtful tacticians love perfecting the details of life, applying creativity and \n rationality to everything they do.\n Their inner world is often a private, complex one.")
	print("Famous personalities whose traits you resemble are: \n Miclelle Obama, Elon Musk, Christopher Nolan, Friedrich Nietzsche.")

elif mbti == 'ISTP':
	str1 = '" The Craftsman "'
	print(mbti +' - '+ str1)
	print("\nThe Craftsman :-")
	print(" A Virtuoso (ISTP) is someone with the Introverted, Observant, \n Thinking, and Prospecting personality traits. They tend to have an individualistic \n mindset, pursuing goals without needing much external connection. They engage in \n life with inquisitiveness and personal skill, varying their approach as needed.")
	print("Famous personalities whose traits you resemble are: \n Bear Grylls, Olivia Wilde, Tom Cruise .")

elif mbti == 'ESFJ':
	str1 = '" The Provider "'
	print(mbti +' - '+ str1)
	print("\nThe Provider :-")
	print(" A Consul (ESFJ) is a person with the Extraverted, Observant, \n Feeling, and Judging personality traits. They are attentive and people-focused,\n and they enjoy taking part in their social community. Their achievements are guided \n by decisive values, and they willingly offer guidance to others. ")
	print("Famous personalities whose traits you resemble are: \n Bill Clinton,Taylor Swift,Steve Harvey.")

elif mbti == 'INFP':
	str1 = '" The Idealist "'
	print(mbti +' - '+ str1)
	print("\nThe Idealist :-")
	print("  A Mediator (INFP) is someone who possesses the Introverted, \n Intuitive, Feeling, and Prospecting personality traits. These rare personality \n types tend to be quiet, open-minded, and imaginative, and they apply a caring and \n creative approach to everything they do. ")
	print("Famous personalities whose traits you resemble are: \n William Shakespeare,To Hiddleston, William Wordsworth.")

elif mbti == 'ESFP':
	str1 = '" The Performer "'
	print(mbti +' - '+ str1)
	print("\nThe Performer :-")
	print("An Entertainer (ESFP) is a person with the Extraverted, Observant, Feeling, and Prospecting personality traits. These people love vibrant experiences, engaging in life eagerly and taking pleasure in discovering the unknown. They can be very social, often encouraging others into shared activities.")
	print("Famous personalities whose traits you resemble are: \n Adele, Elton John, Jamie Foxx ")

elif mbti == 'ENFP':
	str1 = '" The Champion "'
	print(mbti +' - '+ str1)
	print("\nThe Champion :-")
	print(" A Campaigner (ENFP) is someone with the Extraverted, Intuitive, \n Feeling, and Prospecting personality traits. These people tend to embrace big ideas \n and actions that reflect their sense of hope and goodwill toward others. Their vibrant \n energy can flow in many directions. ")
	print("Famous personalities whose traits you resemble are: \n Will Smith, Kelly Clarkson, Robert Downey.")

elif mbti == 'ESTP':
	str1 = '" The Doer "'
	print(mbti +' - '+ str1)
	print("\nThe Doer :-")
	print(" An Entrepreneur (ESTP) is someone with the Extraverted, Observant, \n Thinking, and Prospecting personality traits. They tend to be energetic and \n action-oriented, deftly navigating whatever is in front of them. They love \n uncovering life’s opportunities, whether socializing with others or in more personal pursuits. ")
	print("Famous personalities whose traits you resemble are: \n Jack Nicholson, Ernest Hemingway,Nicolas Sarkozy.")

elif mbti == 'ESTJ':
	str1 = '" The Supervisor "'
	print(mbti +' - '+ str1)
	print("\nThe Supervisor :-")
	print(" An Executive (ESTJ) is someone with the Extraverted, Observant, \n Thinking, and Judging personality traits. They possess great fortitude, \n emphatically following their own sensible judgment. They often serve as a \n stabilizing force among others, able to offer solid direction amid adversity.")
	print("Famous personalities whose traits you resemble are: \n John D Rockefeller, James Monroe , Laura linney, Sonia Sotomayor.")

elif mbti == 'ENTJ':
	str1 = '" The Commander "'
	print(mbti +' - '+ str1)
	print("\nThe Commander :-")
	print(" Their secondary mode of operation is internal, where intuition \n and reasoning take effect. ENTJs are natural born leaders among \n the 16 personality types and like being in charge. They live in a world of \n possibilities and they often see challenges and obstacles as great \n opportunities to push themselves.")
	print("Famous personalities whose traits you resemble are: \n Steve jobs, Franklin D. Roosevelt, Gordon Ramsay.")

elif mbti == 'INTP':
	str1 = '" The Thinker "'
	print(mbti +' - '+ str1)
	print("\nThe Thinker :-")
	print(" A Logician (INTP) is someone with the Introverted, Intuitive, Thinking, \n and Prospecting personality traits. These flexible thinkers enjoy taking \n an unconventional approach to many aspects of life. They often seek out \n unlikely paths, mixing willingness to experiment with personal creativity.")
	print("Famous personalities whose traits you resemble are: \n Albert Einstein, Bill Gates, Issac Newton, Blaise Pascal.")

elif mbti == 'ISFJ':
	str1 = '" The Nurturer "'
	print(mbti +' - '+ str1)
	print("\nThe Nurturer :-")
	print(" A Defender (ISFJ) is someone with the Introverted, Observant,\n Feeling, and Judging personality traits. These people tend to be warm and \n unassuming in their own steady way. They’re efficient and responsible, giving \n careful attention to practical details in their daily lives.")
	print("Famous personalities whose traits you resemble are: \n Queen Elizabeth II, Aretha Franklin, Vin Diesel.")

elif mbti == 'ENTP':
	str1 = '" The Visionary "'
	print(mbti +' - '+ str1)
	print("\nThe Visionary :-")
	print(" A Debater (ENTP) is a person with the Extraverted, Intuitive, \n Thinking, and Prospecting personality traits. They tend to be bold and creative,\n  deconstructing and rebuilding ideas with great mental agility. They pursue \n their goals vigorously despite any resistance they might encounter.. ")
	print("Famous personalities whose traits you resemble are: \n Thomas Edison, Mark Twain, Tom Hanks .")

else :
	str1 = '" The Composer "'
	print(mbti +' - '+ str1)
	print("\nThe Composer :-")
	print(" An Adventurer (ISFP) is a person with the Introverted, Observant, \n Feeling, and Prospecting personality traits. They tend to have open minds,\n approaching life, new experiences, and people with grounded warmth. \n Their ability to stay in the moment helps them uncover exciting potentials.")
	print("Famous personalities whose traits you resemble are: \n Michael Jackson, Jeon jungkook,Frida Kahlo .")

print('===============================================================================>')

