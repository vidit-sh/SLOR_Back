
#oggpnosn 
#hkhr

#geotag and correct location in Accident dataframe 

import pandas as pd 
import urllib, urllib2
from urllib import urlencode
from urllib2 import urlopen
import json 
import nltk
import numpy
import jellyfish as jf

#performs google location search with the given key 
#input: 
# searchTerm: query string 
# key: api key of app engine 
def searchLocation(searchTerm, key):
    searchTerm = searchTerm.lower().replace("nr", "").replace("at", "") 
    if "ponda" not in searchTerm:
        if "goa" in searchTerm:
            searchTerm = searchTerm.replace("goa", "ponda, goa")
        else:
            searchTerm += "ponda, goa"
        
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json?%s&key=%s"
    query = {"query": searchTerm}
    response = urlopen(url%(urlencode(query), key))
    if response.code == 200: #is succesull
        result = json.loads(response.read())
        if result["results"] != []:
            return (result["results"][0]['geometry']['location']['lat'], result["results"][0]['geometry']['location']['lng'])
        else:
            print "SLOR: No results found for %s"%searchTerm
    else:
        print "SLOR: URL access error, response code %d"%response.code
    return None



# map = []
# i = 0
# keys = """ AIzaSyDxgBfrOKZnyMyhP1WknSn4ygQqPRpYAxM


# AIzaSyAtYWGAqcWRAvr368Z-R-knb9kRSXBlWRY


# AIzaSyBAeDv6z5t-GE6VRuHUnxEEf-_z0WiEJ7k


# AIzaSyBAXwAoQGr_yVj3Bd2YHBhZBkJDQZPRkww



# AIzaSyBVpU7rpmxMXi4OpCyi1jkH-sjg9YOXdKo


# AIzaSyCq4fj7R5WawofuyvBv6QZfgcdd2iy-RnM


# AIzaSyDu2d1Eymk_yn5P1bI_o4S_nDwSOnu8EeU



# AIzaSyAvwCPWg9wtPDbAe6uyLzT4bUa8K8zEHi8


# AIzaSyCUdzgVNXqubtyQUJIICoH2-gpANNKDll0



# AIzaSyBdm_wLaI_msk6itbyfythDH2p8uXTpfWs



# AIzaSyAhDJW7nDX9QK1f_PhTsFAGTbSQAbJYcug

# AIzaSyCAOYeoXxJXAgcBDbdAw8n-UULhncmSiu0""".split()

# for place in places:
#     print place
#     result = searchLocation(place, keys[i%12])
#     map.append([place, result])
#     i+=1


# i = 360
# map.append(["NAN", None])
# keys = """ AIzaSyDxgBfrOKZnyMyhP1WknSn4ygQqPRpYAxM


# AIzaSyAtYWGAqcWRAvr368Z-R-knb9kRSXBlWRY


# AIzaSyBAeDv6z5t-GE6VRuHUnxEEf-_z0WiEJ7k


# AIzaSyBAXwAoQGr_yVj3Bd2YHBhZBkJDQZPRkww



# AIzaSyBVpU7rpmxMXi4OpCyi1jkH-sjg9YOXdKo


# AIzaSyCq4fj7R5WawofuyvBv6QZfgcdd2iy-RnM


# AIzaSyDu2d1Eymk_yn5P1bI_o4S_nDwSOnu8EeU



# AIzaSyAvwCPWg9wtPDbAe6uyLzT4bUa8K8zEHi8


# AIzaSyCUdzgVNXqubtyQUJIICoH2-gpANNKDll0



# AIzaSyBdm_wLaI_msk6itbyfythDH2p8uXTpfWs



# AIzaSyAhDJW7nDX9QK1f_PhTsFAGTbSQAbJYcug

# AIzaSyCAOYeoXxJXAgcBDbdAw8n-UULhncmSiu0""".split()

# for place in places[360:]:
#     print place
#     result = searchLocation(place, keys[i%12])
#     map.append([place, result])
#     i+=1



# i = 710
# map.append(["NAN", None])
# keys = """ AIzaSyDxgBfrOKZnyMyhP1WknSn4ygQqPRpYAxM


# AIzaSyAtYWGAqcWRAvr368Z-R-knb9kRSXBlWRY


# AIzaSyBAeDv6z5t-GE6VRuHUnxEEf-_z0WiEJ7k


# AIzaSyBAXwAoQGr_yVj3Bd2YHBhZBkJDQZPRkww



# AIzaSyBVpU7rpmxMXi4OpCyi1jkH-sjg9YOXdKo


# AIzaSyCq4fj7R5WawofuyvBv6QZfgcdd2iy-RnM


# AIzaSyDu2d1Eymk_yn5P1bI_o4S_nDwSOnu8EeU



# AIzaSyAvwCPWg9wtPDbAe6uyLzT4bUa8K8zEHi8


# AIzaSyCUdzgVNXqubtyQUJIICoH2-gpANNKDll0



# AIzaSyBdm_wLaI_msk6itbyfythDH2p8uXTpfWs



# AIzaSyAhDJW7nDX9QK1f_PhTsFAGTbSQAbJYcug

# AIzaSyCAOYeoXxJXAgcBDbdAw8n-UULhncmSiu0""".split()

# for place in places[710:]:
#     print place
#     result = searchLocation(place, keys[i%12])
#     map.append([place, result])
#     i+=1

# undetermined = []
# for location in map:
#     if location[1] == None:
#         undetermined.append(location[0])

# len(undetermined)*100.0/len(accident)

# undetermined


#convert query string into geo coordinates 
# input: 
# searchTerm: search string 
# key: api key for location api 
def getGeocode(searchTerm, key):
    searchTerm = searchTerm.lower().replace("nr", "").replace("at", "")     
    url = "https://maps.googleapis.com/maps/api/geocode/json?%s&key=%s"
    query = {"address": searchTerm}
    response = urlopen(url%(urlencode(query), key))
    if response.code == 200: #is succesull
        result = json.loads(response.read())
        if result["results"]:
            return (result["results"][0]["geometry"]['location']['lat'], result["results"][0]["geometry"]['location']['lng'])
    else:
        raise Exception, "SLOR: Network issues"

result = geocode('ponda goa', "AIzaSyCM7OIQkmeYJWDE8P-D3smfYF7uKNwKVPU")


# accident = accident.drop(710, axis=0)
# accident = accident.drop(360, axis=0)
# accident["S.NO"] = range(1, len(accident)+1)

# len(accident)

# accident = pd.read_csv("Accident2012-2014_V3.csv")


# places = list(accident["Place of Accident "])

# places = [place.lower() for place in places]

# places = [place.replace(',', '').replace('.', '').replace('-', '') for place in places]

# vocabulary = ""
# for place in places:
#     vocabulary += " "+place

# vocabulary = vocabulary.split()

# len(vocabulary)*1.0/len(set(vocabulary))

import nltk #natural language toolkit 

# vocabulary = nltk.FreqDist(vocabulary)

#basis string distance algorithm 
 def match(str1, str2):    
    i =0
    j =0
    count =0
    match =0
    for i in range(len(str1)):
        for j in range(i, len(str1)):
            if str1[i:j+1] in str2:  
                match += 1
            count += 1       
    return match*1.0/count        


# words = list(vocabulary)

# typo = 'fermaguudi'

#calculating best replacement of word based on basic distance measure 
def guess(typo, words):
    bestGuess =""
    bestScore = 0
    for word in words:
        score = match(typo, word)
        if score>bestScore and score != 1.0:
            bestGuess = word
            bestScore = score
    return bestGuess        

def proximity(typo, words):
    prox = {}
    for word in words:
        score = refMatch(typo, word)
        prox[word] = score 
    return prox 

def smartProximity(typo, words, threshold):
    prox = {}
    for word in words:
        score = refMatch(typo, word)
        if score>threshold:
            prox[word] = score 
    return prox 

smartProximity(words[567], words, .55)

def refMatch(str1, str2):
    score1 = match(str1, str2)
    score2 = match(str2, str1)    
    return (score1+score2)/2

clusters = []
for word in words[:50]:
    cluster = smartProximity(word, words, .55)
    clusters.append(cluster)
    

smartProximity("ponda", words, .55)

def encodeSoundex(str1):
    str1 = str1.lower()
    if len(str1) != 0:
        code = str1[0]
        for s in str1[1:]:
            if s=='a' or s=='e' or s=='i' or s=='o' or s=='u' or s=='y' or s=='w':
                code+="0"
            elif s=='b' or s=='f' or s=='p' or s=='v':
                code += "1"
            elif  s=='c' or s=='g' or s=='j' or s=='k' or s=='q' or s=='s' or s=='x' or s=='z':
                code += "2" 
            elif s=='d' or s=='t':
                code += "3"
            elif s=='l':
                code += "4"    
            elif s=='m' or s=='n':
                code += "5"
            elif s=='r':
                code += "6"            
        code = code.replace("0", "")  
#         if len(code) > 4:
#             return code[:4]
#         else:
#             code += "0000"
#             return code[:4]
#         return code      
        return code
    else:
        return ""

def soundex(str1, str2):
    enc1 = encodeSoundex(str1)
    enc2 = encodeSoundex(str2)
    if enc1 == enc2:
        return True
    else:
        return False

def proximitySoundex(str1, words):
    encoding = encodeSoundex(str1)
    results = []
    for word in words:
        if encoding == encodeSoundex(word):
            results.append(word)
    return results        

proximitySoundex(words[0], words)

def suggestSoundex(word, words):
    maxFreq = 0 
    suggestion = word
    result = proximitySoundex(word, words)
    for alter in result:
        freq = vocabulary[alter]
        if freq>maxFreq:
            maxFreq = freq
            suggestion = alter
    return suggestion    

suggestSoundex("college", words)

def spellCorrect(placeString, words):
    places = placeString.split(" ")
    suggestions = []
    for place in places:
        place = place.replace(",", "").replace(".", "").replace("-", "")
        suggestions.append(suggestSoundex(place, words))
    correctAddress = ""
    for suggestion in suggestions:
        correctAddress += suggestion + " "
    return correctAddress    



for i in range(40, 60):
    print undetermined[i], "---->", spellCorrect(undetermined[i], words)

for word in words[:10]:
    print proximitySoundex(word, words)

vehicles = accident["Type of vehicle involved "]

focus = []
for i in range(1105):
    focus.append(vehicles[i].split(" ")[:3])
    

highFocus = [f[0].lower() for f in focus]

dummy = ""
for word in highFocus:
    dummy += word + " "
    

import nltk 
fdist = nltk.FreqDist(highFocus)

fdist

fdist[""]

target = [len(place.split()) for place in places]

import numpy as np
import scipy as sp
np.mean(target)


spell_places = [spellCorrect(place, words) for place in places]

spell_places[1]

for i in range(10):
    print places[i], spell_places[i]

spell_vocab = []
for place in spell_places:
    spell_vocab.extend(place.split(" "))
spell_vocab = nltk.FreqDist(spell_vocab)



print len(spell_vocab), len(vocabulary)

def placesSimilarity(place1, place2):
    vocab1 = place1.split()
    vocab2 = place2.split()
    score = 0
    for word in vocab1:
        if word in vocab2:
            score+=1
    score1 = score*1.0/len(vocab1)
    score = 0
    for word in vocab2:
        if word in vocab1:
            score+=1
    score2 = score*1.0/len(vocab2)
    return max(score1, score2)

def placesSimilarTo(target, f_places, threshold):
    similarTo = []
    for place in f_places:
        score = placesSimilarity(target, place)
        if score > threshold:
            similarTo.append(place)
    return similarTo        

placesSimilarTo(spell_places[1], spell_places, .50)

spell_vocab["bhoma"]

from sklearn.cluster import DBSCAN 


dbscan = DBSCAN(eps=.5, metric=placesSimilarity)


result = dbscan.fit([spell_places, spell_places])

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


##############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)
print X
##############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print db
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

##############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)


def testStringMatching(algo, ts_place, ts_places, threshold):
    result = []
    for place in ts_places:
        score = algo(place, ts_place)
        if score< threshold:
            result.append(place)
    return result         
    

for i in range(20):
    print testStringMatching(levProDistance, places[i], places, 3)

distanceMatrix = []
for place1 in places:
    matrix = []
    for place2 in places:
        matrix.append(levProDistance(place1, place2))
    distanceMatrix.append(matrix)    

from sklearn.cluster import DBSCAN
import numpy as np
distanceMatrixFit = np.array(distanceMatrix)

db = DBSCAN( eps=1, metric="precomputed").fit(distanceMatrixFit)


len(set(db.labels_))

clusters = [[] for i in range(len(set(db.labels_)))]
i=0
for label in db.labels_:
    clusters[label].append(places[i])
    i+=1


for cluster in clusters:
    print cluster[0], "--->", len(cluster)

clusters[1]

testCluster = ['bhoma nr vikas automobile service', 'nr panchayat bhoma ', 'bhoma nr vikas auto mobiles ', 'bhoma nr sai service ', 'nr vikal automobile bhoma ']

def levProDistance(str1, str2):
    c1 = str1.split(" ")
    c2 = str2.split(" ")
    score = 0
    for word in c1:
        levScore = [jf.levenshtein_distance(word , alter) for alter in c2]
        score += min(levScore)
    score2 =0    
    for word in c2:
        levScore = [jf.levenshtein_distance(word , alter) for alter in c1]
        score2 += min(levScore)    
    return ((score2*1.0/len(c2))+(score*1.0/len(c1)))/2


levProDistance('dharbandora nr sugar factory', "nr factory dharbandora")

import jellyfish as jf 
import matplotlib.pyplot as plot 
import numpy as np


def testApproach(algo, places,threshold=.7, i=0, j=5):
    testPlaces = places[i:j]
    result = []
    for testPlace in testPlaces: 
        cluster = []
        for place in places:
            if(algo(testPlace, place) > threshold):
                cluster.append(place)
        result.append(cluster)        
    return result
def getDistanceMatrix(algo, places):
    results = []
    for place1 in places:
        result = []
        for place2 in places:
            result.append(algo(place1, place2))
        results.append(result)
    return np.array(results)    
def interpretLabels(labels, places):
    clusters = [[] for label in set(labels)]
    for i in range(len(labels)):
        clusters[labels[i]].append(places[i])
    return clusters   


testApproach(jf.jaro_distance, places, threshold=.8)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=200)


X = getDistanceMatrix(jf.jaro_distance, places)

kmeans.fit(X)

kmeans.labels_

clusters = interpretLabels(kmeans.labels_, places)

lenClusters = [len(cluster) for cluster in clusters]

plot.hist(lenClusters)
plot.show(
)

screenCluster = [cluster for cluster in clusters if len(cluster)>11]

len(screenCluster)

keys = """slor1  AIzaSyBk21CHntg-QcGbm4l4wq5L0hhV5AKNbGA

slor2  AIzaSyD06_vJcLLFru_5ryM2lr6ii-508zGvT80

slor3  AIzaSyD6-2mfJWRMhRpeQxhUvNn4nujk7ELe5KM

slor4  AIzaSyCDMOfddevCL75YZyg6s9Qvy6fDsFvryGA

slor5  AIzaSyDINs05bqmcq-4yaSpEZZd-rSQbWPpBaew

slor6  AIzaSyCq_uF7ABsC2C6iqLQ-udUknuSrHy8JHtw

slor7 AIzaSyDnOKFoS2V8ywU3N-UfwM6-IkK-IpLFSTk

slor8 AIzaSyDUF1-QPIlTLCS63MYRwlmeNh2bsqUT0vs

slor9 AIzaSyAjcsR6wvMAoW7VLGiy_fYHdNy5xXzxXVY

slor10 AIzaSyDUFgRSWOQ0gtQMnGA0tuQUfYNSH9nSKGs""".split("\n")
keys = [key.split(" ") for key in keys if key!='']
keys = [key[-1] for key in keys]
keys

results = []
def getGeocodeForCluster(clusters, keys):
    i=0
    for cluster in clusters:
        result = {}
        for place in cluster:
            i+=1
            geocode = getGeocode(place, keys[i%10])
            if geocode:
                result["clusterName"] = place
                result["lat"] = geocode[0]
                result["lng"] = geocode[1]                
                break
        result["count"] = len(cluster)
        print result
        results.append(result)
        

from random import randrange
sampleData = []
for i in range(200):
    data = {"lat": randrange(100, 200), "lng": randrange(100, 200), "count": randrange(1, 20)}
    sampleData.append(data)
json.dump(sampleData, open("sampleData.json", "w"))    
    

results = r"""{'clusterName': 'nr sugar factory dharbandora goa ', 'lat': 15.3841266, 'lng': 74.1181234, 'count': 9}
{'clusterName': 'apewhal priol keri ponda goa ', 'lat': 15.4633975, 'lng': 73.9976946, 'count': 11}
{'count': 18}
{'count': 2}
{'count': 1}
{'clusterName': 'nr kirti hotel bethora road ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 9}
{'clusterName': 'patnem gaunem ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 3}
{'clusterName': 'undir bandora ponda ', 'lat': 15.417251, 'lng': 73.9679378, 'count': 3}
{'clusterName': 'nr st anthonys church ponda', 'lat': 15.2959572, 'lng': 73.9769018, 'count': 5}
{'clusterName': 'tamsurle khandola ', 'lat': 15.5219296, 'lng': 73.9648694, 'count': 7}
{'count': 1}
{'clusterName': 'byepass road dhavli ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 6}
{'count': 10}
{'clusterName': 'nr savitrihall curti ponda ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 7}
{'clusterName': 'banastarim junction', 'lat': 15.4899117, 'lng': 73.95757909999999, 'count': 10}
{'count': 7}
{'count': 3}
{'count': 7}
{'count': 9}
{'clusterName': 'nr saibaba temple borim ponda ', 'lat': 15.3597368, 'lng': 74.0008495, 'count': 8}
{'count': 6}
{'count': 3}
{'clusterName': 'dhavli ponda goa ', 'lat': 15.3876702, 'lng': 74.0013673, 'count': 5}
{'clusterName': 'bethki borim ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 11}
{'clusterName': 'silvanagar ponda goa ', 'lat': 15.4030585, 'lng': 74.0175265, 'count': 6}
{'clusterName': 'dharbandora ', 'lat': 15.3841266, 'lng': 74.1181234, 'count': 17}
{'clusterName': 'vazem shiroda ', 'lat': 15.3346783, 'lng': 74.0145886, 'count': 8}
{'clusterName': 'marcela market ', 'lat': 37.7479513, 'lng': -122.4609052, 'count': 5}
{'clusterName': 'nr idc kundaim ponda goa ', 'lat': 15.3735147, 'lng': 73.9277905, 'count': 5}
{'count': 8}
{'clusterName': 'bye pass road nr  hrc transport  curti ponda ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 3}
{'count': 4}
{'clusterName': 'adan madkai', 'lat': 15.423048, 'lng': 73.951131, 'count': 6}
{'clusterName': 'nr ponda municipalty ', 'lat': 12.3171471, 'lng': 122.0817049, 'count': 4}
{'clusterName': 'nr opp junction khandepar ponda goa ', 'lat': 15.4781354, 'lng': 74.017573, 'count': 12}
{'count': 2}
{'count': 10}
{'count': 2}
{'clusterName': 'nr amigos hotel curti ponda goa ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 12}
{'clusterName': 'fermagudi ', 'lat': 15.4126756, 'lng': 73.9892747, 'count': 5}
{'count': 9}
{'count': 9}
{'clusterName': 'chirputwada ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 10}
{'clusterName': 'nr gvm collage farmagudi ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 9}
{'clusterName': 'nr mahalaxmi bandora ', 'lat': 15.4060041, 'lng': 73.9798522, 'count': 9}
{'clusterName': 'ganganagar shrioda ', 'lat': 29.9166667, 'lng': 73.88333329999999, 'count': 8}
{'clusterName': 'borim circle', 'lat': -24.9862918, 'lng': -53.45376510000001, 'count': 6}
{'clusterName': 'nr rajivkala mandir bethoda ', 'lat': 15.3992164, 'lng': 74.0186183, 'count': 10}
{'count': 10}
{'clusterName': 'nr vithoba temple upperbazar ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 17}
{'clusterName': 'old bus stand ponda goa', 'lat': 15.4006658, 'lng': 74.00417449999999, 'count': 7}
{'count': 2}
{'clusterName': 'nanus usgao goa ', 'lat': 15.4489009, 'lng': 74.0711359, 'count': 5}
{'clusterName': 'chirputem bandora ponda goa ', 'lat': 15.4168785, 'lng': 73.9990423, 'count': 7}
{'clusterName': 'usgao circle goa ', 'lat': 15.5092454, 'lng': 74.0470596, 'count': 5}
{'count': 5}
{'clusterName': 'khandepar bridge', 'lat': 15.4234721, 'lng': 74.0501046, 'count': 4}
{'clusterName': 'palwada usgao ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 8}
{'count': 3}
{'clusterName': 'shiroda dabolim ', 'lat': 15.3292, 'lng': 74.02709999999999, 'count': 4}
{'count': 8}
{'clusterName': 'ganjem nanus usgao ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 9}
{'count': 3}
{'clusterName': 'bhoma goa', 'lat': 15.4778442, 'lng': 73.96243270000001, 'count': 5}
{'clusterName': 'varkhandem ponda goa ', 'lat': 15.4045409, 'lng': 74.0032496, 'count': 7}
{'count': 3}
{'clusterName': 'nr civil court ponda', 'lat': 15.3929079, 'lng': 74.001106, 'count': 2}
{'clusterName': 'marcela bus stand ', 'lat': 15.5158679, 'lng': 73.9614225, 'count': 4}
{'clusterName': 'kundaim goa ', 'lat': 15.4874553, 'lng': 73.977126, 'count': 6}
{'count': 6}
{'clusterName': 'bypassroad tisk usgao goa ', 'lat': 15.4148928, 'lng': 74.0821496, 'count': 7}
{'count': 2}
{'clusterName': 'kundaim ', 'lat': 15.4491866, 'lng': 73.9540789, 'count': 2}
{'clusterName': 'sadar ponda', 'lat': 15.3975305, 'lng': 74.0026639, 'count': 4}
{'clusterName': 'tivrem marcela ', 'lat': 15.5125006, 'lng': 73.9691679, 'count': 3}
{'clusterName': 'nr mrf factory usgao goa ', 'lat': 15.4489009, 'lng': 74.0711359, 'count': 8}
{'clusterName': 'nr maruthi temple varkhandem ponda', 'lat': 15.4045409, 'lng': 74.0032496, 'count': 9}
{'clusterName': 'curti ponda goa  ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 5}
{'clusterName': 'bhoma nr sateri temple', 'lat': 29.9524917, 'lng': 74.3812353, 'count': 7}
{'clusterName': 'dharabandora nr panchayat ', 'lat': 15.3917649, 'lng': 74.1221407, 'count': 6}
{'count': 6}
{'clusterName': 'ganganagar ponda ', 'lat': 29.9166667, 'lng': 73.88333329999999, 'count': 5}
{'count': 3}
{'clusterName': 'usgao tisk ', 'lat': 15.4148928, 'lng': 74.0821496, 'count': 4}
{'count': 5}
{'clusterName': 'nr pancyayat borim ', 'lat': 15.3761526, 'lng': 73.97595489999999, 'count': 7}
{'count': 7}
{'clusterName': 'fermagudi ponda ', 'lat': 15.4134164, 'lng': 73.9884165, 'count': 6}
{'count': 1}
{'clusterName': 'dhavali ponda', 'lat': 15.3876702, 'lng': 74.0013673, 'count': 6}
{'clusterName': 'nr rto office ponda ', 'lat': 15.2855823, 'lng': 73.9573431, 'count': 11}
{'clusterName': 'bharbhat shiroda goa ', 'lat': 15.3292, 'lng': 74.02709999999999, 'count': 5}
{'clusterName': 'borim nr panchayat ', 'lat': 15.3761526, 'lng': 73.97595489999999, 'count': 3}
{'clusterName': 'veling mardol ', 'lat': 15.4413483, 'lng': 73.974064, 'count': 5}
{'count': 6}
{'count': 5}
{'clusterName': 'khandola nr collage ', 'lat': 15.5214468, 'lng': 73.9578725, 'count': 4}
{'clusterName': 'bhoma ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 6}
{'count': 7}
{'clusterName': 'bethora ponda', 'lat': 15.3866625, 'lng': 74.03296879999999, 'count': 3}
{'count': 5}
{'clusterName': 'jacy nagar ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 2}
{'count': 3}
{'count': 3}
{'clusterName': 'cuncalim slope mangeshi goa ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 4}
{'count': 3}
{'clusterName': 'borim nr briged ', 'lat': 15.349438, 'lng': 74.00289769999999, 'count': 2}
{'clusterName': 'dabal road dharbandora ', 'lat': 15.3768124, 'lng': 74.1266517, 'count': 5}
{'count': 2}
{'count': 4}
{'clusterName': 'upper bazar nr jamma masjid ', 'lat': 23.3718688, 'lng': 85.32205669999999, 'count': 6}
{'clusterName': 'banastarim goa ', 'lat': 15.4892107, 'lng': 73.96243270000001, 'count': 3}
{'clusterName': 'kelbai curti ponda ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 5}
{'count': 3}
{'count': 1}
{'count': 5}
{'clusterName': 'nr shiroda ferry ', 'lat': 15.3118654, 'lng': 74.0121401, 'count': 6}
{'clusterName': 'mesta wada curti ponda ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 10}
{'clusterName': 'appewal ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 6}
{'clusterName': 'ballawada kundaim goa ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 6}
{'clusterName': 'borim ponda goa ', 'lat': 15.3657309, 'lng': 74.0180379, 'count': 4}
{'clusterName': 'bhawankarkundaim goa ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 5}
{'clusterName': 'conem priol ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 7}
{'clusterName': 'nr sungrace garden ', 'lat': 15.391476, 'lng': 74.02464719999999, 'count': 6}
{'clusterName': 'nr panchayat dharbandora ', 'lat': 15.3917649, 'lng': 74.1221407, 'count': 4}
{'count': 3}
{'clusterName': 'karai shiroda ', 'lat': 15.3275795, 'lng': 74.0145886, 'count': 4}
{'clusterName': 'at karmane  keri ponda goa ', 'lat': 15.4633975, 'lng': 73.9976946, 'count': 12}
{'clusterName': 'nr sai service  centre curti ponda goa ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 8}
{'clusterName': 'nr anant devastan savoiverem  ponda ', 'lat': 15.477887, 'lng': 74.0188891, 'count': 10}
{'clusterName': 'mageshi ', 'lat': 35.9667922, 'lng': 139.5068266, 'count': 1}
{'clusterName': 'bethoda nr sungrace garden ', 'lat': 15.391476, 'lng': 74.02464719999999, 'count': 8}
{'clusterName': 'at curti ponda nr amigos hotel ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 13}
{'clusterName': 'nr dada vidhya chowk ponda ', 'lat': 15.3996521, 'lng': 74.00451509999999, 'count': 9}
{'count': 3}
{'clusterName': 'fermagudi cicrcle ', 'lat': 15.4134164, 'lng': 73.9884165, 'count': 2}
{'count': 8}
{'count': 4}
{'count': 2}
{'count': 6}
{'count': 8}
{'clusterName': 'at ktc bus stand', 'lat': 15.2874714, 'lng': 73.9561878, 'count': 3}
{'clusterName': 'nagzar wada bhoma ponda  goa ', 'lat': 15.4, 'lng': 74.02, 'count': 4}
{'clusterName': 'nr goa bagayatdar society ponda ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 14}
{'clusterName': 'madkai tonka ', 'lat': 15.423048, 'lng': 73.951131, 'count': 3}
{'clusterName': 'manaswada kundaim ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 5}
{'clusterName': 'tisk usgaon', 'lat': 15.4148928, 'lng': 74.0821496, 'count': 2}
{'clusterName': 'mhalwada madkai ', 'lat': 15.423048, 'lng': 73.951131, 'count': 5}
{'count': 2}
{'count': 3}
{'clusterName': 'varkhande ponda ', 'lat': 15.404601, 'lng': 74.0046727, 'count': 7}
{'clusterName': 'nr gps par usgao goa ', 'lat': 15.4489009, 'lng': 74.0711359, 'count': 4}
{'clusterName': 'marcela goa ', 'lat': 15.4587861, 'lng': 74.01225029999999, 'count': 2}
{'count': 3}
{'clusterName': 'nr ktc bus stand ponda goa', 'lat': 15.2874714, 'lng': 73.9561878, 'count': 8}
{'clusterName': 'muslim wada bhoma goa ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 3}
{'clusterName': 'paniwada borim ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 4}
{'clusterName': 'dharbandora ponda goa ', 'lat': 15.3933206, 'lng': 74.12567109999999, 'count': 6}
{'count': 3}
{'clusterName': 'nageshi ponda  ', 'lat': 15.4067091, 'lng': 73.9837373, 'count': 5}
{'clusterName': 'nr sateri temple bhoma goa ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 9}
{'clusterName': 'tisk ponda ', 'lat': 15.392889, 'lng': 74.0017116, 'count': 3}
{'clusterName': 'nr i d hospital ', 'lat': 7.0233694, 'lng': 79.9271857, 'count': 7}
{'count': 1}
{'clusterName': 'nr suzuki showroom', 'lat': 8.998414, 'lng': 76.76482539999999, 'count': 6}
{'clusterName': 'nr madkai idc', 'lat': 15.423048, 'lng': 73.951131, 'count': 6}
{'count': 2}
{'clusterName': 'nr borim bridge', 'lat': 15.349438, 'lng': 74.00289769999999, 'count': 3}
{'count': 1}
{'count': 2}
{'clusterName': 'shantinagar ponda', 'lat': 15.3976023, 'lng': 74.0109161, 'count': 4}
{'count': 3}
{'count': 4}
{'clusterName': 'shiroda ferry', 'lat': 15.3118654, 'lng': 74.0121401, 'count': 1}
{'count': 7}
{'clusterName': 'velling nr bridge ', 'lat': 56.0689113, 'lng': 8.2793808, 'count': 2}
{'clusterName': 'at kalimati bhoma goa ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 4}
{'count': 1}
{'clusterName': 'curti ponda bye pass', 'lat': 15.4021855, 'lng': 74.0277404, 'count': 9}
{'clusterName': 'at ponda  nr maruti temple ', 'lat': 15.4119141, 'lng': 74.0128165, 'count': 7}
{'clusterName': 'nr sahakari farm mestawada curti ponda ', 'lat': 15.4136394, 'lng': 74.0208662, 'count': 13}
{'clusterName': 'upper bazar ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 5}
{'count': 3}
{'clusterName': 'tisk usgao goa ', 'lat': 15.4148928, 'lng': 74.0821496, 'count': 3}
{'count': 4}
{'clusterName': 'borim circle borimponda goa ', 'lat': 15.3516568, 'lng': 74.0035667, 'count': 4}
{'count': 4}
{'clusterName': 'pratapnagar usgao ', 'lat': 15.5553665, 'lng': 74.0288144, 'count': 4}
{'count': 1}
{'clusterName': 'nr raikarjwellersponda  goa ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 6}
{'clusterName': 'nr vaishali hotel kundaim ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 16}
{'clusterName': 'kalimathi bhoma ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 4}
{'count': 4}
{'clusterName': 'marvasadausgao goa ', 'lat': 15.2993265, 'lng': 74.12399599999999, 'count': 3}
{'count': 2}
{'count': 7}
{'count': 4}
{'clusterName': 'ganjem usgao goa ', 'lat': 15.4665014, 'lng': 74.08575429999999, 'count': 3}
{'count': 3}
{'clusterName': 'muslimwada bhoma ponda goa ', 'lat': 15.4, 'lng': 74.02, 'count': 5} """.split("\n")

import ast
results = [ast.literal_eval(result) for result in results]


sampleData = []
for result in results:
    if result.get("lat"):
        sampleData.append({"lat": result["lat"], "lng": result["lng"], "count": result["count"]})
        


json.dump(sampleData, open("heatmap.json", "w"))

len(sampleData)    

chk



