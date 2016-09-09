#oggpnosn 
#hkhr 

#geocoding a the clusters of places extracted out of accident_v3

import pandas as pd 
import urllib, urllib2
from urllib import urlencode
from urllib2 import urlopen
import json 
import nltk
import numpy
import jellyfish as jf
import numpy as np
from sklearn.cluster import KMeans



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


kmeans = KMeans(n_clusters=200)


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


accident = pd.read_csv("Accident2012-2014_V3.csv")
places = list(accident["Place of Accident "])
places = [place.lower() for place in places]
places = [place.replace(',', '').replace('.', '').replace('-', '') for place in places]

X = getDistanceMatrix(jf.jaro_distance, places)
kmeans.fit(X)
clusters = interpretLabels(kmeans.labels_, places)

results = []
getGeocodeForCluster(clusters, keys)

