#oggpnosn 
#hkhr 

#extracting distance out of distance api and storing it as an np array

import json 
import pandas as pd
from urllib2 import urlopen 
from time import sleep
import numpy as np
from urllib import urlencode
import numpy as np 

keys = ["AIzaSyCM7OIQkmeYJWDE8P-D3smfYF7uKNwKVPU", "AIzaSyBJYB4yJz5TCo23vkDFfEvW9wL1ENeXUM0", "AIzaSyDr6Cv1k13Ul1Dt21spv0VPNlQXlIh6_MA", 
        "AIzaSyCGSEsMTcBLeZ0KLkqezgfev96ScZv-rqk", "AIzaSyCAOYeoXxJXAgcBDbdAw8n-UULhncmSiu0", "AIzaSyBGIk3V_6r5uz5eb22C9c4stTv-TrG6scY"]

fob = open("heatmap.json")
data = json.load(fob)

latlong = [(str(item["lat"]), str(item['lng'])) for item in data]

def calculateDistance(geo1, locations,key):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=%s&destinations=%s&key=%s"
    parameter = {}
    parameter["origins"] = ','.join(geo1)
    parameter["destinations"] = '|'.join([','.join(location) for location in locations])
    parameter["key"] = key
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?" + urlencode(parameter)
    resp = urlopen(url)
    if resp.code == 200:
        result =json.loads(resp.read())
        if result["rows"]:
            results = result["rows"][0]["elements"]
            distances = []
            for result in results:
                if result.get("distance"):
                    distances.append(result["distance"]["value"])
                else:
                    distances.append(np.inf)
            return distances
    else:
        raise Exception, "Network issues, its throwing a %s error:%s"%(resp.code, resp.read())
        
matrix = []
for i in range(70, 128):
    resp =calculateDistance(latlong[i], latlong[70:128], keys[i%5])
    print resp
    sleep(10)
    matrix.append(resp)                
matrix = np.array(matrix)    
np.save("distanceMatrix30.npy", matrix)
