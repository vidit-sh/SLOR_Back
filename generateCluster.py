#oggpnosn 
#hkhr

#generating cluster 

import sklearn as sk 
import numpy as np
import os 
from sklearn.cluster import KMeans
import json 

data = json.load(open("heatmap.json"))

latlong = [(str(item["lat"]), str(item['lng'])) for item in data]
X = np.array([[item["lat"], item['lng']] for item in data])

def interpretLabels(labels, places):
    clusters = [[] for label in set(labels)]
    for i in range(len(labels)):
        clusters[labels[i]].append(places[i])
    return clusters   


# X = np.load("distanceMatrixV2.npy")
# X = np.nan_to_num(X)

kmeans = KMeans(n_clusters=int(os.sys.argv[1])) 

kmeans.fit(X)

locations = [] 
for i in range(int(os.sys.argv[1])):
	locations.append({"lat": kmeans.cluster_centers_[i][0], "lng": kmeans.cluster_centers_[i][1]})

# clusters = interpretLabels(kmeans.labels_, latlong) 

# fob = open("visualizeGeocoordinate.html")
text = """ 
<!DOCTYPE html>
<html>
  <head>
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
    <meta charset="utf-8">
    <title>Simple icons</title>
    <style>
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      #map {
        height: 100%;
      }
    </style>
  </head>
  <body>
    <div id="map"></div>
    <script>

      // This example adds a marker to indicate the position of Bondi Beach in Sydney,
      // Australia.
      var ambulances = BAZINGA ;
      function initMap() {
        var map = new google.maps.Map(document.getElementById('map'), {
          zoom: 12,
          center: ambulances[0]
        });

        // var image = 'https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png';
        var i;
        for(i=0; i<ambulances.length; i++){
          var beachMarker = new google.maps.Marker({
            position: ambulances[i],
            map: map
          });
        }
      }
    </script>
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key=AIzaSyD-i-lZjkM5lSE_cf6fn_EQN4BP_M98H5E&callback=initMap">
    </script>
  </body>
</html>
"""

text = text.replace("BAZINGA", str(locations))
fob = open("visualizeGeocoordinate.html", "w")
fob.write(text)
fob.close()

from subprocess import call 
call(["open", "visualizeGeocoordinate.html"])

# parseData= []
# for cluster in clusters:
# 	parseData.append({"lat":cluster[0][0], "lng":cluster[0][1]})
# print parseData	


