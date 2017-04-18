# Author: Ruixuan Zhang

import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import metrics
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from haversine import haversine
from datetime import datetime
from datetime import timedelta
import warnings
import operator

warnings.filterwarnings("ignore")

# Data source: http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml
# dimension of raw data (1576393, 21)
df1 = pd.read_csv("green_tripdata_2016-03.csv",parse_dates=True)

kms_per_radian = 6371.0

# Uses DBSCAN to assign each observation (latitude and longitude) into a cluster, or lable the outliers as -1
# this function returns a label for each observation, which is the cluster that this observation belongs to
def DBSCAN_cluster(data, distance):
    kms_per_radian = 6371.0
    epsilon = distance / kms_per_radian
    algo = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
    algo.fit(np.radians(data))
    labels = algo.labels_
    print "number of clusters : %s" % len(np.unique(labels))
    return labels

# given labels for all the observations, returns the top five clusters
# and the percentage of the trips belongs to these top five clusters
def top_five_cluster(labels):
    clustering_result = Counter(labels)
    # This algorithm allows some observations do not belong to any cluster, those outliers are labeled as -1
    # delete the outliers when counting the top 5 clusters
    del clustering_result[-1]
    #  return a list of top five clusters
    cluster = []
    for i in clustering_result.most_common(5):
        cluster.append(i[0])
    top_five_clusters = { k:v for k,v in clustering_result.items() if k in cluster}
    print "The top five clusters and the number of observations in each cluster", top_five_clusters
    # what percentage of data is from those clusters
    count_sum = 0.0
    for i in clustering_result.most_common(5):
        count_sum += i[1]
    percentage = count_sum/sample.shape[0] * 100
    return top_five_clusters, percentage

# for each timestamp, I compute the number of seconds since midnight
def num_seconds_since_midnight(j):
    d_truncated = j.replace(hour = 0, minute= 0, second = 0)
    delta = j - d_truncated
    seconds = delta.total_seconds()
    return seconds

# cluster time seconds
def time_cluster(data, epsilon):
    print "epsilon : %0.5f" % epsilon
    algo = DBSCAN(eps=epsilon, min_samples=10)
    algo.fit(data)
    labels = algo.labels_
    print "number of clusters : %s" % len(np.unique(labels))
    return labels

# randomly subset 1% of the data as my analysis scope, 15764 observations in total
sample = df1.sample(frac = 0.01)

# Question 1: do certain areas generate more pick-ups than others? What the percentage of trips originated from these locations?
# find the top five areas that generate more pickups

coords = sample[['Pickup_latitude','Pickup_longitude']]
# outputs a label for every observation
# set the argument distance as 0.12
# it means the maximal distance between two points for them to be considered as in the same neighborhood is 0.12km
labels = DBSCAN_cluster(coords, distance = 0.12)
# outputs top five clusters and the percentage of observations that belong to these five clusters
pickup_cluster, pickup_percentage = top_five_cluster(labels)
print "percentage of trips originated from those locations " , pickup_percentage

# Output the top 5 clusters into csv files
# You can uncomment this to see the clusters I generated
# cluster_output = pd.Series()
# for n in pickup_cluster:
#     cluster_output = coords[labels == n]
#     cluster_output.to_csv("pickup%s.csv"%n, index = False)

# Question 2: what are the top five termination points of trips? What percentage of trips terminated in these locations?
coords = sample[['Dropoff_longitude','Dropoff_latitude']]
# outputs a label for every observation
labels = DBSCAN_cluster(coords, distance = 0.10)
dropoff_cluster, dropoff_percentage = top_five_cluster(labels)
print "percentage of trips terminated in those locations " , dropoff_percentage

# Output the top clusters into csv file
# cluster_output = pd.Series()
# for n in dropoff_cluster:
#     cluster_output = coords [labels == n]
#     cluster_output.to_csv("dropoff%s.csv" % n, index=False)

# Question 3 : find the rush hours 
time = sample[['lpep_pickup_datetime']]
time = pd.to_datetime(time['lpep_pickup_datetime'])

# use this new feature to do clustering
second = time.apply(num_seconds_since_midnight)
second = pd.DataFrame(second)

labels = time_cluster(second, 45)

# find the top three clusters
c = Counter(labels)
del c[-1]
cluster = []
for i in c.most_common(3):
    cluster.append(i[0])
print "clusters are ", cluster

# generate the time interval for each cluster
time_interval = []
for n in cluster:
    cluster_output = second[labels == n]
    time_interval.append(cluster_output.min(axis = 0).values)
    time_interval.append(cluster_output.max(axis=0).values)
    # you can uncomment these to generate 3 clusters into csv files
    # cluster_output.to_csv("time%s.csv" % n, index=False)

# convert the number of seconds back to timestamp hour-miniute-second
print "the first cluster time interval is:"
for t in (time_interval[0],time_interval[1]):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    print "%d:%02d:%02d" % (h, m, s)

print "the second cluster time interval is:"
for t in (time_interval[2],time_interval[3]):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    print "%d:%02d:%02d" % (h, m, s)

print "the third cluster time interval is:"
for t in (time_interval[4],time_interval[5]):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    print "%d:%02d:%02d" % (h, m, s)

# question 4: "If we define lucrative trips as generating the highest fare for least amount time spent, what are the top 5 locations for the origin of the most lucrative trips?"
# 0.12, 2964

# series interval is the interval between pickup and dropoff
dropoff = pd.to_datetime(sample['Lpep_dropoff_datetime'])
pickup =  pd.to_datetime(sample['lpep_pickup_datetime'])

# since some time interval is zero , I add 60 seconds to the time interval
interval = (dropoff - pickup).astype('timedelta64[s]') + 60
fare_per_time_ratio = sample['Fare_amount'].abs()/interval

# only focus on the lucrative trips
high_ratio_data = sample[fare_per_time_ratio > fare_per_time_ratio.median()]
trips =high_ratio_data[['Pickup_latitude','Pickup_longitude']]

# there are around 7800 trips can be defined as lucrative trips
labels = DBSCAN_cluster(trips, 0.12)

pickup_cluster_lucrative, pickup_percentage_lucrative = top_five_cluster(labels)

# Output the top clusters into csv file
# cluster_output = pd.Series()
# for n in pickup_cluster_lucrative.keys():
#     cluster_output = trips[labels == n]
#     cluster_output.to_csv("question5%s.csv" % n, index=False)


# question 5: If we define lucrative trips as generating the highest fare for least amount time spent, what are the top 5 locations for the termination of trips
trips =high_ratio_data[['Dropoff_latitude','Dropoff_longitude']]
labels = DBSCAN_cluster(trips, 0.12)

dropoff_cluster_lucrative, dropoff_percentage_lucrative = top_five_cluster(labels)

# Output the top clusters into csv file
# cluster_output = pd.Series()
# for n in dropoff_cluster_lucrative.keys():
#     cluster_output = trips[labels == n]
    # cluster_output.to_csv("question6%s.csv" % n, index=False)
