import tensorflow as tf
import numpy as np

from functions import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 30
seed = 700
embiggen_factor = 70

np.random.seed(seed)

centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

n_samples = tf.shape(samples)[0]
print(n_samples) 
print(samples)
print(centroids)

expanded_samples = tf.expand_dims(samples, 0)
expanded_centroids = tf.expand_dims(centroids, 1)
diff=tf.sub(expanded_samples, expanded_centroids)

model = tf.initialize_all_variables()
with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(centroids)
    diff_value=session.run(diff)   
    
    
print(sample_values.shape);
print(centroid_values.shape);
print(diff_value.shape)

#print(sample_values[0:12])
#print(centroid_values[0:3])
seed2=310
data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters,310)
nearest_indices = assign_to_nearest(samples, initial_centroids)
# nearest_indices = assign_to_nearest(samples, updated_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)
diff= tf.sub(updated_centroids, data_centroids)
distance = tf.reduce_sum( tf.square(diff))
threshold= tf.constant(1.5)

model = tf.initialize_all_variables()
# tricky recursive function chain mapping onto tf
with tf.Session() as session:
    # sample_values = session.run(samples)
    # centroid_values= session.run(data_centroids)
    print(centroid_values)
    for i in range(1000) :
      #if session.run(tf.less(distance, threshold)):
      #  print("distance lower than th, break on ", i)  
      #  break
      #else:
        [updated_centroids_value , distance_value]= session.run([updated_centroids, distance])
        # distance_value=session.run(distance)
        #print(updated_centroids_value)
        print(distance_value)
        # nearest_indices = assign_to_nearest(samples, updated_centroids)
        if(distance_value < 1.5):
            print(distance_value,", distance lower than th, break on ", i)  
            print(updated_centroids_value, session.run(data_centroids))
            break

