import tensorflow as tf
import numpy as np

from functions import create_samples

n_features = 2
n_clusters = 3
n_samples_per_cluster = 100
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

print(sample_values[0:12])
print(centroid_values[0:3])

