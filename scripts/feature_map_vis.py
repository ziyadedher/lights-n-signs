import numpy as np

feature_map_1 = np.load("feature_map_1.npy")
feature_map_2 = np.load("feature_map_2.npy")
feature_map_3 = np.load("feature_map_3.npy")

print(feature_map_1.min())
print(feature_map_2.min())
print(feature_map_3.min())

print(feature_map_1[0, 0, 0, 0:12])
print(feature_map_1[0, 0, 0, 12:24])
print(feature_map_1[0, 0, 0, 24:36])

print(feature_map_1.shape)
# print(feature_map_1)
