import os
import urllib.request
import pandas
from sklearn import tree

dataset_path = ".\\dataset\\500-person-gender-height-weight-bodymassindex.zip"
if not os.path.exists(dataset_path):
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    dataset_url = r"https://storage.googleapis.com/kaggle-datasets/34879/46976/500-person-gender-height-weight-bodymassindex.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1533243842&Signature=aIc6eUHFYv7BKXC3CKaIAiS%2FBqJO4F%2FeNZXCs3ub1BKiNkjhgvSgNNpG1r%2F9VyLzs31pC1goHQh7ywPQzoqPrmKCqVeSEnx9bYKHLKum1JTv3uzZP6jD4fX8QJaqAhx4kYU3RrmAyLk%2FqrKR84dbTQc%2BChIa9YufX3RwodHqkY5sAAskKun80dZah58XUi9zDyYq%2FIa%2FB1E%2Bp%2By1TMAibmtf3NIcjt2dTlA%2Bm2W%2FKwvggyPml%2BiusRBSXjKk5zYtE2%2BeLHf8ZQlsid07Td%2FJIWOPnoVBvGuEyXBL9f7aH23Rgp2qmic78luetrTYGyxd9d4XrTWSy5sZgzcTZgBzjQ%3D%3D"
    urllib.request.urlretrieve(dataset_url, filename=dataset_path)

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!

print(prediction)
