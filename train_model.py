# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
params = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
	"gamma": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
model = GridSearchCV(SVC(kernel="rbf", gamma="auto",
	probability=True), params, cv=3, n_jobs=-1)
model.fit(data["embeddings"], labels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()