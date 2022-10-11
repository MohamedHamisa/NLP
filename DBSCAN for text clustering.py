
#Fit model
from sklearn.cluster import DBSCAN

#Use cosine because spacy uses cosine. min_samples = 2 because a cluster should have atleast 2 similar words
dbscan = DBSCAN(metric='cosine', eps=0.3, min_samples=2).fit(word_vectors)

########################################

#Unfortunately, the DBSCAN model does not have a built in predict function which we can use to label new tags
# However, the code below manually does that 

#Function for returning label prediction since there is no builtin function
def dbscan_predict(model, X):

    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]


test_words = ' '.join(['university', 'colleges', 'education', 'courses']).replace('-', ' ')
test_tokens = nlp(test_words)

test_vectors = []
for i in test_tokens:
  test_vectors.append(i.vector)
test_vectors = np.array(test_vectors)

print('Label for university:'+str(dbscan_predict(dbscan,np.array([test_vectors[0]]))[0]))
print('Label for colleges:'+str(dbscan_predict(dbscan,np.array([test_vectors[1]]))[0]))
print('Label for education:'+str(dbscan_predict(dbscan,np.array([test_vectors[2]]))[0]))
print('Label for courses:'+str(dbscan_predict(dbscan,np.array([test_vectors[3]]))[0]))
