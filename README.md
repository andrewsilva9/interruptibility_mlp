# Interruptibility Estimator

This is an estimator that pulls in Feature Vectors that have a person and associated information (pose information, objects, gaze information, etc) and then makes binary interruptibility estimations [0, 1] on that information. The person data is scaled with a scikit-learn StandardScaler and the classifier is a scikit-learn MLPClassifier.