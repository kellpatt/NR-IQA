from brisque import *
from sklearn import svm
import pandas as pd
import joblib
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import os

if __name__=="__main__":
    path = "Images/KonIQ-10k/512x384"
    brisque_scores = []
    files = os.listdir(path)
    data = {}
    data["brisque"] = []

    startSample = 1
    endSample = 2000
    for i in range(startSample, endSample + 1):
        file = files[i]
        print(f"{i}, {file}")
        image = cv2.imread(f"{path}/{file}")
        features = GetBrisqueFeatures(image)

        data["brisque"].append(features)

    # Load the MOS scores into a pandas data frame
    scores_file = "Images/KonIQ-10k/koniq10k_scores_and_distributions.csv"
    print(os.listdir("Images/KonIQ-10k"))
    mos_df = pd.read_csv(scores_file, usecols=["MOS"]).loc[startSample:endSample]
    y = mos_df["MOS"]
    X = pd.DataFrame(data["brisque"])

    # Train SVM using MOS scores and the 36 feature vectors
    # BRISQUE uses a radial basis function ("rbf") to perform support vector regression
    svr = svm.SVR(kernel='rbf')
    svr.fit(X, y)

    y_predic = svr.predict(X)

    # Save the trained model
    joblib.dump(svr, f"brisque_koniq_svr_samples{startSample, endSample}.pkl")

    # plot the predicted values against the true values
    xVals = np.linspace(1, y.shape[0], num=y.shape[0])
    plt.scatter(xVals, y, color='darkorange',
                label='data')
    plt.plot(xVals, y_predic, color='cornflowerblue',
             label='prediction')
    plt.legend()
    plt.show()
    plt.savefig("brisque_predics")


