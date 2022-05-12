import pickle
import cv2
import numpy as np
from PIL import Image
import modelv2 as model
import os


def pipeline_model(path, filename, color):
    gender_pre = ['Male', 'Female']
    font = cv2.FONT_HERSHEY_SIMPLEX

    # loading all the models
    haar = cv2.CascadeClassifier('/Users/jeanxavier/Documents/workspace/opencv-flask/modelv2/haarcascade_frontalface_default.xml')
    mean = pickle.load(open('/Users/jeanxavier/Documents/workspace/opencv-flask/modelv2/mean_preprocess.pickle', 'rb'))
    model_svm = pickle.load(open('/Users/jeanxavier/Documents/workspace/opencv-flask/modelv2/model_svm.pickle', 'rb'))
    model_pca = pickle.load(open('/Users/jeanxavier/Documents/workspace/opencv-flask/modelv2/pca_50.pickle', 'rb'))

    # test data
    test_data = path
    color = color

    # reading the image
    img = cv2.imread(test_data)

    # convert to gray color
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # detect and crop of the image of iterest
    faces = haar.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw the rectangle
        roi = gray[y:y + h, x:x + w]

        # normalize the image
        roi = roi / 255.0

        # ste5 resize the iamge
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)

        # setp-6: Flattening the image by (1x10000)
        roi_reshape = roi_resize.reshape(1, 10000)  # 1, -1

        # step7: substract iwth mean
        roi_mean = roi_reshape - mean

        # 8 get eigen image
        eigen_image = model_pca.transform(roi_mean)

        # step 9: pass to svm model
        results = model_svm.predict_proba(eigen_image)[0]

        # 10
        predict = results.argmax()  # 0 or 1
        score = results[predict]

        # 11
        text = "%s : %0.2f" % (gender_pre[predict], score)
        cv2.putText(img, text,(x,y), font, 1, (0,255,0),2)

    cv2.imwrite('./static/predict/{}'.format(filename),img)

    return gender_pre[predict]
    # cv2.imshow('Gender Prediction', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return img


# Apply to video mode
def pipeline_model_video():
    cap = cv2.VideoCapture('video.mp4')

    while True:
        ret, frame = cap.read()

        if ret == False:
            break
        frame = pipeline_model(frame, color='bgr')
        cv2.imshow('Gender Detection',frame)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
if __name__ == '__main__':
    test_data = 'photo.jpeg'
    color = 'bgr'

    img = Image.open(test_data)
    img = np.array(img)
    img = pipeline_model(test_data,color)

    # pipeline_model_video()


