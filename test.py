import cv2
import numpy as np
from keras.models import load_model

model = load_model('MNISTKerasTrainAndTast.h5')  # 选取自己的.h模型名称

image = cv2.imread('5.png')
img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # RGB图像转为gray

# 需要用reshape定义出例子的个数，图片的 通道数，图片的长与宽。具体的参加keras文档
img = (img.reshape(1, 28, 28, 1)).astype('float32') / 255
# predict = model.predict_classes(img)
predict = np.argmax(model.predict(img), axis=-1)  # 执行的是多分类
print('识别为：', predict)

cv2.imshow("Image1", image)
cv2.waitKey(0)
