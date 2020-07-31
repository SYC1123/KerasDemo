# 1. 导入库和模块
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# 2. 加载数据
from keras.datasets import mnist

# https://blog.csdn.net/u014626748/article/details/86674768
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
# https://blog.csdn.net/qq_45465526/article/details/103125997?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param
# (60000, 28, 28)
# plt.imshow(x_train[50])
# plt.show()
#
# # 3. 数据预处理
img_x, img_y = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# print(y_train)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(y_train.shape)
# (60000, 10)
#
#
# # 4. 定义模型结构
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(img_x, img_y, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))
#
# # 5. 编译
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # 6. 训练
model.fit(x_train, y_train, batch_size=128, epochs=10)
# # 7.保存模型
# model.save('./MNISTKerasTrainAndTast.h5')
# # 8. 评估模型
score = model.evaluate(x_test, y_test)
print(score[0])
print('acc', score[1])
# acc 0.9926
