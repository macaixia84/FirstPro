import numpy as np
import matplotlib.pyplot as plt
import h5py
import skimage.transform as tf

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5',"r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5',"r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

index = 208
plt.imshow(train_set_x_orig[index])
print("标签为" + str(train_set_y[:,index]) + ",这是一个'"+classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"'图片.")

print("train_set_x_orig shape:" + str(train_set_x_orig.shape))
print("train_set_y shape:"+str(train_set_y.shape))
print("test_set_x_orig shape:"+str(test_set_x_orig.shape))
print("test_set_y shape:"+str(test_set_y.shape))
#
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]
#
print("训练样本数：m_train = "+str(m_train))
print("测试样本数：m_test = "+str(m_test))
print("每张图片的宽/高：num_px = " + str(num_px))
#
train_set_X_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print("train_set_x_flatten shape:" + str(train_set_X_flatten.shape))
print("test_set_x_flatten shape:" + str(test_set_x_flatten.shape))
#
# train_set_x = train_set_X_flatten/255
# test_set_x = test_set_x_flatten/255
#
# def sigmoid(z):
#     s = 1 / (1 + np.exp(-z))
#     return s
#
# def initialize_with_zeros(dim):
#     w = np.zeros((dim,1))
#     b = 0
#     return w,b
# def propagate(w,b,X,Y):
#     m = X.shape[1]
#     A = sigmoid(np.dot(w.T,X)+b)
#     cost = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
#
#     dZ = A-Y
#     dw = np.dot(X,dZ.T)/m
#     db = np.sum(dZ)/m
#
#     grads = {"dw":dw,"db":db}
#     return grads,cost
#
# def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
#     costs = []
#     for i in range(num_iterations):
#         grads,cost = propagate(w,b,X,Y)
#
#         dw = grads['dw']
#         db = grads['db']
#
#         w = w - learning_rate*dw
#         b = b - learning_rate*db
#
#         if i%100 == 0:
#             costs.append(cost)
#             if print_cost:
#                 print("优化%i次后成本是：%f"%(i,cost))
#         params = {"w":w,"b":b}
#         return params,costs
# def predict(w,b,X):
#     m = X.shape[1]
#     Y_prediction = np.zeros((1,m))
#
#     A = sigmoid(np.dot(w.T,X) + b)
#
#     for i in range(A.shape[1]):
#         if A[0,i] >= 0.5:
#             Y_prediction[0,i] = 1
#     return Y_prediction
#
# def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,print_cost = False):
#     w,b = initialize_with_zeros(X_train.shape[0])
#     parameters,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
#
#     w = parameters['w']
#     b = parameters['b']
#
#     Y_prediction_train = predict(w,b,X_train)
#     Y_prediction_test = predict(w,b,X_test)
#
#     print("对训练图片的预测准确率为：{}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train))*100))
#     print("对测试图片的预测准确率为:{}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test))*100))
#
#     d = {"costs":costs,
#          "Y_prediction_test":Y_prediction_test,
#          "Y_prediction_train":Y_prediction_train,
#          "w":w,
#          "b":b,
#          "learning_rate":learning_rate,
#          "num_iterations":num_iterations}
#     return d
#
# d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations= 2000,learning_rate = 0.005,print_cost= True)
#
# index = 8
# plt.imshow(test_set_x[:,index].reshape((num_px,num_px,3)))
# print("这张图的标签是"+str(test_set_y[0,index]) + ",预测结果是"+str(int(d["Y_prediction_test"][0,index])))
#
# # costs = np.squeeze(d['costs'])
# # plt.plot(costs)
# # plt.ylabel('cost')
# # plt.xlabel('iterations(per hundreds)')
# # plt.title("learning rate = " + str(d["learning_rate"]))
# # plt.show()
# learning_rates = [0.01,0.001,0.0001]
# # models = {}
# # for i in learning_rates:
# #     print("学习率为:"+str(i) + "时")
# #     models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
# #     print('\n' + "-------------------------------"+'\n')
# #
# # for i in learning_rates:
# #     plt.plot(np.squeeze(models[str(i)]['costs']),label = str(models[str[i]]["learning_rate"]))
# #
# # plt.ylabel('cost')
# # plt.xlabel('iterations (hundreds)')
# #
# # legend = plt.legend(loc = 'upper center',shadow = True)
# # frame = legend.get_frame()
# # frame.set_facecolor('0.90')
# # plt.show()
# #
#
# my_image = "my_image1.jpg"
# fname = "images/" + my_image
#
# image = np.array(plt.imread(fname))
# my_image = tf.resize(image,(num_px,num_px),mode = 'reflect').reshape((1,num_px*num_px*3)).T
# my_predicted_image = predict(d["w"], d["b"], my_image)
#
# plt.imshow(image)
# print("预测结果为 "+str(int(np.squeeze(my_predicted_image))))