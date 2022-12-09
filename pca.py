
import cv2
import numpy as np
import os
import re

CASC_PATH = 'C:\\Users\\Radon\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml'
CASC_PATH2 = 'C:\\Users\\Radon\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml'
TRAIN_PATH = 'C:\\Users\\Radon\\py\\pca\\photo'#测试集全部经过预处理
TEST_PATH = 'C:\\Users\\Radon\\py\\pca\\test_s'

thedataisgood=0#0：验证集需要人脸检测，1：验证集不需要人脸检测，要求图片格式为128*128

if thedataisgood==1:
    TEST_PATH = 'C:\\Users\\Radon\\py\\pca\\test'
else:
    TEST_PATH = 'C:\\Users\\Radon\\py\\pca\\test_s'

scaleFactor_a=1.2#1.1速度较慢，但可能过度检测，1.3速度较快s，可能检测不到人脸
minNeighbors_a=3
TrainList = []
faceCascade=cv2.CascadeClassifier(CASC_PATH2)

def showimage(image):#人脸矩阵显示，调试代码时使用，主程序未使用
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imshow("image",image)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()

def findAllFile(base):#文件夹查找文件
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def walking(path):#在文件夹查找文件，并输出文件地址到一列表
    lis=[]
    if os.path.exists(path):
        pass        
    else:
        print('this path not exist')
        return
    for i in findAllFile(path):
        lis.append(i)
    return lis

def sloving(test_i):#人脸检测，在图片中寻找人脸，输出128*128灰色人脸图片矩阵
    test_list=[]
    test_image = cv2.imread(test_i,0)
    test_faces = faceCascade.detectMultiScale(test_image,scaleFactor_a, minNeighbors_a)
    if len(test_faces)==0:
        return None,0
    for (x, y, w, h) in test_faces:
        test_cutResize = cv2.resize(test_image[y:y + h, x:x + w], (128, 128),interpolation=cv2.INTER_CUBIC)  
    for x in range(test_cutResize.shape[0]):
        for y in range(test_cutResize.shape[1]):
            test_list.append(test_cutResize[x, y]) 
    return test_list,1

def training():#总和所有人脸矩阵
    for i in lisa:
        TrainList.append(sloving(i))

def nosloving(test_i):#不进行人脸检测，直接将图片转换为灰色，矩阵输出
    test_list=[]
    test_image = cv2.imread(test_i,0)
    for x in range(test_image.shape[0]):
        for y in range(test_image.shape[1]):
            test_list.append(test_image[x, y]) 
    return test_list

def gamma_bianhuan(image,gamma):#图片伽马值修改，主程序未使用
    image=image/255.0
    New=np.power(image,gamma)
    return New

def gamma_j(image):#图片伽马值修改，输入为图片，输出为灰色人脸矩阵
    a=cv2.imread(image,cv2.IMREAD_UNCHANGED)
    image1 = cv2.split(a)[0]#蓝
    image2 = cv2.split(a)[1]#绿
    image3 = cv2.split(a)[2]#红
    image_1= gamma_bianhuan(image1, 0.5)
    image_2 = gamma_bianhuan(image2,0.5)
    image_3 = gamma_bianhuan(image3,0.5)
    merged = cv2.merge([image1, image2, image3])
    merged=cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    return merged

lisa=walking(TRAIN_PATH)#查找训练集地址

for i in lisa:#所有人脸矩阵
    list=[]
    image = cv2.imread(i,0)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            list.append(image[x, y])
    TrainList.append(list)


#PCA降维主程序
trainFaceMat = np.mat(TrainList)
meanFaceMat = np.mean(trainFaceMat, axis=0)#中心
normTrainFaceMat = trainFaceMat - meanFaceMat#中心化
covariance = np.cov(normTrainFaceMat)#协方差矩阵
eigenvalue, featurevector = np.linalg.eig(covariance)#特征值，特征向量
sorted_Index = np.argsort(eigenvalue)#排序
topk_evecs = featurevector[:,sorted_Index[:-40-1:-1]]
eigenface = np.dot(np.transpose(normTrainFaceMat), topk_evecs)
eigen_train_sample = np.dot(normTrainFaceMat, eigenface)



#验证
success=0#成功个数
def readn(content):#查找字符串中数字
    content =content.replace(" ","")
    model = re.compile("[0-9]+")
    if model.search(content) is not None:
       pos = model.search(content).span()
       return int(content[pos[0]:pos[1]])

lisb=walking(TEST_PATH)#查找验证集地址

for test_i in lisb:
    if thedataisgood==0:
        (temp,state)=sloving(test_i)
    else:
        state=1
        temp=nosloving(test_i)
    if state!=1:
        print('该图片（位于：'+test_i+'）无法检测到人脸，请调整角度后重试。\n')
        continue
    testFaceMat = np.mat(temp)
    normTestFaceMat = testFaceMat - meanFaceMat
    eigen_test_sample = np.dot(normTestFaceMat, eigenface)
    minDistance = np.linalg.norm(eigen_train_sample[0] - eigen_test_sample)
    num = lisa[0]
    for i in range(1, eigen_train_sample.shape[0]):
        distance = np.linalg.norm(eigen_train_sample[i] - eigen_test_sample)
        if minDistance > distance:
            minDistance = distance
            num = lisa[i]

    print(test_i+'\0match\0'+num)
    if readn(test_i)==readn(num):
        success=success+1
    print('')
print(success)#输出成功个数

