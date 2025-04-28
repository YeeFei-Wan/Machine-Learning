import numpy as np
import random
import pandas as pd

class Network(object):
 
    def __init__(self, sizes): 
        """
        网络初始化，size 为网络从第一层输出层神经元数目如 [748,30,10]
        """
        
        self.num_layers = len(sizes)                    # 层次数，如上例返回 3
        self.sizes = sizes                              # 各层神经元数量，和 size 相等
        self.biases = [np.random.randn(y, 1) 
                       for y in sizes[1:]]              # 创建从第二层开始的各层的偏置，符合高斯分布
                                                        # biases中包含了第2，3，... 等层的偏置值
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
                                                        # 创建第1层到第2层，第2层到第3层，。。权值，
                                                        # 设size=[784,30,10],则2个矩阵大小分别为30*784，
                                                        # 10*30 权值按照高斯概率分布

    
    def feedforward(self, a): #前馈计算，给定输入a,得到神经网络的输出
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights): #例如输入是数字7的748*1的列向量，两层，第一层30*748，第二层10*30
            a = sigmoid(np.dot(w, a)+b) #则循环第一次输出计算大小为30*784的权值矩阵和748*1的乘积+偏置，得到30*1作为下层输入，
        return a #返回前馈后的激活输出
 
    
    def SGD(self,training_data,epochs, mini_batch_size,eta,test_data=None):
        """
        Parameters
        ----------
        training_data : list
            一个长度为样本数的列表，每个元素都是一个元组结构，对应一个样本。元组中包括了一个变量和一个label。
        epochs : int
            训练次数
        mini_batch_size : int
            用于小批量梯度下降的样本数.
        eta : float
            学习率.
        test_data : list, optional
            测试集，结构与training_data同. 默认参数是 None.

        Returns
        -------
        None.

        """
        
        if test_data: n_test = len(test_data)                       # 若测试集不为空，则对测试集进行测试
        n = len(training_data)                                      # 训练集的大小
        
        for j in range(epochs):
            random.shuffle(training_data)                           # 对训练集列表中的数据随机排序，然后从
                                                                    # 里面以min_batch_size为一批选取训练数据集合
            mini_batches = [training_data[k:k+mini_batch_size] 
                            for k in range(0, n, mini_batch_size)]  # 例如总的数据为40000，每个批次400，则mini_batches中
                                                                    # 存放了100个大小为400的数据集                                       
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)             # 对每个数据集，调用更新函数，计算一次正向传播和一次反
                                                                    # 向传播后的数据变动
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j)) # 输出字符串“第j代完成”，其中j分别取0，1，2.。。
 
    
    def update_mini_batch(self, mini_batch, eta):
        """
        利用梯度下降更新权值和偏置，其中代价函数对权值和偏导的计算使用反向传播得到。本函数只往前进一步
        
        Parameters
        ----------
        mini_batch : list
            训练集的截取.
        eta : float
            学习率.

        Returns
        -------
        None.
        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]          # 初始化代价函数对偏置 b 的偏导，全部取 0
        nabla_w = [np.zeros(w.shape) for w in self.weights]         # 初始化代价函数对权值 w 的偏导，全部取 0
        
        for x, y in mini_batch:                                     # 对于mini_batch中的每个x,y，其中x是28*28图片数据，y是10*1类别向量，计算：
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)      # 利用后向传播函数对一个样本对梯度的贡献进行计算
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]   # 遍历 mini_batch 里面每一个样本，把其对梯度的贡献相加
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]    # 更新权重
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
 
    
    def backprop(self, x, y): 
        """
        后向传播函数计算梯度

        Parameters
        ----------
        x : array
            储存一个样本所有变量的数组.
        y : array
            储存一个样本的标签的数组.

        Returns
        -------
        nabla_b : list
            此一个样本对偏置梯度的贡献.
        nabla_w : list
            此一个样本对权重梯度的贡献.

        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]          # 初始化
        nabla_w = [np.zeros(w.shape) for w in self.weights]         # 初始化
        
        # feedforward 前向传播计算输出
        activation = x                                              # 第一层的输出即是输入
        activations = [x]                                           # 每一层的带权输出，即激活值。对于输入层，激活值等于输入值
        zs = []                                                     # 每一层的带权输入存放在列表 zs 中
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)                          # 将当前层的激活值，也就是输出值放入到列表中
        
        # 后向传播
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])    # 求解最后输出层误差，输出层的就是
        nabla_b[-1] = delta                                         # 最后一层的偏置的偏导
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())    # 最后一层的权值的偏导
        for l in range(2, self.num_layers):                         # 后向传播，计算每一层偏置和权值的调整也就是梯度，从-2个算起
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp              # 临时变量，即中间隐层的偏差
            nabla_b[-l] = delta                                                     # 第 -l 层偏置的偏导 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())              # 第 -l 层权重的偏导
            
        return (nabla_b, nabla_w)
 
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) # argmax返回最大输出的索引值
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
 
    def cost_derivative(self, output_activations, y):
        return (output_activations-y) # 二次代价函数对激活输出的导数等于输出的激活值-y
 
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
 
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
 
###############################################################################################################################
# 定义函数
def to_categorical(labels):
    """
    将 0-9 之间的数字标签转换为 one-hot 格式的标签
    
    变量:
        labels: Bx1 大小的 narray
        
    返回：
        one_hot_labels: 10xB 大小的 narray
    """
    
    B = labels.size
    one_hot_labels = np.zeros((B, 10))
    
    for sample_index in range(B):
        one_hot_labels[sample_index, labels[sample_index]] = 1
        
    return one_hot_labels


# 读入图片数据
data = pd.read_csv(r"D:\DeskTop\Code\MyPython\西瓜书\3-线性模型\data\digits.csv")
train_data = data.sample(6000)
test_data = data.sample(10000)

x_train, y_train = train_data.iloc[:,1:].values, train_data.iloc[:,0].values
x_test = test_data.iloc[:,1:].values
y_test = test_data.iloc[:,0].values

y_train = to_categorical(y_train)

x_train=[x_train[i].reshape(784,1)/255 for i in range(x_train.shape[0])] # 对6000个图像转换成784*1列向量并归一化，放在列表中
y_train=[y_train[i].reshape(10,1) for i in range(y_train.shape[0])]#y_train中的类别标签也转换称为10*1的列向量

x_test=[x_test[i].reshape(784,1) for i in range(x_test.shape[0])] #测试数据集中的图像转换为784*列向量，但是标签不需要转换

training_data=list(zip(x_train,y_train))
test_data=list(zip(x_test,y_test))
net=Network([784,30,10])
net.SGD(training_data, epochs=30, mini_batch_size=200, eta=1.0, test_data=test_data)