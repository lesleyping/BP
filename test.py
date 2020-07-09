#coding:utf-8
## https://zhuanlan.zhihu.com/p/38006693
import sklearn
import sklearn.datasets
import numpy as np

def sigmoid(input_sum):
    output = 1.0/(1 + np.exp(-input_sum))
    return output, input_sum

def sigmoid_back(derror_wrt_output, input_sum):
    output = 1.0/(1 + np.exp(-input_sum))
    doutput_wrt_dinput = output * (1 - output)
    derror_wrt_dinput = derror_wrt_output * doutput_wrt_dinput
    return derror_wrt_dinput

def activated(activation_choose, input):
    if activation_choose == 'sigmoid':
        return sigmoid(input)

    return sigmoid(input)

class NeuralNetwork:
    def __init__(self, layers_structure, print_cost=False):
        self.layers_structure = layers_structure
        self.layers_num = len(layers_structure)

        self.param_layers_num = self.layers_num - 1

        self.learning_rate = 0.0618
        self.num_iterations = 2000
        self.x = None
        self.y = None
        self.w = dict()
        self.b = dict()
        self.costs = []
        self.print_cost = print_cost

        self.init_w_b()
    
    def init_w_b(self):
        np.random.seed(3)

        for l in range(1, self.layers_num):
            self.w["w" + str(l)] = np.random.randn(self.layers_structure[l], self.layers_structure[l-1])/np.sqrt(self.layers_structure[l-1])
            self.b["b" + str(l)] = np.zeros((self.layers_structure[l], 1))
        
        return self.w, self.b
    
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_num_iterations(self, num_iterations):
        self.num_iterations = num_iterations

    def set_xy(self, input, label):
        self.x = input
        self.y = label

    def layer_activation_forward(self, x, w, b, activation_choose):
        input_sum = np.dot(w,x) + b
        output, _ = activated(activation_choose, input_sum)
        return output, (x, w, b, input_sum)

    def forward_propagation(self, x):
        caches = []
        output_prev = x
        L = self.param_layers_num
        for l in range(1,L):
            input_cur = output_prev
            output_prev, cache = self.layer_activation_forward(input_cur, self.w["w" + str(l)], self.b["b" + str(l)], "sigmoid")
            caches.append(cache)
        
        output, cache = self.layer_activation_forward(output_prev, self.w["w" + str(L)], self.b["b" + str(L)], "sigmoid")
        caches.append(cache)

        return output, caches

    def compute_error(self, output):
        m = self.y.shape[1]
        error = np.sum(0.5 * (self.y - output) ** 2) / m
        error = np.squeeze(error)

        return error

    def layer_activation_backward(self, derror_wrt_output, cur_cache):
        input, w, b, input_sum = cur_cache
        output_prev = input
        m = output_prev.shape[1]

        derror_wrt_dinput = sigmoid_back(derror_wrt_output, input_sum)
        derror_wrt_dw = np.dot(derror_wrt_dinput, output_prev.T) / m
        
        derror_wrt_db = np.sum(derror_wrt_dinput, axis=1, keepdims=True)/m

        derror_wrt_output_prev = np.dot(w.T, derror_wrt_dinput)

        return derror_wrt_output_prev, derror_wrt_dw, derror_wrt_db

    def back_propagation(self, output, caches):

        """
        函数:
            神经网络的反向传播
        输入:
            output：神经网络输
            caches：所有网络层（输入层不算）的缓存参数信息  [(x, w, b, input_sum), ...]
        返回:
            grads: 返回当前迭代的梯度信息
        """

        grads = {}
        L = self.param_layers_num #
        output = output.reshape(output.shape)  # 把输出层输出输出重构成和期望输出一样的结构

        expected_output = self.y

        # 见式(5.8)
        #derror_wrt_output = -(expected_output - output)

        # 交叉熵作为误差函数
        derror_wrt_output = - (np.divide(expected_output, output) - np.divide(1 - expected_output, 1 - output))

        # 反向传播：输出层 -> 隐藏层，得到梯度：见式(5.8), (5.13), (5.15)
        current_cache = caches[L - 1] # 取最后一层,即输出层的参数信息
        grads["derror_wrt_output" + str(L)], grads["derror_wrt_dw" + str(L)], grads["derror_wrt_db" + str(L)] = \
            self.layer_activation_backward(derror_wrt_output, current_cache)

        # 反向传播：隐藏层 -> 隐藏层，得到梯度：见式 (5.28)的(Σδ·w), (5.28), (5.32)
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            derror_wrt_output_prev_temp, derror_wrt_dw_temp, derror_wrt_db_temp = \
                self.layer_activation_backward(grads["derror_wrt_output" + str(l + 2)], current_cache)

            grads["derror_wrt_output" + str(l + 1)] = derror_wrt_output_prev_temp
            grads["derror_wrt_dw" + str(l + 1)] = derror_wrt_dw_temp
            grads["derror_wrt_db" + str(l + 1)] = derror_wrt_db_temp

        return grads
    
    def update_w_and_b(self, grads):
        """
        函数:
            根据梯度信息更新w，b
        输入:
            grads：当前迭代的梯度信息
        返回:

        """

        # 权值w和偏置b的更新，见式:（5.16),(5.18)
        for l in range(self.param_layers_num):
            self.w["w" + str(l + 1)] = self.w["w" + str(l + 1)] - self.learning_rate * grads["derror_wrt_dw" + str(l + 1)]
            self.b["b" + str(l + 1)] = self.b["b" + str(l + 1)] - self.learning_rate * grads["derror_wrt_db" + str(l + 1)]


    def training_module(self):
        np.random.seed(5)
        for i in range(0 ,self.num_iterations):
            output, caches = self.forward_propagation(self.x)            
            cost = self.compute_error(output)
            grads = self.back_propagation(output, caches)
            self.update_w_and_b(grads)
            #当次迭代结束，打印误差信息
            if self.print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 1000 == 0:
                self.costs.append(cost)

        return self.w, self.b


if __name__ == "__main__":
    # 60个点，noise噪声系数越大噪声越大，两种点
    xy, colors = sklearn.datasets.make_moons(60, noise=1.0)
    # 输出层2个神经元，[1,0] [0,1]为两个类别
    y = []
    for c in colors:
        if c == 1:
            y.append([0,1])
        else:
            y.append([1,0])
    y = np.array(y).T
    # import pdb
    # pdb.set_trace()
    
    hidden_layer_neuron_num_list = [1,2,4,10,20,50]

    #for i, hidden_layer_neuron_num in enumerate(hidden_layer_neuron_num_list):
    nn = NeuralNetwork([2, 4, 2], True)
    print(nn)
    nn.set_xy(xy.T, y)
    nn.set_num_iterations(30000)
    nn.set_learning_rate(0.1)
    w, b = nn.training_module()
    print("%i iter w:", w)
    print("%i iter b:", b)


    