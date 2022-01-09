import random
from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np

# np.set_printoptions(threshold=np.inf)


# 计算适应度函数（像素差的平方的倒数作为相似度，但这个值会很小，采取什么值更合适？）
def compute_fitness(individual, target):
    # print('individual:', individual)
    # print('target:', target)
    # print(individual.shape)
    # score = np.sum(np.square((individual - target)))    # 像素差的平方应该越小越好，应该取倒数...之前搞错了
    diff = np.abs(individual - target)
    # print(diff, diff.shape)
    score = np.sum(diff)
    # print('sum_score:', score)
    score = height*width*channels / score   # 适应度=像素点个数*通道数/所有像素点4通道差值累积和

    return score


# 初始化染色体
def init_chromosome(x, y, z):
    # np.random.seed(id)
    # chromosome = np.random.random((x, y, z))
    chromosome = np.random.randint(256, size=(x, y, z))
    return chromosome


# 初始化个体（设置染色体和适应度函数评分）
def init_individual(x, y, z, target):
    chromosome = init_chromosome(x, y, z)
    # print('chromosome:', chromosome)
    # print('target:', target)
    score = compute_fitness(chromosome, target)
    indv = {}
    indv['score'] = score
    indv["chromosome"] = chromosome
    return indv


# 初始化种群
def init_polulation(population_num, x, y, z, target):
    population = []
    for i in range(population_num):
        indv = init_individual(x, y, z, target)
        # print(str(i) + " score:" + str(indv['score']))
        population.append(indv)     # 注意种群列表的元素是一个dict
        # print(i, indv["chromosome"])
    return population


def find_score(indv):
    return indv['score']


# 选择作为父本的种群
def select(population_num, population):
    # # 轮盘赌算法选择生成新的种群作为父本
    # sum_fitness = 0
    # for i in range(population_num):
    #     sum_fitness += population[i]['score']
    # prob = [0 for i in range(population_num)]
    # for i in range(population_num):
    #     prob[i] += population[i]['score'] / sum_fitness
    #     if i > 0:
    #         prob[i] += prob[i-1]    # 为了符合轮盘赌的规则进行概率的构造
    # new_population = []
    # for i in range(population_num):     # 构成种群数量前后保持一致
    #     rand = random.random()
    #     for j in range(population_num):     # 找到轮盘赌对应的区间的个体
    #         if prob[j] >= rand:
    #             new_population.append(population[j])
    #             break
    # 感觉可以选择前n/2适应度的个体作为父本，将父本和孩子作为新的种群
    population.sort(key=find_score, reverse=True)
    new_population = population[:16]

    return new_population


# 交叉（两个个体对应的每个基因都有一定的概率进行交换，或者简单处理找一个交叉点）
def cross():    # 感觉交叉没起作用，需要debug一下
    global population

    unselected = [i for i in range(16)]
    while len(unselected)!=0:
        p1 = random.choice(unselected)
        unselected.remove(p1)
        indv1 = population[p1]["chromosome"]
        p2 = random.choice(unselected)
        unselected.remove(p2)
        indv2 = population[p2]["chromosome"]

        new_indv1 = np.zeros([height, width, channels], dtype=np.int)     # 这里注意np.zeros的值是浮点数0.不是整数0，要规定dtype

        for i in range(height):
            for j in range(width):
                p0 = random.random()
                if p0 < 0.5:
                    new_indv1[i][j] += indv1[i][j]
                else:
                    new_indv1[i][j] += indv2[i][j]

        new_indv2 = indv1 + indv2 - new_indv1

        tmp1 = {}
        tmp1['score'] = 0
        tmp1["chromosome"] = new_indv1
        tmp2 = {}
        tmp2['score'] = 0
        tmp2["chromosome"] = new_indv2
        population.append(tmp1)
        population.append(tmp2)
    return


# 变异（可以直接对像素点进行突变，也可以利用正态分布等方式生成数值）
def mutate():
    global population
    # p1 = random.randint(0, population_num-1)
    # indv = population[p1]["chromosome"]
    # # 对该点距离为2的区域进行变异（因图形特点，相邻区域往往颜色相近）
    # r = 1
    # for m in range(height):
    #     for n in range(width):
    #         p = random.random()
    #         if p < mutate_rate:     # 应该对每一个像素点都进行一次突变判断
    #             # pos_x = random.randint(0, width-1)
    #             # pos_y = random.randint(0, height-1)
    #             pos_x = m
    #             pos_y = n
    #             lu = (max(0, pos_x - r), max(0, pos_y - r))
    #             ld = (max(0, pos_x - r), min(height-1, pos_y + r))
    #             ru = (min(width-1, pos_x + r), max(0, pos_y - r))
    #             rd = (min(width-1, pos_x + r), min(height-1, pos_y + r))
    #             # color = [random.random() for i in range(4)]
    #             color = np.random.randint(256, size=(4))
    #             for i in range(lu[1], ld[1]):
    #                 for j in range(lu[0], ru[0]):
    #                     indv[i][j] = color
    #             # indv[pos_x][pos_y] = [random.random() for i in range(4)]
    # population[p1]["chromosome"] = indv
    for k in range(population_num):
        m_x = random.sample(range(height), int(height * mutate_rate))
        m_y = random.sample(range(width), int(width * mutate_rate))
        indv = population[k]["chromosome"]
        for i, j in zip(m_x, m_y):
            color = np.random.randint(256, size=(4))
            indv[i][j] = color
        population[k]["chromosome"] = indv
    return


iter_num = 50000            # 最大迭代次数
output_frequency = 500       # 输出频率
population_num = 32         # 种群数量
# cross_rate = 0.8            # 交叉率，指后代基因来自两个父本基因各自的比例
mutate_rate = 0.1           # 变异率，可以设置为模拟退火中的逐渐减小的概率
output_dir = './result/'    # 图像输出目录

img = mpimg.imread('firefox.png')   # 256*256*4(RGBA)
# print('img:', img)  # 注意这里图片RGBA的值范围为[0,1]不是[0,255]？
# print(img.shape)
plt.subplot(1, 3, 1)
plt.imshow(img)

# img1 = misc.imresize(img, 0.0625) # 16*16
img1 = misc.imresize(img, 0.125)  # 32*32
# img1 = misc.imresize(img, 0.25)  # 64*64
# print('img1:', img1)    # 这里图片RGBA的值范围又变成了[0,255]？
print(img1.shape)
plt.subplot(1, 3, 2)
plt.imshow(img1)
height = img1.shape[0]
width = img1.shape[1]
channels = img1.shape[2]

generate_img = init_chromosome(height, width, channels)
print(generate_img.shape)
# print(generate_img)
plt.subplot(1, 3, 3)
plt.imshow(generate_img)
plt.show()

# 初始化种群
population = init_polulation(population_num, height, width, channels, img1)

for j in range(iter_num):
    # print("第" + str(j) + "次迭代：")
    # 挑选作为父本的种群
    # print("---------------select---------------")
    population = select(population_num, population)
    # print(len(population))

    # for i in range(population_num):
    #     print(i, population[i]["chromosome"])
    # for i in range(8):
    #     print(i, population[i]["score"])

    # 交叉
    # print("---------------cross---------------")
    cross()    # 这里应该是每次从unselected中随机选两个个体交叉，将其remove，重复8次让所有个体都发生交叉过程
    # print(len(population))

    # for i in range(population_num):
    #     print(i, population[i]["chromosome"])
    # for i in range(16):
    #     print(i, population[i]["score"])

    # 变异
    # print("---------------mutute---------------")
    mutate()

    # 更新，重新计算新种群个体适应度
    # print("---------------update---------------")
    new_population = []
    for i in range(population_num):
        # print(population[i]["chromosome"].shape)
        score = compute_fitness(population[i]["chromosome"], img1)
        # print(score)
        indv = {}
        indv['score'] = score
        indv["chromosome"] = population[i]["chromosome"]
        # print(indv["chromosome"])
        new_population.append(indv)
    population = new_population

    # for i in range(16):
    #     print(i, population[i]["score"])

    if j % output_frequency == 0:
        avg_score = 0
        print("第" + str(j) + "次迭代：")
        for i in range(population_num):
            plt.subplot(8, 4, i + 1)
            # 对于三维数组（彩色图像），plt.imshow() 函数并不会自动对输入数据归一化处理，而是对数据取值范围提出要求：
            # 如果是float型数据，取值范围应在[0,1]；如果是int型数据，取值范围应在[0,255]
            # print(i, population[i]["chromosome"])
            plt.imshow(population[i]["chromosome"])
            print("score:", population[i]["score"])
            avg_score += population[i]["score"]
        plt.savefig(output_dir + str(j) + '_' + str((avg_score)/32) + '.png')
        # plt.show()
