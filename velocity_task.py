from matplotlib.pyplot import hist
import modneat
import math
import random
import numpy as np

TASK_LV = 5 #LV.0: 進化のみ， Lv.1: 学習のみ、Lv.2: 二次学習も、という感じ
#Lv.0 ターゲットが変動しない
#Lv.1 ターゲットが変動する。ただし、変動速度は全世代で一定
#Lv.2 ターゲットが変動が変動する。ただし、変動速度は各世代でランダムに決定
#Lv.3 ターゲットが変動する。ただし、変動速度は各変動ごとにランダムに決定
#Lv.4(?) ターゲットが変動する。ただし、変動の加速度は全世代で一定
#Lv.5(?) ターゲットが変動する。変動の加速度は各世代ごとにランダムに決定

PHASE_NUM = 10 #生涯内で何回変更が生じるか. ステップ数に応じて正規化することを忘れないように
TARGET_UPPER_LIMIT = 1
TARGET_LOWER_LIMIT = 0
TARGET_V = 0.01 #Lv:1の時の変化量
TARGET_V_UPPER_LIMIT = 0.1
TARGET_V_LOWER_LIMIT = 0.01
TARGET_A = 0.005 #ターゲットの加速度
TARGET_A_UPPER_LIMIT = 0.01
TARGET_A_LOWER_LIMIT = 0.001

LOOP_NUM =  10 #ネットワークをリセットして何回同一タスクを実行するか。

def sigmoid(x):
    return 1/ (1+np.exp(-x))

class velocity_task_N:
    def __init__(self, network_type):
        self.network_type = network_type
    
    def eval_fitness(self, net):
        n_loop = LOOP_NUM
        total_fitness = 0.0
        history = []
        for n in range(n_loop):
            fitness = 0.0
            history.append([])
            net.reset()
            env = velocity_env()
            input = env.reset()
            is_done = False
            env_step = 0
            while(not is_done):
                env_step += 1
                input = sigmoid(input)
                output = net.activate([input])
                history[-1].append({'target': env.target, 'output':output, 'input': input })
                input, error, is_done = env.step(output)
                fitness -= error
            fitness /= env_step #適応度をステップ数に応じて正規化
            total_fitness += fitness

        total_fitness /= n_loop

        return total_fitness, history

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = self.network_type.create(genome, config)
            genome.fitness, genome.history = self.eval_fitness(net)
    
    def eval_single_genome(self, genome, config):
        net = self.network_type.create(genome, config)
        genome.fitness, genome.history = self.eval_fitness(net)
        return genome.fitness, genome.history

    def show_results(self, best_genome, config, stats, out_dir):
        pass

class velocity_env:
    def __init__(self):
        self.lv = TASK_LV
        self.target = 0
        self.a = 0

    def reset(self):
        self.step_num = 1
        self.change_num = 0
        self.stag = 50 + (random.randint(-10,10))
        if(self.lv == 0):
            self.target = (TARGET_UPPER_LIMIT + TARGET_LOWER_LIMIT) / 2
            self.target_v = 0
        elif(self.lv == 1 or self.lv == 4 or self.lv == 5):
            #self.target = random.choice([TARGET_UPPER_LIMIT, TARGET_LOWER_LIMIT])
            self.target = TARGET_LOWER_LIMIT
            if(self.target == TARGET_LOWER_LIMIT):
                self.is_growing = True
            else:
                self.is_growing = False
            self.target_v = TARGET_V

            if(self.lv == 4):
                self.target_a = TARGET_A
            elif(self.lv == 5):
                self.target_a = random.uniform(TARGET_A_LOWER_LIMIT, TARGET_A_UPPER_LIMIT)

        elif(self.lv == 2 or self.lv == 3):
            #self.target = random.choice([TARGET_UPPER_LIMIT, TARGET_LOWER_LIMIT])
            self.target = TARGET_LOWER_LIMIT
            if(self.target == TARGET_LOWER_LIMIT):
                self.is_growing = True
            else:
                self.is_growing = False
            self.target_v = random.uniform(TARGET_V_LOWER_LIMIT, TARGET_V_UPPER_LIMIT)

        return(self.target)
    
    def step(self, net_output):
        observation = self.target - net_output[0]
        error = abs(observation)

        if(self.lv == 0):
            pass
        else:
            if(self.stag >= 0):
                self.stag -= 1
            else:
                if(self.is_growing):
                    self.target += self.target_v
                else:
                    self.target -= self.target_v
                
                #加速度の実装
                if(self.lv == 4 or self.lv == 5):
                    self.target_v += self.target_a

                if(self.is_growing and self.target >= TARGET_UPPER_LIMIT):
                    self.target = TARGET_UPPER_LIMIT
                    self.is_growing = False
                    self.change_num += 1
                    self.stag = 50 + (random.randint(-10,10))

                    #速度のリセット
                    if(self.lv == 4 or self.lv == 5):
                        self.target_v = TARGET_V

                elif(not self.is_growing and self.target <= TARGET_LOWER_LIMIT):
                    self.target = TARGET_LOWER_LIMIT
                    self.is_growing = True
                    self.change_num += 1

                    if(self.lv == 4 or self.lv == 5):
                        self.target_v = TARGET_V

                    if(self.lv == 3):
                        self.target_v = random.uniform(TARGET_V_LOWER_LIMIT, TARGET_V_UPPER_LIMIT)

                    self.stag = 50 + (random.randint(-10,10))

        self.step_num += 1
        if(self.lv == 0):
            if(self.step_num <= 100):
                done = False
            else:
                done = True
        else:
            if(self.change_num <= PHASE_NUM ):
                done = False
            else:
                done = True

        return observation, error, done