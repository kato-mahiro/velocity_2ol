import modneat
import math
import random

class velocity_task_0:
    def __init__(self, network_type):
        self.network_type = network_type
        self.order = 0
    
    def eval_fitness(self, net):
        history=[]
        net.reset()
        env = velocity_env(self.order)
        input = env.reset()
        is_done = False
        fitness = 0.0
        while(not is_done):
            output = net.activate([input])
            history.append([env.target_v, output])
            input, error, is_done = env.step(output)
            fitness -= error

        return fitness, history

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

class velocity_task_1(velocity_task_0):
    def __init__(self, network_type):
        super().__init__(network_type)
        self.order = 1

class velocity_task_2(velocity_task_0):
    def __init__(self, network_type):
        super().__init__(network_type)
        self.order = 2

class velocity_env:
    def __init__(self, order:int):
        self.order = order
        self.target_v = 0
        self.a = 0

    def reset(self):
        self.step_num = 1
        self.a_stag = 1
        if(self.order == 0):
            self.target_v = 0.5
            self.a = 0
        elif(self.order == 1):
            self.target_v = random.uniform(0.0, 1.0)
            self.a = 0
        elif(self.order == 2):
            self.target_v = random.uniform(0.0, 1.0)
            self.a = random.uniform(-0.1, 0.1)
            self.asserted_stag = 10
            self.stag = 1
            self.setting_change_prob = 1.0
        
        self.pre_target_v = None
        
        return(self.target_v)
    
    def step(self, net_output):
        observation = self.target_v - net_output[0]
        error = abs(observation)

        if(self.order == 2):
            self.target_v += self.a
            if(self.target_v > 1.0):
                self.target_v = 1.0
            elif(self.target_v < 0.0):
                self.target_v = 0.0
            
            #target_vがasserted_stagで設定したステップ数以上変化していない場合には一定確率でaを変更
            if(self.pre_target_v == self.target_v):
                self.stag += 1
            else:
                self.stag = 1
            if(self.stag >= self.asserted_stag):
                if random.random() < self.setting_change_prob:
                    if(self.target_v == 1.0):
                        self.a = random.uniform(-0.2, 0.0)
                    elif(self.target_v == 0.0):
                        self.a = random.uniform(0.0, 0.2)
                    else:
                        self.a = random.uniform(-0.2, 0.2)

        self.pre_target_v = self.target_v

        self.step_num += 1
        if(self.step_num <= 100):
            done = False
        else:
            done = True

        return observation, error, done