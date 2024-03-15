import random
import torch

class Enviroment(object):
    '''
        train_x: a batch of training data eg:image, point cloud, speech etc.
        train_y：data label
        action_space： means number of class
        teacher_list： teacher mapping
        student: student model
        sample: sample data in mini batch
        range: sample range, related batch size
    '''
    def __init__(self, train_x, train_y, action_space, teacher_list, student, sample=1, range=32, opt=None):
        self.train_X = train_x
        self.train_Y = train_y
        self.student = student
        self.action_space = action_space - 1
        self.teacher = teacher_list
        self.sample_times = sample
        self.range = range
        self.opt = opt

    def step(self):
        reward_list = []
        i = 0
        for tea in self.teacher:
            r = 0
            gt = [] # ground true
            data = None
            for j in range(self.sample_times-1):
                if self.sample_times == 1 and j == 0:
                    self.current_index = self._sample_index()
                    data = self.train_X[self.current_index].unsqueeze(0)
                    gt.append(self.current_index)
                elif j == 0:
                    self.current_index = self._sample_index()
                    data1 = self.train_X[self.current_index].unsqueeze(0)
                    gt.append(self.current_index)
                    self.current_index = self._sample_index()
                    data2 = self.train_X[self.current_index].unsqueeze(0)
                    gt.append(self.current_index)
                    data = torch.cat([data1, data2], dim=0)
                    j = j + 1
                else:
                    self.current_index = self._sample_index()
                    data1 = self.train_X[self.current_index].unsqueeze(0)
                    gt.append(self.current_index)
                    data = torch.cat([data, data1], dim=0)
            preact = False
            if self.opt.distill in ['abound']:
                preact = True
            with torch.no_grad():
                tea_feat, tea_action = tea(data, is_feat=True, preact=preact)
            tea_action = tea_action.max(dim=1)[1]
            stu_feat, stu_action = self.student(data, is_feat=True, preact=preact)
            stu_action = stu_action.max(dim=1)[1]
            r = r + self.reward(tea_action.squeeze(), stu_action.squeeze(), gt)
            reward_list.append(r)
            i += 1

        return reward_list.index(max(reward_list)), max(reward_list), reward_list

    # reward
    def reward(self, tea_action, stu_action, indexs):
        r = 0
        for index in range(self.sample_times):
            # print(tea_action[index], stu_action[index])
            r = r + self.check(tea_action[index], stu_action[index], indexs[index])
        return r

    # action
    def sample_actions(self):
        return random.randint(0, self.action_space-1)

    def _sample_index(self):
        return random.randint(0, len(self.train_X)-1)

    def check(self, tea_action, stu_action, index):
        c = self.train_Y[index]
        return 1 if ((c == tea_action and stu_action == c) and tea_action == stu_action) else 0

