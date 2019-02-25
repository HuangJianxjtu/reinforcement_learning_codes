import logging
import numpy
import random
from gym import spaces
import gym

logger = logging.getLogger(__name__)

class GridMazeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.states = list(range(1,18+1)) #状态空间
        self.x=[150,250,350,550, 150,250,350,550, 350,450,550, 150,250,350,450,550, 150,250]
        self.y=[500,500,500,500, 400,400,400,400, 300,300,300, 200,200,200,200,200, 100,100]
        self.terminate_states = dict()  #终止状态为字典格式
        self.terminate_states[11] = 1

        self.actions = ['n','e','s','w']

        self.rewards = dict()        #回报的数据结构为字典
        self.rewards['8_s'] = 1.0
        self.rewards['10_e'] = 1.0
        self.rewards['16_n'] = 1.0

        self.t = dict()             #状态转移的数据格式为字典
        self.t['1_s'] = 5
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['2_s'] = 6
        self.t['3_w'] = 2
        self.t['3_s'] = 7
        self.t['4_s'] = 8
        self.t['5_n'] = 1
        self.t['5_e'] = 6
        self.t['6_w'] = 5
        self.t['6_n'] = 2
        self.t['6_e'] = 7
        self.t['7_n'] = 3
        self.t['7_s'] = 9
        self.t['7_w'] = 6
        self.t['8_n'] = 4
        self.t['8_s'] = 11
        self.t['9_n'] = 7
        self.t['9_e'] = 10
        self.t['9_s'] = 14
        self.t['10_e'] = 11
        self.t['10_s'] = 15
        self.t['10_w'] = 9
        self.t['12_e'] = 13
        self.t['12_s'] = 17
        self.t['13_e'] = 14
        self.t['13_s'] = 18
        self.t['13_w'] = 12
        self.t['14_n'] = 9
        self.t['14_e'] = 15
        self.t['14_w'] = 13
        self.t['15_n'] = 10
        self.t['15_e'] = 16
        self.t['15_w'] = 14
        self.t['16_n'] = 11
        self.t['16_w'] = 15
        self.t['17_n'] = 12
        self.t['17_e'] = 18
        self.t['18_n'] = 13
        self.t['18_w'] = 17


        self.gamma = 0.8         #折扣因子(*)
        self.viewer = None
        self.state = None

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions
    
    def getTerminate_states(self):
        return self.terminate_states

    def getState(self):
        return self.state

    def setState(self,s):
        self.state=s

    def getReward(self,s,a):  # at state s and take action a, this function will return the respect reward
        key = key = "%d_%s"%(s,a)
        if key in self.rewards:
            return self.rewards[key]
        else:
            return 0

    def transform(self,s,a):
        key = "%d_%s"%(s, a)
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = s
        return next_state
         
    def _step(self, action):    # ***物理引擎***      
        state = self.state  #系统当前状态
        if state in self.terminate_states:
            print("congratulations! We arrive at the terminal!!!")
            return state, 0, True, {}   # 返回下一时刻的动作,回报,是否终止和调试信息(一般调试信息为空)
        key = "%d_%s"%(state, action)   # 将状态和动作组成字典的键值
        #状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
            print(key+":invalid step, so I don't move!")
        self.state = next_state   #完成状态转移
        is_terminal = False
        if next_state in self.terminate_states:
            is_terminal = True
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]
        return next_state, r,is_terminal,{}    # 返回下一时刻的动作,回报,是否终止和调试信息(一般调试信息为空)

    def _reset(self):
        self.state = self.states[int(random.random() * len(self.states))]  #
        return self.state
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _render(self, mode='human', close=False):   # ***图像引擎***  
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 700
        screen_height = 600
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #创建网格世界
            self.line1 = rendering.Line((100, 50), (600, 50))
            self.line2 = rendering.Line((100, 150), (600, 150))
            self.line3 = rendering.Line((100, 250), (600, 250))
            self.line4 = rendering.Line((100, 350), (600, 350))
            self.line5 = rendering.Line((100, 450), (600, 500-50))
            self.line6 = rendering.Line((100, 550), (600, 550))
            self.line7 = rendering.Line((100, 50), (100, 550))
            self.line8 = rendering.Line((200, 50), (200, 550))
            self.line9 = rendering.Line((300, 50), (300, 550))
            self.line10 = rendering.Line((400, 50), (400, 550))
            self.line11 = rendering.Line((500, 50), (500, 550))
            self.line12 = rendering.Line((600, 50), (600, 550))
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            # 创建第一个黑色区域
            self.rectangle1 = rendering.make_polygon([(100,250),(100,350),(300,350),(300,250)])
            self.rectangle1.set_color(0, 0, 0)
            self.viewer.add_geom(self.rectangle1)
            # 创建第二个黑色区域
            self.rectangle2 = rendering.make_polygon([(400,350),(400,550),(500,550),(500,350)])
            self.rectangle2.set_color(0, 0, 0)
            self.viewer.add_geom(self.rectangle2)
            # 创建第三个黑色区域
            self.rectangle3 = rendering.make_polygon([(300,50),(300,150),(600,150),(600,50)])
            self.rectangle3.set_color(0, 0, 0)
            self.viewer.add_geom(self.rectangle3)
            # exit/出口
            self.square =  rendering.make_polygon([(500,250),(500,350),(600,350),(600,250)])
            self.square.set_color(0, 50, 50)
            self.viewer.add_geom(self.square)
            # 创建机器人
            self.robot = rendering.make_circle(40)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(50, 0, 0)
            self.viewer.add_geom(self.robot)
        # 更新机器人位置
        if self.state is None: return None
        self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')