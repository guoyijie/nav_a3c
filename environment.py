from multiprocessing import Process, Pipe
import sys

import numpy as np
import scipy.misc

from gibson.envs.husky_env import HuskyNavigateEnv
import config

def game_process(conn):
    env = HuskyNavigateEnv(config='gibson_configs/beechwood_c0_rgb_skip50_random_initial_separate.yaml')
    conn.send(0)
    done = False
    while True:
        cmd, arg = conn.recv()
        if cmd=='reset':
            ob = env.reset()
            done = False
            conn.send(ob)
        elif cmd=='action':
            ob, reward, done, info = env.step(arg)
            r, p, yaw = env.robot.body_rpy
            rot_speed = np.array(
                [[np.cos(-yaw), -np.sin(-yaw), 0],
                 [np.sin(-yaw), np.cos(-yaw), 0],
                 [        0,             0, 1]]
            )
            vx, vy, vz = np.dot(rot_speed, env.robot.robot_body.speed())
            avx, avy, avz = np.dot(rot_speed, env.robot.robot_body.angular_speed())
            vel = [vx, vy, vz, avx, avy, avz]
            conn.send([ob, reward, running, vel])
        elif cmd=='observe':
            eye_pos, eye_quat = env.get_eye_pos_orientation()
            pose = [eye_pos, eye_quat]
            ob = env.render_observations(pose)
            conn.send([ob])
        elif cmd=='running':
            running = not done
            conn.send([running])
        elif cmd=='stop':
            break
    env.close()
    conn.send(0)
    conn.close()


class Environment():
    def __init__(self):
        self.conn, child_conn = Pipe()
        self.proc = Process(target=game_process, args=(child_conn,))
        self.proc.start()
        self.conn.recv()
        self.reset()

    def reset(self):
        self.conn.send(['reset', 0])
        obs = self.conn.recv()

    def stop(self):
        self.conn.send(['stop', 0])
        _ = self.conn.recv()
        self.conn.close()
        self.proc.join()

    def preprocess_frame(self, depth):
        #d = depth[16:-16, :] # crop
        #d = d[:, 2:-2] # crop
        #d = d[::13, ::5] # subsample
        d = depth[16:-16,:]
        d = d[:,16:-16]
        d = d[::12, ::12]
        d = d.flatten()
        d = np.power(d/255.0, 10)
        d = np.digitize(d, [0,0.05,0.175,0.3,0.425,0.55,0.675,0.8,1.01])
        d -= 1
        return d

    def step(self, action_idx):
        self.conn.send(['action', action_idx])
        ob, reward, running, vel = self.conn.recv()
        if running:
            rgb = ob['rgb_filled']
            d = self.preprocess_frame(ob['depth'])
            return rgb, d, vel, reward, running
        else:
            rgb, d = 0, 0
            return rgb, d, 0, reward, running

    def frame(self):
        self.conn.send(['observe', 0])
        ob = self.conn.recv()[0]
        return ob['rgb_filled'], self.preprocess_frame(ob['depth'])

    def running(self):
        self.conn.send(['running', 0])
        running = self.conn.recv()
        return running
