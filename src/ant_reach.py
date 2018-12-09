import numpy as np
import mujoco_py
import gym
import gym.spaces
from collections import deque
import time
import matplotlib.pyplot as plt
from gym import spaces

class AntReach:
    def __init__(self):

        # Simulator objects
        self.modelpath = "/home/silverjoda/SW/python-research/Hexaprom/src/envs/ant_reach.xml"
        self.model = mujoco_py.load_model_from_path(self.modelpath)
        self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim  = self.q_dim + self.qvel_dim
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Gym reference (Environment dimensions)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
        self.action_space =  gym.spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,))

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0
        self.goal = None
        self.goal_dim = 3  # x,y,theta (SE2 group)
        self.n_episodes = 0
        self.success_rate = 0
        self.success_queue = deque(maxlen=100)
        self.xy_dev = 0.2
        self.psi_dev = 0.3
        self.current_pose = None

        # Initial methods
        self.reset()
        self.setupcam()


    def setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 1.3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20


    def get_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        a = qpos + qvel
        return np.asarray(a, dtype=np.float32)


    def get_state(self):
        return self.sim.get_state()


    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1
        init_q[1] = np.random.randn() * 0.1
        init_q[2] = 0.60 + np.random.rand() * 0.1
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        obs = np.concatenate((init_q, init_qvel))

        self.current_pose = self.get_pose(obs)
        self.goal = self._sample_goal(self.current_pose)

        # Set object position
        init_q[self.q_dim - 2:] = self.goal[0:2]

        # Set environment state
        self.set_state(init_q, init_qvel)

        return obs


    def _sample_goal(self, pose):
        while True:
            x, y, psi = pose
            nx = x + np.random.randn() * (2. + 3 * self.success_rate)
            ny = y + np.random.randn() * (2. + 3 * self.success_rate)
            npsi = y + np.random.randn() * (0.3 + 1 * self.success_rate)

            goal = nx, ny, npsi

            if not self.reached_goal(pose, goal):
                break

        return np.array(goal)



    def _update_stats(self, reached_goal):
        self.success_queue.append(1. if reached_goal else 0.)
        self.success_rate = np.mean(self.success_queue)


    def reached_goal(self, pose, goal):
        x,y,psi = pose
        xg,yg,psig = goal
        return (x-xg)**2 < self.xy_dev and (y-yg)**2 < self.xy_dev


    def get_pose(self, obs):
        x,y = obs[0:2]
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*obs[3:7]))
        return x,y,psi


    def render(self, human=True):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        if not human:
            return self.sim.render(camera_name=None,
                                   width=224,
                                   height=224,
                                   depth=False)
            #return viewer.read_pixels(width, height, depth=False)

        self.viewer.render()


    def step(self, ctrl):

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()

        self.step_ctr += 1

        obs = self.get_obs()

        # Make relevant pose from observation (x,y,psi)
        x, y = obs[0:2]
        theta, phi, psi = quaternion.as_euler_angles(np.quaternion(*obs[3:7]))
        pose = (x,y,psi)

        current_dist  = np.linalg.norm(np.asarray(self.current_pose[0:2]) - np.asarray(self.goal[0:2]))
        prev_dist = np.linalg.norm(np.asarray(pose[0:2]) - np.asarray(self.goal[0:2]))

        # Check if goal has been reached
        reached_goal = self.reached_goal(pose, self.goal)

        # Reevaluate termination condition
        done = reached_goal or self.step_ctr > 400

        if reached_goal:
            print("SUCCESS")

        # Update success rate
        if done:
            self._update_stats(reached_goal)

        ctrl_effort = np.square(ctrl).sum() * 0.1
        target_progress = (current_dist - prev_dist) * 10
        target_trueness = 0

        r = 1. if reached_goal else 0.
        r +=  target_progress + target_trueness

        self.current_pose = pose

        return obs, r, done, None




if __name__ == "__main__":
    ant = AntReach()
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()