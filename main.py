import pybullet as p
import time
import numpy as np
import math
import matplotlib.pyplot as plt

GUI = False

g = 10 # ускорение свободного падения
m = 1 # масса маятника
L = 0.5 # длина маятника
q0 = 0.0 # начальное положение
qd = math.pi/2  # желаемое положение


dt = 1 / 240  # шаг симуляции

maxTime = 3 # время симуляции
T = 2 # время движения
kf = 0.1

kp = 5005
kv = 100
ki = 0

if (GUI):
    physicsClient = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=180,
        cameraPitch=0,
        cameraTargetPosition=[0, 0, 1]
    )
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0], 4)
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0], 4)
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1], 4)
else:
    physicsClient = p.connect(p.DIRECT)

p.setGravity(0, 0, -g)
bodyId = p.loadURDF("./pendulum.urdf")

pos = q0
vel = 0
idx = 1

logTime = np.arange(0.0, maxTime, dt)
sz = logTime.size
logPos = np.zeros(sz)
logVel = np.zeros(sz)
logAcc = np.zeros(sz)
logCtr = np.zeros(sz - 1)
logddd = np.zeros(sz)
logPos[0] = q0
logRef = np.zeros(sz)
logRef[0] = q0
logRefd = np.zeros(sz)
logRefdd = np.zeros(sz)
logRefddd = np.zeros(sz)

p.changeDynamics(bodyId, 1, linearDamping=0)

p.setJointMotorControl2(bodyIndex=bodyId,
                        jointIndex=1,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=q0)
for _ in range(1000):
    p.stepSimulation()

q0 = p.getJointState(bodyId, 1)[0]

p.setJointMotorControl2(bodyIndex=bodyId,
                        jointIndex=1,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0,
                        force=0)


def degree_3_interpol(q0, qd, T, t):
    a0 = 0
    a1 = 0
    a2 = 3 / T ** 2
    a3 = -2 / T ** 3
    s = a3 * t ** 3 + a2 * t ** 2 + a1 * t + a0
    ds = 3 * a3 * t ** 2 + 2 * a2 * t + a1
    dds = 6 * a3 * t + 2 * a2
    q = q0 + (qd - q0) * s
    dq = (qd - q0) * ds
    ddq = (qd - q0) * dds
    return (q, dq, ddq) if (t <= T) else (qd, 0, 0)


def degree_5_interpol(q0, qd, T, t):
    a3 = 10 / T ** 3
    a4 = -15 / T ** 4
    a5 = 6 / T ** 5
    s = a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    ds = 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
    dds = 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3
    diff = qd-q0
    q = q0 + diff * s
    dq = diff * ds
    ddq = diff * dds
    return (q, dq, ddq) if (t <= T) else (qd, 0, 0)


def degree_7_interpol(q0, qd, T, t):
    a4 = 35 / T ** 4
    a5 = -84 / T ** 5
    a6 = 70 / T ** 6
    a7 = -20 / T ** 7
    s = a4 * t ** 4 + a5 * t ** 5 + a6 * t ** 6 + a7 * t ** 7
    ds = 4 * a4 * t ** 3 + 5 * a5 * t ** 4 + 6 * a6 * t ** 5 + 7 * a7 * t ** 6
    dds = 12 * a4 * t ** 2 + 20 * a5 * t ** 3 + 30 * a6 * t ** 4 + 42 * a7 * t ** 5
    ddds = 24 * a4 * t + 60 * a5 * t ** 2 + 120 * a6 * t ** 3 + 210 * a7 * t ** 4
    diff = qd - q0
    q = q0 + diff * s
    dq = diff * ds
    ddq = diff * dds
    dddq = diff * ddds
    return (q, dq, ddq, dddq) if (t <= T) else (qd, 0, 0, 0)
   

def feedback_lin(pos, vel, posd, veld, accd):
    u = -kp * (pos - posd) - kv * vel
    ctrl = m * L * L * ((g / L) * math.sin(pos) + kf / (m * L * L) * vel + u)
    return ctrl


prev_vel = 0
prev_acc=0
for t in logTime[1:]:
    (posd, veld, accd, ddd_r) = degree_7_interpol(q0, qd, T, t)
    ctrl = feedback_lin(pos, vel, posd, veld, accd)

    p.setJointMotorControl2(bodyIndex=bodyId,
                            jointIndex=1,
                            controlMode=p.TORQUE_CONTROL,
                            force=ctrl)
    p.stepSimulation()
    pos = p.getJointState(bodyId, 1)[0]
    vel = p.getJointState(bodyId, 1)[1]
    acc = (vel - prev_vel) / dt
    ddd=(acc-prev_acc)/dt
    prev_vel = vel
    prev_acc = ddd

    logRef[idx] = posd
    logRefd[idx] = veld
    logRefdd[idx] = accd
    logRefddd[idx] = ddd_r

    logPos[idx] = pos
    logVel[idx] = vel
    logAcc[idx] = acc
    logddd[idx] = ddd
    logCtr[idx - 1] = ctrl

    idx += 1
    if (GUI):
        time.sleep(dt)

p.disconnect()





plt.subplot(5, 1, 1)
plt.grid(True)
plt.plot(logTime, logPos, label="Pos")
plt.plot(logTime, logRef, label="PosRef")
plt.legend()

plt.subplot(5, 1, 2)
plt.grid(True)
plt.plot(logTime, logVel, label="Vel")
plt.plot(logTime, logRefd, label="VelRef")
plt.legend()

plt.subplot(5, 1, 3)
plt.grid(True)
plt.plot(logTime, logAcc, label="Acc")
plt.plot(logTime, logRefdd, label="AccRef")
plt.legend()

plt.subplot(5, 1, 4)
plt.grid(True)
plt.plot(logTime, logddd, label="ddd")
plt.plot(logTime, logRefddd, label="dddRef")
plt.legend()

plt.subplot(5, 1, 5)
plt.grid(True)
plt.plot(logTime[0:-1], logCtr, label="Ctr")
plt.legend()

plt.show()
