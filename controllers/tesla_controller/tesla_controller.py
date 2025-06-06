from deepbots.supervisor import DeepbotsSupervisorEnv
from controller import Robot,Supervisor,Motor,Node
import numpy as np
import socket
import json
import time

class CustomCarEnv(DeepbotsSupervisorEnv):
    def __init__(self):

        self.tesla_node = Supervisor()
        self.MAX_TIMESTEPS = 300
        self.timestep = int(self.tesla_node.getBasicTimeStep())
        print(f"Timestep: {self.timestep} ms")
        
        self.left_motor = self.tesla_node.getDevice('left_rear_wheel')
        self.right_motor = self.tesla_node.getDevice('right_rear_wheel')
        self.car = self.tesla_node.getFromDef('tesla')
        self.tesla_translation = self.car.getField('translation')
        self.tesla_rotation = self.car.getField('rotation')
        
        if self.left_motor is None or self.right_motor is None:
            print("ERRORE: Motori non trovati. Assicurati di usare un nodo basato su 'Car' come TeslaModel3.")
            exit()
        
        self.currentTimestep = 0
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Sensors
        self.gps = self.tesla_node.getDevice('gps')
        if self.gps is not None:
            self.gps.enable(self.timestep)
        else:
            print("AVVISO: GPS non trovato.")

        self.imu = self.tesla_node.getDevice('inertial unit')
        

        #target
        self.target_x = 50.0  # Esempio: Coordinata X del target
        self.target_y = 0.0  # Esempio: Coordinata Y del target
        self.target_z = 0.4  # Esempio: Coordinata Z del target (di solito l'altezza sul terreno)
        self.target_threshold = 2 # Metri: quanto vicino deve essere l'auto al target per considerarlo raggiunto


        if self.imu is not None:
            self.imu.enable(self.timestep)
        else:
            print("AVVISO: Inertial Unit non trovata.")
        
        print("Inizializzazione completata.")  # DEBUG

        

        self.reset()

    def step(self, action): #on gym the step method returns: obs,reward,done,info
        
        left_speed, right_speed = action[0],action[1]
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        #print(f"STEP: azione ricevuta = {action}")  # DEBUG
        
        if self.tesla_node.step(self.timestep) == -1:
            print("Simulazione interrotta durante lo step.")  # DEBUG
            return None, 0.0, True, {}

        current_observations = self.get_obs()
        reward = self._compute_reward(current_observations, action)
        self.currentTimestep += 1
        done = self.is_done(current_observations)
        print(f"STEP {self.currentTimestep}: reward = {reward:.4f}, done = {done}")  # DEBUG

        return current_observations, reward, done, {}

    def get_obs(self):
        
        tesla_velocity_left = np.array([self.left_motor.getVelocity()],dtype=np.float32)
        tesla_velocity_right = np.array([self.right_motor.getVelocity()],dtype=np.float32)
        #tesla rotation and translation (x,y,z)
        #self.tesla_translation.getSFVec3f()
        tesla_rotation = np.array(self.tesla_rotation.getSFRotation(),dtype=np.float32)
        #tesla gps (x,y,z)
        gps_values = [0.0, 0.0, 0.0]
        gps_values = np.array(self.gps.getValues(),dtype=np.float32)
        #tesla imu (angle_x,angle_y,angle_z)
        imu_orientation = [0.0, 0.0, 0.0]
        imu_orientation = np.array(self.imu.getRollPitchYaw(),dtype=np.float32)
        #TODO - LIDAR

        observations = np.concatenate((tesla_velocity_left,tesla_velocity_right,gps_values,
                                      tesla_rotation,imu_orientation),dtype=np.float32)
        
        return observations

    def _compute_reward(self, obs, action):
        
        # Azioni: left_speed, right_speed
        left_speed, right_speed = action[0],action[1]
        
        # Osservazioni: Assumendo l'ordine in get_obs()
        # [0]: tesla_velocity_left
        # [1]: tesla_velocity_right
        # [2]: gps_x
        # [3]: gps_y
        # [4]: gps_z
        # [5]: tesla_rotation_x (asse)
        # [6]: tesla_rotation_y (asse)
        # [7]: tesla_rotation_z (asse)
        # [8]: tesla_rotation_angle
        # [9]: imu_roll
        # [10]: imu_pitch
        # [11]: imu_yaw

        # Ricompensa per la velocità media in avanti
        forward_velocity_reward = (left_speed + right_speed) / 2.0 
        
        # Penalità per deviazione dalla direzione diritta (yaw)
        yaw_angle = obs[11] # L'angolo di yaw dell'IMU
        straight_penalty = -np.abs(yaw_angle) * 0.5 # Aumenta la penalità per deviazioni maggiori

        # Penalità per inclinazioni (pitch e roll)
        roll_angle = obs[9]
        pitch_angle = obs[10]
        tilt_penalty = -(np.abs(roll_angle) + np.abs(pitch_angle)) * 0.1

        # Componente 2: Ricompensa basata sulla distanza dal target
        # Ottieni la posizione corrente della Tesla dal GPS
        current_x = obs[2]
        current_y = obs[3]
        current_z = obs[4]

        # Calcola la distanza euclidea 3D dal target
        distance_to_target = np.sqrt(
            (self.target_x - current_x)**2 +
            (self.target_y - current_y)**2 +
            (self.target_z - current_z)**2
        )

        # Per semplicità, qui penalizziamo la distanza:
        distance_penalty = -distance_to_target * 0.1 # Penalizza più sei lontano

        # Componente 3: Grande ricompensa al raggiungimento del target
        target_reached_bonus = 0.0
        if distance_to_target < self.target_threshold:
            target_reached_bonus = 500.0 # Una ricompensa molto alta


        # Ricompensa totale
        reward = forward_velocity_reward + straight_penalty + tilt_penalty + distance_penalty + target_reached_bonus

        return reward

    def is_done(self, obs):
        done = False
        current_x = obs[2]
        current_y = obs[3]
        current_z = obs[4]

        # Calcola la distanza euclidea 3D dal target
        distance_to_target = np.sqrt(
            (self.target_x - current_x)**2 +
            (self.target_y - current_y)**2 +
            (self.target_z - current_z)**2
        )
        
        if (self.currentTimestep == self.MAX_TIMESTEPS or distance_to_target < self.target_threshold or (current_z < -1 or current_z > 1)):
            done = True

        return done

    def reset(self):

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.tesla_translation.setSFVec3f([0.0,-2.0,0.5])
        self.tesla_rotation.setSFRotation([0.0,0.0,1.0,0.0])
        
        print(f"------ Fine episodio! ----- ")
        self.car.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.currentTimestep = 0

        for _ in range(100):
            if self.tesla_node.step(self.timestep) == -1:
                print("Simulazione interrotta durante il reset.") #DEBUG
                return np.zeros(self.observation_space , dtype=np.float32)
            
        return self.get_obs()


# --- Socket server per comunicazione RL esterna ---
HOST = '127.0.0.1'
PORT = 10000

env = CustomCarEnv()

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print("Controller Webots in ascolto sulla porta", PORT, "...")

        conn, addr = s.accept()
        with conn:
            print(f"Connesso a: {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Client disconnesso.")
                    break

                try:
                    msg = json.loads(data.decode())
                except json.JSONDecodeError:
                    print(f"Errore di decodifica JSON: {data.decode()}")
                    continue

                if msg['cmd'] == 'reset':
                    obs = env.reset()
                    conn.send(json.dumps({'obs': obs.tolist()}).encode())

                elif msg['cmd'] == 'step':
                    obs, reward, done, _ = env.step(msg['action'])
                    conn.send(json.dumps({
                        'obs': obs.tolist(),
                        'reward': float(reward),
                        'done': bool(done)
                    }).encode())

                elif msg['cmd'] == 'exit':
                    print("Comando 'exit' ricevuto.")
                    break

except Exception as e:
    print(f"Errore nel server socket: {e}")
finally:
    print("Chiusura del controller Webots.")