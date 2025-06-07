from deepbots.supervisor import DeepbotsSupervisorEnv
from controller import Robot,Supervisor,Motor,Node
import numpy as np
import socket
import json
import time

class CustomCarEnv(DeepbotsSupervisorEnv):

    # --- 1. Costanti di configurazione per un tuning più semplice ---
    MAX_TIMESTEPS = 1000
    TARGET_X = 80.0
    TARGET_Y = 0.0
    TARGET_THRESHOLD = 3.0 # Aumentato leggermente per facilitare il raggiungimento
    COLLISION_THRESHOLD = 0.4 # Distanza minima prima di considerare una collisione
    STALL_LIMIT = 200 # Numero di step a velocità quasi nulla prima di terminare

    # --- Pesi per la funzione di reward ---
    REWARD_GOAL = 100.0
    REWARD_PROGRESS_MULTIPLIER = 50
    REWARD_FORWARD_VELOCITY = 0.2
    PENALTY_COLLISION = -100.0
    PENALTY_OFF_ROAD = -50.0
    PENALTY_STALL = -20.0
    PENALTY_STEERING = 0.5 # Piccola penalità per sterzate eccessive
    PENALTY_TIME = -0.1 # Piccola penalità per ogni timestep per incentivare la velocità
    
    robot = Supervisor()

    def __init__(self):
        #super().__init__()
        
        
        self.timestep = int(self.robot.getBasicTimeStep())

        self.MAX_SPEED = 100
        self.MAX_STEER_ANGLE = 0.6 # radianti (circa 34 gradi)

        self.car_node = self.robot.getFromDef('tesla')
        self.tesla_translation = self.car_node.getField('translation')
        self.tesla_rotation = self.car_node.getField('rotation')

        self.left_motor = self.robot.getDevice('left_rear_wheel')
        self.right_motor = self.robot.getDevice('right_rear_wheel')
        self.left_steer = self.robot.getDevice('left_steer')
        self.right_steer = self.robot.getDevice('right_steer')
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Sensori
        self.gps = self.robot.getDevice('gps')
        self.imu = self.robot.getDevice('inertial unit')
        self.lidar = self.robot.getDevice('lidar')
        
        self.gps.enable(self.timestep)
        self.imu.enable(self.timestep)
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        
        self.lidar_horizontal_resolution = self.lidar.getHorizontalResolution()
        self.lidar_max_range = self.lidar.getMaxRange()
        self.num_lidar_sectors = 10 # 10 settori per i dati Lidar

        # --- Inizializzazione stato episodio ---
        self.current_timestep = 0
        self.stall_counter = 0
        self.previous_distance_to_target = 0.0

        print("Ambiente CustomCarEnv inizializzato correttamente.")

        self.reset()

    def step(self, action):
        
        # Applica l'azione
        target_velocity = action[0] * self.MAX_SPEED
        # Incentiva a non andare in retromarcia se non necessario
        if target_velocity < 0: target_velocity = 0 
        
        steer_angle = action[1] * self.MAX_STEER_ANGLE

        self.left_motor.setVelocity(target_velocity)
        self.right_motor.setVelocity(target_velocity)
        self.left_steer.setPosition(steer_angle)
        self.right_steer.setPosition(steer_angle)

        # Avanza la simulazione
        if self.robot.step(self.timestep) == -1:
            return None, 0.0, True, True, {} # obs, reward, terminated, truncated, info

        self.current_timestep += 1

        # Ottieni nuove osservazioni
        obs = self.get_obs()
        
        # Calcola la reward e controlla se l'episodio è terminato
        reward, terminated = self.get_reward(obs, action)
        
        # Controlla se l'episodio deve essere troncato (es. time limit)
        truncated = self.current_timestep >= self.MAX_TIMESTEPS
        if truncated:
            print("--- Episodio troncato per limite di tempo ---")
        
        done = terminated or truncated
        
        print_every = 50
        if((self.current_timestep % print_every) == 0):
            print(f"STEP {self.current_timestep}: ACTION = {action}  REWARD: {reward}")  # DEBUG

        return obs, reward, done, {} # Manteniamo l'output standard di `step` (obs, reward, done, info)
        

    def get_obs(self):
        
        # Velocità delle ruote
        v_left = self.left_motor.getVelocity()
        v_right = self.right_motor.getVelocity()

        # GPS e IMU
        gps_values = self.gps.getValues()
        imu_values = self.imu.getRollPitchYaw()
        
        # Rotazione (asse-angolo)
        rotation_values = self.tesla_rotation.getSFRotation()

        # --- 4. Processamento Lidar più efficiente ---
        lidar_raw = self.lidar.getRangeImage()
        if not lidar_raw:
            lidar_sectors = np.full(self.num_lidar_sectors, self.lidar_max_range, dtype=np.float32)
        else:
            lidar_full_range = np.array(lidar_raw, dtype=np.float32)
            # Prendiamo solo i dati centrali, spesso i più rilevanti per la guida
            central_layer = lidar_full_range[self.lidar_horizontal_resolution : 2*self.lidar_horizontal_resolution]
            
            # Sostituiamo inf con max_range per i calcoli
            central_layer[central_layer == np.inf] = self.lidar_max_range
            
            # Dividiamo in settori e prendiamo la distanza minima per settore
            sector_size = self.lidar_horizontal_resolution // self.num_lidar_sectors
            lidar_sectors = [np.min(central_layer[i*sector_size:(i+1)*sector_size]) for i in range(self.num_lidar_sectors)]
        
        # Normalizzazione Lidar [0, 1]
        normalized_lidar = np.clip(np.array(lidar_sectors) / self.lidar_max_range, 0.0, 1.0)

        # Concatenazione di tutte le osservazioni
        obs = np.concatenate([
            [v_left, v_right],
            gps_values,
            rotation_values,
            imu_values,
            normalized_lidar
        ]).astype(np.float32)
        
        return obs

    def get_reward(self, obs, action):

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
        # [12-21]: lidar data

        #action space
        # [0]: velocity
        # [1]: steer angle

        # --- Estrazione ---
        terminated = False
        reward = 0.0
        
        gps_pos = obs[2:5]
        lidar_min_dist = np.min(obs[12:]) * self.lidar_max_range # De-normalizza per il controllo
        avg_speed = (obs[0] + obs[1]) / 2.0

        # --- 5. Funzione di Reward Semplice ed Efficace ---

        # 1. Penalità di collisione (evento terminale)
        if lidar_min_dist < self.COLLISION_THRESHOLD:
            print(f"--- Fine episodio: Collisione! (dist: {lidar_min_dist:.2f}m) ---")
            terminated = True
            return self.PENALTY_COLLISION, terminated

        # 2. Penalità per uscita di strada (evento terminale)
        # Assumiamo che la strada sia attorno a y=0 e z=0.4
        if abs(gps_pos[1]) > 4.0 or gps_pos[2] < 0.1 or gps_pos[2] > 1.0:
            print(f"--- Fine episodio: Uscita di strada! (y: {gps_pos[1]:.2f}, z: {gps_pos[2]:.2f}) ---")
            terminated = True
            return self.PENALTY_OFF_ROAD, terminated
            
        # 3. Controllo dello stallo (evento terminale)
        if abs(avg_speed) < 0.1:
            self.stall_counter += 1
        else:
            self.stall_counter = 0 # Resetta se si muove
        
        if self.stall_counter >= self.STALL_LIMIT:
            print("--- Fine episodio: Stallo prolungato ---")
            terminated = True
            return self.PENALTY_STALL, terminated

        # 4. Raggiungimento del target (evento terminale con grande reward)
        current_distance_to_target = np.linalg.norm([self.TARGET_X - gps_pos[0], self.TARGET_Y - gps_pos[1]])
        if current_distance_to_target < self.TARGET_THRESHOLD:
            print("--- OBIETTIVO RAGGIUNTO! ---")
            terminated = True
            return self.REWARD_GOAL, terminated

        # --- Se l'episodio non è terminato, calcoliamo le reward intermedie ---
        
        # Reward per il progresso verso il target (fondamentale)
        progress = self.previous_distance_to_target - current_distance_to_target
        reward += progress * self.REWARD_PROGRESS_MULTIPLIER
        self.previous_distance_to_target = current_distance_to_target
        
        # Reward per mantenere una buona velocità in avanti
        reward += self.REWARD_FORWARD_VELOCITY * np.clip(avg_speed, 0, self.MAX_SPEED)

        # Piccola penalità per sterzate brusche (incoraggia guida fluida)
        steer_angle_action = action[1]
        reward -= self.PENALTY_STEERING * abs(steer_angle_action)
        
        # Piccola penalità costante per ogni timestep (incentiva a finire prima)
        reward += self.PENALTY_TIME
        
        return reward, terminated

        
    def reset(self):

        # Resetta la posizione e velocità della Tesla
        initial_translation = [0.0, -2.0, 0.6]
        initial_rotation = [0.0, 0.0, 1.0, 0.0]
        self.tesla_translation.setSFVec3f(initial_translation)
        self.tesla_rotation.setSFRotation(initial_rotation)
        self.car_node.setVelocity([0, 0, 0, 0, 0, 0])
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Resetta la simulazione
        self.robot.simulationResetPhysics()
        #self.robot.simulationReset()

        self.robot.step(self.timestep * 5) # Lascia stabilizzare la simulazione
        
        # Resetta le variabili di stato dell'episodio
        self.current_timestep = 0
        self.stall_counter = 0
        
        # Calcola la distanza iniziale dal target
        initial_obs = self.get_obs()
        gps_pos = initial_obs[2:5]
        self.previous_distance_to_target = np.linalg.norm([
            self.TARGET_X - gps_pos[0],
            self.TARGET_Y - gps_pos[1]
        ])

        return initial_obs


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
                    env.robot.simulationSetMode(0)
                    env.robot.simulationReset()
    
                    break

except Exception as e:
    print(f"Errore nel server socket: {e}")
finally:
    print("Chiusura del controller Webots.")

