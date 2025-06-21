from deepbots.supervisor import DeepbotsSupervisorEnv
from controller import Robot,Supervisor,Motor,Node
import numpy as np
import socket
import json
import time

class CustomCarEnv(DeepbotsSupervisorEnv):

     # --- 1. Costanti di configurazione per un tuning più semplice ---
    MAX_TIMESTEPS = 1000
    TARGET_THRESHOLD = 3 # Aumentato leggermente per facilitare il raggiungimento
    COLLISION_THRESHOLD = 0.6 # Distanza minima prima di considerare una collisione
    STALL_LIMIT = 150 # Numero di step a velocità quasi nulla prima di terminare    
    MAX_SPEED = 30 #rad/s
    MAX_STEER_ANGLE = 0.55 # radianti (circa 34 gradi)
    ROAD_WIDTH = 12
    ROAD_LENGTH = 120

    robot = Supervisor()

    def __init__(self):
        self.total_timesteps = 0
        self.episode = 0
        #initializing obstacles
        self.num_obstacles = 4
        self.obstacles = []
        self.cumulative_reward = 0.0
        for i in range(self.num_obstacles):
            self.obstacles.append(self.robot.getFromDef(f"obstacle_{i+1}"))
        
        self.timestep = int(self.robot.getBasicTimeStep())

        #initializing target
        self.target = self.robot.getFromDef('target')
        self.target_position = self.target.getField('translation').getSFVec3f()
        self.TARGET_X = self.target_position[0]
        self.TARGET_Y = self.target_position[1]

        #initializing the car
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
        
        steer_angle = action[1] * self.MAX_STEER_ANGLE

        self.left_motor.setVelocity(target_velocity)
        self.right_motor.setVelocity(target_velocity)
        self.left_steer.setPosition(steer_angle)
        self.right_steer.setPosition(steer_angle)

        # Avanza la simulazione
        if self.robot.step(self.timestep) == -1:
            return None, 0.0, True, True, {} # obs, reward, terminated, truncated, info

        self.current_timestep += 1
        self.total_timesteps += 1
        # Ottieni nuove osservazioni
        obs = self.get_obs()
        
        # Calcola la reward e controlla se l'episodio è terminato
        reward, terminated = self.get_reward(obs, action)
        
        # Controlla se l'episodio deve essere troncato (es. time limit)
        truncated = self.current_timestep >= self.MAX_TIMESTEPS
        if truncated:
            print("--- Episodio troncato per limite di tempo ---")
        
        done = terminated or truncated
        '''
        print_every = 50
        if((self.current_timestep % print_every) == 0):
            print("=====================================")
            print(f"EPISODE: {self.episode}")
            print(f"STEP {self.current_timestep} ")  # DEBUG
            print(f"OBS: {obs}")
            print(f"REWARD: {reward}")
            print(f"ACTION: {action}")
            print("=====================================")
        '''
        print("=====================================")
        print(f"EPISODE: {self.episode}")
        print(f"CUMULATIVE REWARD: {self.cumulative_reward}")
        print(f"TOT TIMESTEPS: {self.total_timesteps}")
        print("=====================================")
        return obs, reward, done, {} # Manteniamo l'output standard di `step` (obs, reward, done, info)
        

    def get_obs(self):
        
        # Velocità delle ruote
        v_left = self.left_motor.getVelocity()
        v_right = self.right_motor.getVelocity()
        mean_v = (v_left + v_right)/2
        normalized_v = normalize_to_range(mean_v,0,self.MAX_SPEED,0.0,1.0)

        # GPS e IMU
        gps_values = self.gps.getValues()
        normalized_gps_x = normalize_to_range(gps_values[0],0,self.ROAD_LENGTH,0.0,1.0)
        normalized_gps_y = normalize_to_range(gps_values[1],-self.ROAD_WIDTH/2,self.ROAD_WIDTH/2,-1.0,1.0)
        normalized_gps_z = normalize_to_range(gps_values[2],0,1,0.0,1.0,True)
        normalized_gps = [normalized_gps_x,normalized_gps_y,normalized_gps_z]

        imu_values = self.imu.getRollPitchYaw()
        imu_yaw = imu_values[2]
        normalized_imu_yaw = normalize_to_range(imu_yaw,-np.pi,np.pi,-1.0,1.0)
        # --- 4. Processamento Lidar più efficiente ---
        # step 1: prepara la range image
        lidar_raw = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar_raw[lidar_raw == np.inf] = self.lidar_max_range

        res = self.lidar.getHorizontalResolution()
        layers = self.lidar.getNumberOfLayers()
        sectors = 10
        sector_size = res // sectors

        lidar_sectors = []

        for i in range(sectors):
            all_layer_sector = []

            for l in range(layers):
                start = l * res + i * sector_size
                end = start + sector_size
                sector_slice = lidar_raw[start:end]
                all_layer_sector.append(np.min(sector_slice))  # puoi anche considerare np.mean()

            # step 2: informazione utile = minimo dei layer
            sector_value = min(all_layer_sector)

            # step 3: amplifica i settori più pericolosi (vicini)
            danger_amplified = (self.lidar_max_range - sector_value) / self.lidar_max_range
            lidar_sectors.append(danger_amplified)

        normalized_lidar = np.clip(np.array(lidar_sectors), 0.0, 1.0)

        target_distance  = np.linalg.norm([
            self.TARGET_X - gps_values[0],
            self.TARGET_Y - gps_values[1]
        ])
        normalized_target_distance = normalize_to_range(target_distance,0,self.ROAD_LENGTH,0.0,1.0)

        # Concatenazione di tutte le osservazioni
        obs = np.concatenate([
            [normalized_v], #mean velocity
            normalized_gps, #
            [normalized_target_distance],
            [normalized_imu_yaw], #only the imu_yaw
            normalized_lidar #10 values 
        ]).astype(np.float32)
        
        return obs


    def get_reward(self, obs, action):

        # Osservazioni: Assumendo l'ordine in get_obs()
        # [0]: tesla_mean_velocity  ---> [0,1]
        # [1]: gps_x ---> [0,1]
        # [2]: gps_y ---> [-1,1]
        # [3]: gps_z ---> [-1,1]
        # [4]: target_distance ---> [0,1]
        # [5]: tesla_imu_yaw ---> [-1,1]
        # [6-15]: lidar  ---> [0,1]

        #action space
        # [0]: velocity
        # [1]: steer angle

        # --- Estrazione ---
        terminated = False
        reward = 0.0
        # --- Parametri ---
        steer = action[1]
        mean_v = obs[0] * self.MAX_SPEED
        gps_x = obs[1] * self.ROAD_LENGTH
        gps_y = obs[2] * (self.ROAD_WIDTH / 2)
        gps_z = obs[3] * 1
        target_distance = obs[4] * self.ROAD_LENGTH
        tesla_yaw = obs[5] * np.pi
        lidar = obs[6:16] #contains the risk factor

        # --- 1. Progresso verso il target ---
        progress = self.previous_distance_to_target - target_distance
        reward += progress * 55
        self.previous_distance_to_target = target_distance

        # --- 3. Penalità sterzate forti ---
        reward -= abs(steer) * 1.0  # oppure un valore più basso tipo 0.5

        reward -= abs(tesla_yaw) * 2

        # --- 4. Penalità prossimità ostacoli ---
        risk_score = np.sum(np.array(lidar) ** 2)
        reward -= risk_score * 15.0

        weights = np.array([1.25, 1.75, 2, 3, 4, 4, 3, 2, 1.75, 1.25])
        directional_penalty = np.sum(weights * (np.array(lidar) ** 2))
        reward -= directional_penalty * 3.5

        if np.max(lidar) > 0.95:
            reward -= 50.0
            terminated = True

        # --- 6. Penalità costante temporale ---
        reward -= 0.1

        # --- 7. Penalità / terminazione eventi critici ---

        if abs(gps_y) > (self.ROAD_WIDTH/2) or gps_z < 0 or gps_z > 1.0:
            print("--- Fuori strada ---")
            reward -= 50.0
            terminated = True


        if mean_v < 1:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        if self.stall_counter >= self.STALL_LIMIT:
            print("--- Stallo ---")
            reward -= 50.0
            terminated = True

        if target_distance < self.TARGET_THRESHOLD or abs(self.TARGET_X - gps_x) < 2:
            print("--- Obiettivo raggiunto ---")
            reward += 300.0
            terminated = True

        reward = reward * 0.1
        self.cumulative_reward += reward

        return reward, terminated

    def apply_UDR_to_obstacles(self): #to apply uniform domain randomization
        
        range_x = [10,90]
        range_y = [-self.ROAD_WIDTH/2 + 1 ,self.ROAD_WIDTH/2 -1]
        z = 0.4

        placed_positions = []

        for obstacle in self.obstacles:
            is_position_valid = False
            while not is_position_valid:
                # Genera una nuova posizione casuale
                new_x = np.random.uniform(range_x[0], range_x[1])
                new_y = np.random.uniform(range_y[0], range_y[1])
                new_position = [new_x, new_y, z]

                # Controlla che non sia troppo vicino a un altro ostacolo
                is_overlapping = any(
                    np.linalg.norm(np.array(new_position[:2]) - np.array(pos[:2])) < 5
                    for pos in placed_positions
                )

                if not is_overlapping:
                    is_position_valid = True

            # Applica la nuova posizione valida e salvala
            obstacle.getField('translation').setSFVec3f(new_position)
            placed_positions.append(new_position)
    
    def apply_UDR_to_tesla(self):
        
        range_x = [0,5]
        range_y = [-2,2]
        new_z = 0.6
        range_yaw = [-0.3,0.3] #in radiants

        new_x = np.random.uniform(range_x[0],range_x[1])
        new_y = np.random.uniform(range_y[0],range_y[1])
        new_yaw = np.random.uniform(range_yaw[0],range_yaw[1])

        new_pos = [new_x,new_y,new_z]
        new_rot = [0.0,0.0,1.0,new_yaw]

        self.tesla_translation.setSFVec3f(new_pos)
        self.tesla_rotation.setSFRotation(new_rot)


    def reset(self):

        # Resetta la posizione e velocità della Tesla
        initial_translation = [0.0, -2.0, 0.6]
        initial_rotation = [0.0, 0.0, 1.0, 0.0]
        self.tesla_translation.setSFVec3f(initial_translation)
        self.tesla_rotation.setSFRotation(initial_rotation)
        self.car_node.setVelocity([0, 0, 0, 0, 0, 0])
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        #if(self.total_timesteps >= 10000):
        self.apply_UDR_to_obstacles()
        #self.apply_UDR_to_tesla()

        # Resetta la simulazione
        self.robot.simulationResetPhysics()
        self.cumulative_reward = 0.0
        self.robot.step(self.timestep * 5) # Lascia stabilizzare la simulazione
        
        # Resetta le variabili di stato dell'episodio
        self.current_timestep = 0
        self.stall_counter = 0
        self.episode += 1
        # Calcola la distanza iniziale dal target
        initial_obs = self.get_obs()
        gps_pos = initial_obs[1:4]
        gps_x = gps_pos[0] * self.ROAD_LENGTH
        gps_y = gps_pos[1] * (self.ROAD_WIDTH / 2)
        self.previous_distance_to_target = np.linalg.norm([
            self.TARGET_X - gps_x,
            self.TARGET_Y - gps_y
        ])

        return initial_obs

def normalize_to_range(value, min_val, max_val, new_min=0.0, new_max=1.0, clip=False):
    """
    Normalizza 'value' da [min_val, max_val] a [new_min, new_max].

    Args:
        value (float): valore da normalizzare
        min_val (float): valore minimo del range originale
        max_val (float): valore massimo del range originale
        new_min (float): valore minimo del nuovo range (default: 0.0)
        new_max (float): valore massimo del nuovo range (default: 1.0)
        clip (bool): se True, forza il valore normalizzato a stare nel nuovo intervallo

    Returns:
        float: valore normalizzato
    """
    if max_val == min_val:
        raise ValueError("Failed to normalize , min_val = max_val !")

    normalized = (value - min_val) * (new_max - new_min) / (max_val - min_val) + new_min

    if clip:
        return np.clip(normalized, new_min, new_max)
    return normalized

# --- Socket server per comunicazione RL esterna ---
if __name__=='__main__':
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

