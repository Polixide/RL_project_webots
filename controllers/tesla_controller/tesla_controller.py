from deepbots.supervisor import DeepbotsSupervisorEnv
from controller import Robot,Supervisor,Motor,Node
import numpy as np
import socket
import json
import time

class CustomCarEnv(DeepbotsSupervisorEnv):

    tesla_node = Supervisor()
    def __init__(self):

        
        self.MAX_TIMESTEPS = 1000
        self.timestep = int(self.tesla_node.getBasicTimeStep())
        print(f"Timestep: {self.timestep} ms")

        self.left_motor = self.tesla_node.getDevice('left_rear_wheel')
        self.right_motor = self.tesla_node.getDevice('right_rear_wheel')
        self.car = self.tesla_node.getFromDef('tesla')
        self.tesla_translation = self.car.getField('translation')
        self.tesla_rotation = self.car.getField('rotation')

        self.left_steer = self.tesla_node.getDevice('left_steer')
        self.right_steer = self.tesla_node.getDevice('right_steer')

        if self.left_motor is None or self.right_motor is None:
            print("ERRORE: Motori non trovati. Assicurati di usare un nodo basato su 'Car' come TeslaModel3.")
            exit()
        
        self.currentTimestep = 0
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_steer.setPosition(0.0)
        self.right_steer.setPosition(0.0)
        
        # Sensors
        self.gps = self.tesla_node.getDevice('gps')
        if self.gps is not None:
            self.gps.enable(self.timestep)
        else:
            print("AVVISO: GPS not found.")

        self.imu = self.tesla_node.getDevice('inertial unit')
        self.lidar = self.tesla_node.getDevice('lidar')
        #Lidar
        if self.lidar is not None:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            print(f"Lidar enabled")
            self.lidar_horizontal_resolution = self.lidar.getHorizontalResolution()
            self.lidar_number_of_layers = self.lidar.getNumberOfLayers()
            self.lidar_min_range = self.lidar.getMinRange()
            self.lidar_max_range = self.lidar.getMaxRange()
            self.num_lidar_sectors = 10
            self.relevant_lidar_layers_indices = [0,1,2]
        else:
            print("ERRORE: Lidar not found!")
        
        self.stall_counter = 0
        self.stall_limit = 60
        #target
        self.target_x = 50.0  # Esempio: Coordinata X del target
        self.target_y = 0.0  # Esempio: Coordinata Y del target
        self.target_z = 0.4  # Esempio: Coordinata Z del target (di solito l'altezza sul terreno)
        self.target_threshold = 2 # Metri: quanto vicino deve essere l'auto al target per considerarlo raggiunto
        self.previous_distance_to_target = 0.0

        if self.imu is not None:
            self.imu.enable(self.timestep)
        else:
            print("AVVISO: Inertial Unit non trovata.")
        
        print("Inizializzazione completata.")  # DEBUG

        self.reset()

    def step(self, action): #on gym the step method returns: obs,reward,done,info
        
        left_speed, right_speed = action[0],action[1]
        speed = (left_speed + right_speed)/2
        self.left_motor.setVelocity(speed)
        self.right_motor.setVelocity(speed)

        steer_scale_factor = 0.5 #to have a maximum steering angle of 30 degrees
        steer_angle = steer_scale_factor * action[2]
        self.left_steer.setPosition(steer_angle)
        self.right_steer.setPosition(steer_angle)
        
        
        
        if self.tesla_node.step(self.timestep) == -1:
            print("Simulazione interrotta durante lo step.")  # DEBUG
            return None, 0.0, True, {}

        current_observations = self.get_obs()
        reward = self._compute_reward(current_observations, action)
        self.currentTimestep += 1
        done = self.is_done(current_observations)
        print_every = 50
        if((self.currentTimestep % print_every) == 0):
            print(f"STEP {self.currentTimestep}: ACTION = {action}  REWARD: {reward}")  # DEBUG

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
        #telsa lidar
        lidar_data_raw = self.lidar.getRangeImage()
        lidar_obs_features = np.full(self.num_lidar_sectors, self.lidar_max_range, dtype=np.float32)

        if lidar_data_raw is not None:
            lidar_data_np = np.array(lidar_data_raw, dtype=np.float32)
            sector_size = self.lidar_horizontal_resolution // self.num_lidar_sectors

            for i in range(self.num_lidar_sectors):
                start_idx_horizontal = i * sector_size
                end_idx_horizontal = (i + 1) * sector_size
                if i == self.num_lidar_sectors - 1:
                    end_idx_horizontal = self.lidar_horizontal_resolution

                current_sector_min_distance = self.lidar_max_range

                for layer_idx in self.relevant_lidar_layers_indices:
                    layer_offset = layer_idx * self.lidar_horizontal_resolution
                    sector_data_for_layer = lidar_data_np[
                        layer_offset + start_idx_horizontal : layer_offset + end_idx_horizontal
                    ]

                    valid_distances_in_sector = sector_data_for_layer[
                        (sector_data_for_layer > self.lidar_min_range) &
                        (sector_data_for_layer < self.lidar_max_range) &
                        (~np.isnan(sector_data_for_layer)) &
                        (~np.isinf(sector_data_for_layer))
                    ]

                    if len(valid_distances_in_sector) > 0:
                        current_sector_min_distance = min(current_sector_min_distance, np.min(valid_distances_in_sector))

                range_diff = self.lidar_max_range - self.lidar_min_range
                if range_diff > 0:
                    normalized_distance = (current_sector_min_distance - self.lidar_min_range) / range_diff
                else:
                    normalized_distance = 0.0 # O un altro valore di default sensato

                lidar_obs_features[i] = normalized_distance
                lidar_obs_features = np.array(lidar_obs_features,dtype=np.float32)
                
        observations = np.concatenate((tesla_velocity_left,tesla_velocity_right,gps_values,
                                      tesla_rotation,imu_orientation,lidar_obs_features),dtype=np.float32)
        
        return observations

    def _compute_reward(self, obs, action):
        
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
        # [0]: left motor velocity
        # [1]: right motor velocity
        # [2]: steer angle
        # --- Estrazione ---
        gps_x, gps_y, gps_z = obs[2], obs[3], obs[4]
        v_left, v_right = obs[0], obs[1]
        steer_input = action[2]
        lidar_sectors = obs[12:22]
        
        # --- 1. Distanza dal target (più vicino è meglio) ---
        distance = np.linalg.norm([
            self.target_x - gps_x,
            self.target_y - gps_y,
            self.target_z - gps_z
        ])
        distance_reward = 1.0 - np.tanh(distance / 20.0)  # in [0, 1]

        # --- 2. Accelerazione / avanzamento ---
        avg_speed = (v_left + v_right) / 2.0
        acceleration_reward = np.clip(avg_speed / 10.0, 0.0, 1.0)

        # --- 3. Sterzata brusca ---
        steering_penalty = 1.0 - min(abs(steer_input), 1.0)  # in [0,1], sterzata alta = penalità

        # --- 4. Collisione o prossimità ostacoli ---
        lidar_min = np.min(lidar_sectors)
        if lidar_min < 0.1:
            collision_penalty = 0.0  # collisione o quasi: penalità massima
        else:
            collision_penalty = np.clip(lidar_min, 0.0, 1.0)  # più lontano = meglio

        # --- 5. Velocità equilibrata (non troppo lenta né troppo veloce) ---
        target_speed = 6.0  # m/s ottimale
        balanced_speed = 1.0 - abs(avg_speed - target_speed) / target_speed
        balanced_speed = np.clip(balanced_speed, 0.0, 1.0)

        # --- Pesi dei termini ---
        w_distance = 0.35
        w_acceleration = 0.2
        w_steering = 0.15
        w_safety = 0.2
        w_balance = 0.1

        total_reward = (
            w_distance * distance_reward +
            w_acceleration * acceleration_reward +
            w_steering * steering_penalty +
            w_safety * collision_penalty +
            w_balance * balanced_speed
        )

        # --- Scala in [0, 100] ---
        return float(np.clip(total_reward * 100.0, 0.0, 100.0))
        
    

    def is_done(self, obs):
        current_x, current_y, current_z = obs[2], obs[3], obs[4]
        distance_to_target = np.linalg.norm([
            self.target_x - current_x,
            self.target_y - current_y,
            self.target_z - current_z
        ])

        # Stop per distanza
        if distance_to_target < self.target_threshold:
            print("Traguardo raggiunto!")
            return True

        # Stop per tempo massimo
        if self.currentTimestep >= self.MAX_TIMESTEPS:
            print("Limite di tempo raggiunto!")
            return True

        # Stop se fuori area definita (es. fuori strada)
        if abs(current_y) > 5 or current_z < -0.5 or current_z > 1.5:
            print("Sei uscito dalla strada!")
            return True


        # collisione stimata se troppo vicino e fermo
        lidar_avg = np.mean(obs[12:22])
        v_left = obs[0]
        v_right = obs[1]

        if lidar_avg < 0.5:
            print("Collisione stimata tramite lidar.")
            return True

        # Stop per stallo (opzionale)
        if hasattr(self, "stall_counter") and self.stall_counter >= self.stall_limit:
            print("Episodio terminato per stallo prolungato.")
            return True

        return False

        
    def reset(self):

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.tesla_translation.setSFVec3f([0.0,-2.0,0.5])
        self.tesla_rotation.setSFRotation([0.0,0.0,1.0,0.0])
        
        print(f"------ Fine episodio! ----- ")
        self.car.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.car.resetPhysics()
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
                    env.tesla_node.simulationQuit(0)
                    break

except Exception as e:
    print(f"Errore nel server socket: {e}")
finally:
    print("Chiusura del controller Webots.")

