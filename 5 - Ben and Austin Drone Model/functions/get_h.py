from drone_model_with_PWM import DroneModel
import numpy as np
import drone_config
FLAG = drone_config.FLAG

drone_model = DroneModel()

def outer(func):
    def wrapper(*args, **kwargs):
        global dt
        args = list(args) + [dt]
        value = func(*args)
        return value
    return wrapper
if FLAG == '1':
    f_c = outer(drone_model.f_c_car)
elif FLAG == '2':
    f_c = outer(drone_model.f_c_car2)
elif FLAG == '3':
    f_c = outer(drone_model.f_c_car3)
elif FLAG == '4':
    f_c = outer(drone_model.f_c_car4)

ALL_SENSORS = ['IMU',
               'IMU + OPTIC_FLOW',
               'IMU + WIND',
               'IMU + WIND + OPTIC_FLOW']
SENSORS = ['IMU', 'IMU + OPTIC_FLOW', 'IMU + WIND', 'IMU + WIND + OPTIC_FLOW']

def outer_h(func, input_sensor=None):
    if input_sensor is None:
        # If no explicit sensor is passed, use whatever Y_SWEEP currently is
        def wrapper(*args, **kwargs):
            global Y_SWEEP
            # Convert args → list, append Y_SWEEP, then call func
            full_args = list(args) + [Y_SWEEP]
            return func(*full_args)
    else:
        # If input_sensor is provided in the decorator, always use that
        def wrapper(*args, **kwargs):
            full_args = list(args) + [input_sensor]
            return func(*full_args)
    return wrapper

def get_h_c(Y_SWEEP):
    if Y_SWEEP == 'ALL':
        # We want h_c to be a dict:  { sensor_str: wrapped_function, … }
        h_c = {}
        h_c_ukf = {}
        for sensor_name in ALL_SENSORS:
            print(f'getting measurement dynamics for sensors: {sensor_name}')
            # Decorate h_c_car so that it always receives sensor_name as its last argument
            if FLAG == '1':
                h_c[sensor_name] = outer_h(drone_model.h_c_car, input_sensor=sensor_name)
            elif FLAG == '2':
                h_c[sensor_name] = outer_h(drone_model.h_c_car2, input_sensor=sensor_name)
            elif FLAG == '3':
                h_c[sensor_name] = outer_h(drone_model.h_c_car3, input_sensor=sensor_name)
            elif FLAG == '4':
                h_c[sensor_name] = outer_h(drone_model.h_c_car4, input_sensor=sensor_name)

    else:
        # Only a single sensor is requested—wrap h_c_car so it uses whatever Y_SWEEP is
        print(f'getting measurement dynamics for sensors: {Y_SWEEP}')
        if FLAG == '1':
            h_c = outer_h(drone_model.h_c_car, input_sensor=Y_SWEEP)
        elif FLAG == '2':
            h_c = outer_h(drone_model.h_c_car2, input_sensor=Y_SWEEP)
        elif FLAG == '3':
            h_c = outer_h(drone_model.h_c_car3, input_sensor=Y_SWEEP)
        elif FLAG == '4':
            h_c = outer_h(drone_model.h_c_car4, input_sensor=Y_SWEEP)
    return h_c

def cluster_data(y_sim_noise, Y_SWEEP):
    if FLAG == '1':
        P_cluster = np.array(np.vstack([y_sim_noise['x'], y_sim_noise['y'], y_sim_noise['z']]))
        V_cluster = np.array(np.vstack([y_sim_noise['v_x'], y_sim_noise['v_y'], y_sim_noise['v_z']]))
        Attitude_cluster = np.array(np.vstack([y_sim_noise['phi'], y_sim_noise['theta'], y_sim_noise['psi']]))
        Omega_cluster = np.array(np.vstack([y_sim_noise['omega_x'], y_sim_noise['omega_y'], y_sim_noise['omega_z']]))
        Acc_cluster = np.array(np.vstack([y_sim_noise['Axt'], y_sim_noise['Ayt'], y_sim_noise['Az']]))
        IMU_cluster = np.array(np.vstack([y_sim_noise['phi'], y_sim_noise['theta'], y_sim_noise['psi'], y_sim_noise['omega_x'], y_sim_noise['omega_y'], y_sim_noise['omega_z'], y_sim_noise['Axt'], y_sim_noise['Ayt'], y_sim_noise['Az']]))
        Wind_cluster = np.array(np.vstack([y_sim_noise['Wax'], y_sim_noise['Way']]))
        OF_cluster = np.array(np.vstack([y_sim_noise['rx'], y_sim_noise['ry']]))
    elif FLAG == '2' or FLAG == '3':
        P_cluster = np.array(np.vstack([y_sim_noise['x'], y_sim_noise['y'], y_sim_noise['z']]))
        V_cluster = np.array(np.vstack([y_sim_noise['v_x'], y_sim_noise['v_y'], y_sim_noise['v_z']]))
        Attitude_cluster = np.array(np.vstack([y_sim_noise['psi']]))
        Acc_cluster = np.array(np.vstack([y_sim_noise['Ax'], y_sim_noise['Ay'], y_sim_noise['Az']]))
        IMU_cluster = np.array(np.vstack([y_sim_noise['psi'], y_sim_noise['Ax'], y_sim_noise['Ay'], y_sim_noise['Az']]))
        Wind_cluster = np.array(np.vstack([y_sim_noise['Wax'], y_sim_noise['Way']]))
        OF_cluster = np.array(np.vstack([y_sim_noise['rx'], y_sim_noise['ry']]))
    elif FLAG == '4':
        P_cluster = np.array(np.vstack([y_sim_noise['x'], y_sim_noise['y'], y_sim_noise['z']]))
        V_cluster = np.array(np.vstack([y_sim_noise['v_x'], y_sim_noise['v_y'], y_sim_noise['v_z']]))
        Attitude_cluster = np.array(np.vstack([y_sim_noise['phi'], y_sim_noise['theta'], y_sim_noise['psi']]))
        Acc_cluster = np.array(np.vstack([y_sim_noise['Ax'], y_sim_noise['Ay'], y_sim_noise['Az']]))
        IMU_cluster = np.array(np.vstack([y_sim_noise['phi'], y_sim_noise['theta'], y_sim_noise['psi'], y_sim_noise['Ax'], y_sim_noise['Ay'], y_sim_noise['Az']]))
        Wind_cluster = np.array(np.vstack([y_sim_noise['Wax'], y_sim_noise['Way']]))
        OF_cluster = np.array(np.vstack([y_sim_noise['rx'], y_sim_noise['ry']]))


    if Y_SWEEP == 'ALL':
        print('Running all sensors...')
        Y_noise = {}
        for sensor in SENSORS:
            if sensor == 'IMU':
                Y_noise[sensor] = np.vstack((IMU_cluster))
            elif sensor == 'IMU + OPTIC_FLOW':
                Y_noise[sensor] = np.vstack((IMU_cluster, OF_cluster))
            elif sensor == 'IMU + WIND':
                Y_noise[sensor] = np.vstack((IMU_cluster, Wind_cluster))
            elif sensor == 'IMU + WIND + OPTIC_FLOW':
                Y_noise[sensor] = np.vstack((IMU_cluster, OF_cluster, Wind_cluster))
            elif sensor == 'IMU + VEL + WIND':
                Y_noise[sensor] = np.vstack((IMU_cluster, V_cluster, Wind_cluster))
            else:
                raise ValueError('Invalid sensor type')
        print(f'Running sensors: {list(Y_noise.keys())}')
    else:
        print(f'Running sensors: {Y_SWEEP}')
        if Y_SWEEP == 'IMU':
            Y_noise = np.vstack((IMU_cluster))
        elif Y_SWEEP == 'IMU + OPTIC_FLOW':
            Y_noise = np.vstack((IMU_cluster, OF_cluster))
        elif Y_SWEEP == 'IMU + WIND':
            Y_noise = np.vstack((IMU_cluster, Wind_cluster))
        elif Y_SWEEP == 'IMU + WIND + OPTIC_FLOW':
            Y_noise = np.vstack((IMU_cluster, OF_cluster, Wind_cluster))
        elif Y_SWEEP == 'IMU + VEL + WIND':
            Y_noise = np.vstack((IMU_cluster, V_cluster, Wind_cluster))
        else:
            raise ValueError('Invalid Y_SWEEP value')
    
    return Y_noise