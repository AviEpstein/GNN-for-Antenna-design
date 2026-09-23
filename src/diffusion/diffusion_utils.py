def create_parameter_dict(config):

    env_dict = {
        'patch_x': config.get('patch_x', 28),
        'patch_y': config.get('patch_y', 28),
        'ground_x': config.get('ground_x', 50),
        'ground_y': config.get('ground_y', 50),
        'radius': config.get('radius', 75),
        'num_reflectors': config.get('num_reflectors', 2),
        'h': config.get('h', 4),
        'box_size': config.get('box_size', 50),
        'tan_d': config.get('tan_d', 0.0027),
        'eps_r': config.get('eps_r', 3.55),
        'reflector_distance': 3, 'reflector_scale': 1.5, 'threshold': 0.5,

    }
    reflectors_dict = None
    return env_dict, reflectors_dict
