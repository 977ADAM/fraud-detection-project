

class Config:
    version = "1.0.0"
    dataset_id = "fraud:v1"
    name = "fraud"
    max_iter = 2000
    random_state = 42
    test_size = 0.3
    class_weight = 'balanced'
    solver = 'lbfgs'

config = Config()