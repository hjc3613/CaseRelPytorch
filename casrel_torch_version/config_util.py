import yaml

class Conf:
    def __init__(self, conf_path='config.yaml'):
        with open(conf_path, encoding='utf8') as f:
            self.conf_data = yaml.safe_load(f)