import yaml
import importlib.resources as pkg_resources

class InturaConfig:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InturaConfig, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        try:
            config_file = f"config.yaml"
            data = pkg_resources.read_text('intura_ai', config_file)
            self.data = yaml.safe_load(data)
        except FileNotFoundError:
            self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)