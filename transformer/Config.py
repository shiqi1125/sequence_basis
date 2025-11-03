import json

class config():
    def load_configuration(self, file_path: str) -> dict:
        try:
            with open(file_path, 'r') as file:
                config_data = json.load(file)
                file.close()
            return config_data
        except Exception as e:
            print(f"Error loading configuration from '{file_path}': {e}")
            raise

    def write_configuration(self, file_path: str, config: dict) -> None:
        try:
            with open(file_path, 'w') as file:
                json.dump(config, file, indent=4)
            print(f"Configuration successfully written to '{file_path}'.")
        except Exception as e:
            print(f"Error writing configuration to '{file_path}': {e}")
            raise

if __name__ == "__main__":
    config_file_path = "config.json"
    c = config()
    try:
        conf = c.load_configuration(config_file_path)
        print("Configuration loaded successfully:")
        print(conf)
    except Exception:
        conf = {
            "parameter1": "default_value1",
            "parameter2": 2,
            "parameter3": True
        }
        print("Initialized default configuration.")
    conf["parameter1"] = "updated_value"
    conf["parameter4"] = 3.14
    c.write_configuration(config_file_path, conf)