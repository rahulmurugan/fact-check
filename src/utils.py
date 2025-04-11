import logging
import sys

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up and returns a logger with the specified name and logging level.
    
    This logger prints to the standard output and formats each message with a
    timestamp, logger name, log level, and the message.
    
    Parameters:
        name (str): The name of the logger.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers to the logger.
    if not logger.handlers:
        # Create a stream handler that writes to sys.stdout.
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Define a formatter which outputs time, logger name, level, and message.
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add the handler to the logger.
        logger.addHandler(handler)
    
    return logger


def read_config(config_path: str) -> dict:
    """
    Reads a JSON configuration file from the given path and returns the configuration as a dictionary.
    
    Parameters:
        config_path (str): The path to the JSON configuration file.
    
    Returns:
        dict: Configuration parameters.
    """
    import json
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except Exception as e:
        raise Exception(f"Error reading configuration file {config_path}: {e}")


def save_json(data: dict, output_path: str):
    """
    Saves a dictionary as a JSON file to the specified output path.
    
    Parameters:
        data (dict): The data to save.
        output_path (str): The file path where the JSON data will be written.
    """
    import json
    try:
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except Exception as e:
        raise Exception(f"Error saving JSON to {output_path}: {e}")
