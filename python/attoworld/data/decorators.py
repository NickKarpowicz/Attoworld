"""Class and function decorators."""

from dataclasses import is_dataclass

import numpy as np
import yaml


def add_method(cls, name: str):
    """Adds the decorated function as a member function of a class. The first argument of the function should be 'self'.

    Args:
        cls: The class type
        name: What the method will be called

    """

    def decorator(func):
        setattr(cls, name, func)
        return func

    return decorator


def yaml_io(cls):
    """Adds functions to save and load the dataclass as yaml."""

    def from_dict(cls, data: dict):
        """Takes a dict and makes an instance of the class.

        Args:
            cls: the class (hidden)
            data (dict): the result of a call of .to_dict on the class

        """

        def handle_complex_array(serialized_array) -> np.ndarray:
            """Helper function to deserialize numpy arrays, handling complex types."""
            if isinstance(serialized_array, list) and all(
                isinstance(item, dict) and "re" in item and "im" in item
                for item in serialized_array
            ):
                return np.array(
                    [complex(item["re"], item["im"]) for item in serialized_array],
                    dtype=np.complex128,
                )
            return np.array(serialized_array)

        loaded_data = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_type is np.ndarray:
                loaded_data[field_name] = handle_complex_array(data[field_name])
            elif is_dataclass(field_type):
                loaded_data[field_name] = field_type.from_dict(data[field_name])
            else:
                loaded_data[field_name] = data[field_name]
        return cls(**loaded_data)

    def load_yaml(cls, filename: str):
        """Load from a yaml file.

        Args:
            cls: the class (hidden)
            filename (str): path to the file

        """
        with open(filename, "r") as file:
            data = yaml.load(file, yaml.SafeLoader)
            return cls.from_dict(data)

    def load_yaml_bytestream(cls, stream):
        data = yaml.load(stream, yaml.SafeLoader)
        return cls.from_dict(data)

    def save_yaml(instance, filename: str):
        """Save to a yaml file.

        Args:
            instance: the class (hidden)
            filename (str): path to the file

        """
        data_dict = instance.to_dict()
        with open(filename, "w") as file:
            yaml.dump(data_dict, file)

    def to_dict(instance):
        """Serialize the class into a dict."""
        data_dict = {}
        for field_name, field_type in instance.__annotations__.items():
            field_value = getattr(instance, field_name)
            if field_type is np.ndarray:
                if field_value.dtype == np.complex128:
                    data_dict[field_name] = [
                        {"re": num.real, "im": num.imag} for num in field_value.tolist()
                    ]
                else:
                    data_dict[field_name] = field_value.tolist()
            elif is_dataclass(field_type):
                data_dict[field_name] = field_value.to_dict()
            elif field_type is np.float64 or field_type is float:
                data_dict[field_name] = float(field_value)
            else:
                data_dict[field_name] = field_value
        return data_dict

    cls.from_dict = classmethod(from_dict)
    cls.load_yaml = classmethod(load_yaml)
    cls.load_yaml_bytestream = classmethod(load_yaml_bytestream)
    cls.to_dict = to_dict
    cls.save_yaml = save_yaml
    return cls
