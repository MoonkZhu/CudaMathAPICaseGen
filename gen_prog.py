#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import socket
import subprocess
from typing import List, Union

# --------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------

def sanitize_file_name(file_name: str) -> str:
    """Replace invalid characters in file names with underscores."""
    return re.sub(r'[<>:"/\\|?*]', '_', file_name)

def dir_name() -> str:
    """Generate a temporary directory name."""
    return '_tmp_' + socket.gethostname() + "_" + str(os.getpid())

def create_experiments_dir() -> str:
    """Create a temporary directory for experiments."""
    p = dir_name()
    print("Creating dir:", p)
    try:
        os.mkdir(p)
    except OSError as e:
        print(f"Error: Creation of the directory {p} failed: {e}")
        exit(1)
    return p


def getOperatorName(operator_name: str) -> str:
    """
    Return the string name of a given operator.

    :param operator_name: A string representing an operator (e.g., '+', '-', '*', '/').
    :return: The name of the operator as a string.
    """
    operator_map = {
        "+": "Addition",
        "-": "Subtraction",
        "*": "Multiplication",
        "/": "Division",
        "%": "Modulo",
        "**": "Exponentiation",
        "//": "Floor_Division",
        "==": "Equality",
        "!=": "Inequality",
        "<": "Less_Than",
        "<=": "Less_Than_or_Equal_To",
        ">": "Greater_Than",
        ">=": "Greater_Than_or_Equal_To",
        "and": "Logical_AND",
        "or": "Logical_OR",
        "not": "Logical_NOT",
        "&": "Bitwise_AND",
        "|": "Bitwise_OR",
        "^": "Bitwise_XOR",
        "~": "Bitwise_NOT",
        "<<": "Left_Shift",
        ">>": "Right_Shift",
        "+=": "Addition_Assignment",
        "-=": "Subtraction_Assignment",
        "*=": "Multiplication_Assignment",
        "/=": "Division_Assignment",
        "%=": "Modulo_Assignment",
        "//=": "Floor_Division_Assignment",
        "**=": "Exponentiation_Assignment",
        "&=": "Bitwise_AND_Assignment",
        "|=": "Bitwise_OR_Assignment",
        "^=": "Bitwise_XOR_Assignment",
        "<<=": "Left_Shift_Assignment",
        ">>=": "Right_Shift_Assignment",
    }

    return operator_map.get(operator_name, "Unknown Operator")

# --------------------------------------------------------------------------
# CUDA Code Generation
# --------------------------------------------------------------------------

def generate_CUDA_code(fun_name: str, params: List[str], return_type: str, directory: str) -> str:
    """Generate a CUDA source file for the given function."""
    sanitized_fun_name = sanitize_file_name(fun_name)
    i = 0
    file_name = f'{return_type}_cuda_code_{sanitized_fun_name}.cu'
    file_path = os.path.join(directory, file_name)

    try:
        with open(file_path, 'w') as fd:
            fd.write('// Automatically generated - do not modify\n\n')
            fd.write('#include <stdio.h>\n')
            fd.write('#include <cuda_fp16.h>\n\n')  # Required for half-precision
            fd.write('__global__ void kernel_1(\n')

            # Generate function signature
            signature = ""
            param_names = []
            for i, param in enumerate(params):
                signature += param + f' x{i}, '
                param_names.append(f'x{i}')

            # Write kernel function
            fd.write('  ' + signature[:-2] + ', ' + return_type + ' *ret) {\n')  # Remove trailing comma and space
            fd.write('   *ret = ' + fun_name + '(' + ', '.join(param_names) + ');\n')  # Corrected line
            fd.write('}\n\n')

            # Write wrapper function
            fd.write('extern "C" {\n')
            fd.write('  ' + return_type + ' kernel_wrapper_1(' + signature[:-2] + ') {\n')
            fd.write('  ' + return_type + ' *dev_p;\n')
            fd.write('  cudaMalloc(&dev_p, sizeof('+ return_type +'));\n')
            fd.write('  kernel_1<<<1,1>>>(' + ', '.join(param_names) + ', dev_p);\n')  # Corrected line
            fd.write('  ' + return_type + ' res;\n')
            fd.write('  cudaMemcpy(&res, dev_p, sizeof(' + return_type +'), cudaMemcpyDeviceToHost);\n')
            fd.write('  cudaFree(dev_p);\n')
            fd.write('  return res;\n')
            fd.write('}\n')
            fd.write('}\n\n')
        print(f"Generated CUDA code: {file_name}")
    except OSError as e:
        print(f"Error: Failed to write to file {file_path}: {e}")
        exit(1)

    return file_name


def half_operator_generate_CUDA_code(operator_name: str, params: List[str], return_type:str, directory: str) -> str:
    """Generate a CUDA source file for the given operator."""
    sanitized_fun_name = getOperatorName(operator_name)  # Assume this sanitizes operator names
    file_name = f'half_cuda_code_{sanitized_fun_name}.cu'
    file_path = os.path.join(directory, file_name)

    try:
        with open(file_path, 'w') as fd:
            fd.write('// Automatically generated - do not modify\n\n')
            fd.write('#include <stdio.h>\n')
            fd.write('#include <cuda_fp16.h>\n\n')  # Required for half-precision
            fd.write('__global__ void kernel_1(\n')

            # Generate function signature and parameter names
            signature = ""
            param_names = []
            for i, param in enumerate(params):
                print(params[i])
                signature += param[i] + f' x{i}, '
                param_names.append(f'x{i}')

            # Write kernel function signature
            fd.write('  ' + signature[:-2] + ', '+ return_type +' *ret) {\n')  # Remove trailing comma and space

            # Kernel function body
            if len(param_names) == 1:  # Unary operator
                fd.write(f'   *ret = {operator_name}{param_names[0]};\n')
            elif len(param_names) == 2:  # Binary operator
                fd.write(f'   *ret = {param_names[0]} {operator_name} {param_names[1]};\n')
            else:
                raise ValueError("Only unary and binary operators are supported.")
            fd.write('}\n\n')

            # Write wrapper function
            fd.write('extern "C" {\n')
            fd.write(return_type + ' half_kernel_wrapper_1(' + signature[:-2] + ') {\n')
            fd.write('  half *dev_p;\n')
            fd.write('  cudaMalloc(&dev_p, sizeof(half));\n')
            fd.write(f'  kernel_1<<<1,1>>>({", ".join(param_names)}, dev_p);\n')
            fd.write('  half res;\n')
            fd.write('  cudaMemcpy(&res, dev_p, sizeof(half), cudaMemcpyDeviceToHost);\n')
            fd.write('  cudaFree(dev_p);\n')  # Free allocated memory
            fd.write('  return res;\n')
            fd.write('}\n')
            fd.write('}\n\n')

        print(f"Generated CUDA code: {file_name}")
    except OSError as e:
        print(f"Error: Failed to write to file {file_path}: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    return file_name

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------

class FunctionSignature:
    """Represents a function signature with a return type, name, and input types."""
    def __init__(self, return_type: str, fun_name: str, input_types: List[str]):
        self.return_type = return_type
        self.fun_name = fun_name
        self.input_types = input_types

    def __repr__(self):
        return f"FunctionSignature(return_type={self.return_type}, fun_name={self.fun_name}, input_types={self.input_types})"

class OperatorType:
    """Represents an operator type with a return type, name, and input types."""
    def __init__(self, return_type: str, operator_name: str, input_types: List[str]):
        self.return_type = return_type
        self.operator_name = operator_name
        self.input_types = input_types

    def __repr__(self):
        return f"OperatorType(return_type={self.return_type}, operator_name={self.operator_name}, input_types={self.input_types})"

# --------------------------------------------------------------------------
# Parsing Functions
# --------------------------------------------------------------------------

def parse_functions_to_test(file_name: str) -> List[Union[FunctionSignature, OperatorType]]:
    """
    Parse the input file to extract function signatures and operator types.

    Args:
        file_name (str): Path to the input file.

    Returns:
        List[Union[FunctionSignature, OperatorType]]: A list of parsed objects.
    """
    results = []
    try:
        with open(file_name, 'r') as fd:
            for line in fd:
                # Skip comments and empty lines
                if line.lstrip().startswith('#') or not line.strip():
                    continue

                if line.lstrip().startswith('FUNCTION:'):
                    line = line.strip()
                    try:
                        signature = line.split('FUNCTION:')[1]

                        # Parse the return type and function name
                        parts = signature.split(' ', 1)
                        if len(parts) != 2:
                            raise ValueError(f"Invalid FUNCTION format: {line}")
                        return_type = parts[0].strip()
                        rest = parts[1]
                        fun_name, params_raw = rest.split('(', 1)
                        params = params_raw.split(')')[0].split(',')
                        results.append(FunctionSignature(return_type.strip(), fun_name.strip(), [p.strip() for p in params]))
                        #print(params)
                    except Exception as e:
                        raise ValueError(f"Failed to parse FUNCTION: {line} - {e}")

                elif line.lstrip().startswith('OPERATOR:'):
                    line = line.strip()
                    try:
                        signature = line.split('OPERATOR:')[1]

                        # Parse the return type and operator name
                        parts = signature.split(' ', 1)
                        if len(parts) != 2:
                            raise ValueError(f"Invalid OPERATOR format: {line}")
                        return_type = parts[0].strip()
                        rest = parts[1]
                        operator_name, params_raw = rest.split('(', 1)
                        params = params_raw.split(')')[0].split(',')
                        results.append(OperatorType(return_type.strip(), operator_name.strip(), [p.strip() for p in params]))
                    except Exception as e:
                        raise ValueError(f"Failed to parse OPERATOR: {line} - {e}")
    except FileNotFoundError as e:
        print(f"Error: File {file_name} not found: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: Failed to parse {file_name}: {e}")
        exit(1)

    return results

# --------------------------------------------------------------------------
# Main Script
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Numerical Bug Detection tool')
    parser.add_argument('function', metavar='FUNCTION_TO_TEST', nargs=1, help='Function to test (file or shared library .so)')
    parser.add_argument('-c', '--clean', action='store_true', help='Remove temporal directories (begin with _tmp_)')
    args = parser.parse_args()

    # Cleaning mode
    if args.clean:
        print('Removing temporal directories...')
        for fname in os.listdir('.'):
            if fname.startswith("_tmp_"):
                shutil.rmtree(fname, ignore_errors=True)
        exit()

    input_file = args.function[0]
    functions_to_test = parse_functions_to_test(input_file)

    # Create directory for experiments
    experiments_dir = create_experiments_dir()

    # Generate CUDA code for each function
    for item in functions_to_test:
        if isinstance(item, FunctionSignature):
            generate_CUDA_code(item.fun_name, item.input_types ,item.return_type ,experiments_dir)
        # elif isinstance(item, OperatorType):
            # half_operator_generate_CUDA_code(item.operator_name, item.return_type, item.input_types, experiments_dir)
