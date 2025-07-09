"""
Genetics structures and types - corresponds to genetics.h
Unified wrapper classes that combine C structures with user-friendly interfaces
"""

from .base import Structure, POINTER, c_uint64, c_double, List, Optional, ctypes, liblgp, VECT_ALIGNMENT
from .vm import Program, Memblock, OperationStruct, Instruction

class Program(Structure):
    """Standard Program structure without special alignment"""
    _fields_ = [
        ("content", POINTER(Instruction)),
        ("size", c_uint64)
    ]


class Individual(Structure):
    """Corresponds to struct Individual in genetics.h with integrated user interface"""
    _fields_ = [
        ("prog", Program),
        ("fitness", c_double),
    ]
    
    def print_program(self) -> None:
        """Print the program of this individual"""
        from .utils import print_program
        print_program(self)
    
    @property
    def size(self) -> int:
        """Return the program size (number of instructions)"""
        return self.prog.size
    


class Population(Structure):
    """Corresponds to struct Population in genetics.h with integrated user interface"""
    _fields_ = [
        ("individual", POINTER(Individual)),
        ("size", c_uint64)
    ]
    
    def __del__(self):
        """Free the population memory when the object is deleted"""
        liblgp.free_population(ctypes.byref(self))
    
    def get(self, index: int) -> Individual:
        """
        Get an individual from the population by index
        
        Args:
    # get_INSTRSET function
    liblgp.get_INSTRSET.argtypes = [c_uint32]
    liblgp.get_INSTRSET.restype = POINTER(OperationStruct)
            index: Individual index (0-based)
            
        Returns:
            Individual at the requested index
        """
        if not self.individual:
            raise RuntimeError("Population not properly initialized")
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range for population of size {self.size}")
        return self.individual[index]
        
class InstructionSet(Structure):
    """Corresponds to struct InstructionSet in genetics.h with integrated user interface"""
    _fields_ = [
        ("size", c_uint64),
        ("op", POINTER(OperationStruct))
    ]
    
    def __init__(self, operations: List[Instruction]):
        """
        Create an InstructionSet from a list of Operations
        
        Args:
            operations: List of Operations (optional for ctypes compatibility)
        """
        if operations is None or len(operations) == 0:
            raise ValueError("Instruction set cannot be empty")
        
        # Create array of OperationStruct
        op_array = (OperationStruct * len(operations))()
        
        # Fill array with OperationStruct from Operations
        for i, op in enumerate(operations):
            # op.value è l'OperationStruct, non op.c_wrapper
            op_array[i] = op.value
        
        # Initialize structure fields
        super().__init__(
            size=len(operations),
            op=ctypes.cast(op_array, POINTER(OperationStruct))
        )
    
    @staticmethod
    def complete():
        return InstructionSet.in_dll(liblgp, "INSTRSET")


class LGPInput(Structure):
    """Corresponds to struct LGPInput in genetics.h with integrated user interface"""
    _fields_ = [
        ("input_num", c_uint64),
        ("rom_size", c_uint64),
        ("res_size", c_uint64),
        ("ram_size", c_uint64),
        ("instr_set", InstructionSet),
        ("memory", POINTER(Memblock))
    ]

    def __init__(self, c_allocated=False):
        super().__init__()
        self._c_allocated = c_allocated

    def __del__(self):
        if self._c_allocated:
            liblgp.free_lgp_input(ctypes.byref(self))


    @classmethod
    def from_df(cls, df, y: List[str], instruction_set: InstructionSet, ram_size: Optional[int] = None):
        """
        Create an LGPInput instance from a pandas DataFrame
        
        Args:
            df: DataFrame containing the data
            y: List of columns to consider as targets (y)
            instruction_set: Available instruction set
            ram_size: RAM size (default: res_size)
        """
        import pandas as pd
        # Parameter validation
        if len(y) == 0:
            raise ValueError("y list cannot be empty")
        
        if len(df) == 0:
            raise ValueError("DataFrame cannot be empty")
        
        # Calculate dimensions
        input_num = len(df)
        rom_size_val = len(df.columns) - len(y)
        res_size = len(y)
        
        if rom_size_val <= 0:
            raise ValueError("rom_size must be positive (insufficient input columns)")
        
        if ram_size is None:
            ram_size = res_size
        
        if ram_size < res_size:
            raise ValueError("ram_size cannot be less than res_size")
        
        # Calculate block_size: rom + ram
        block_size = rom_size_val + ram_size
        
        # Allocate memory for all blocks
        total_memory_size = input_num * block_size
        memory = (Memblock * total_memory_size)()
        
        # Separate feature (X) and target (y) columns
        feature_cols = [col for col in df.columns if col not in y]
        target_cols = y
        
        # Fill memory following psb2.c pattern
        for i in range(input_num):
            base_idx = i * block_size
            
            # Fill ROM with features (X)
            for j, col in enumerate(feature_cols):
                memory[base_idx + j].f64 = float(df.iloc[i][col])
            
            # Fill initial RAM part with targets (y)
            for j, col in enumerate(target_cols):
                memory[base_idx + rom_size_val + j].f64 = float(df.iloc[i][col])
            
            # Initialize rest of RAM to 0.0 if ram_size > res_size
            for j in range(res_size, ram_size):
                memory[base_idx + rom_size_val + j].f64 = 0.0
        
        # Create and return instance
        instance = cls()
        instance.input_num = input_num
        instance.rom_size = rom_size_val
        instance.res_size = res_size
        instance.ram_size = ram_size
        instance.instr_set = instruction_set
        instance.memory = ctypes.cast(memory, POINTER(Memblock))
        
        return instance
    
    @classmethod
    def from_numpy(cls, X, y, instruction_set: InstructionSet, ram_size: Optional[int] = None):
        """
        Create an LGPInput instance from numpy arrays/lists
        
        Args:
            X: Array/matrix of input features (n_samples, n_features)
            y: Array/list of targets - can be 1D or list of values
            instruction_set: Available instruction set
            ram_size: RAM size (default: max(1, y.shape[1]) if y is 2D, otherwise 1)
        """
        import numpy as np
        # Convert X to numpy array if not already
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Convert y to numpy array and handle 1D or list case
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Dimension validation
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples: {X.shape[0]} vs {y.shape[0]}")
        
        # Calculate dimensions
        input_num = X.shape[0]
        rom_size_val = X.shape[1]
        res_size = y.shape[1]
        
        if rom_size_val <= 0:
            raise ValueError("rom_size must be positive (number of input features)")
        
        if ram_size is None:
            ram_size = max(1, res_size)
        
        if ram_size < res_size:
            raise ValueError("ram_size cannot be less than res_size")
        
        # Calculate block_size: rom + ram (following C code pattern)
        block_size = rom_size_val + ram_size
        
        # Allocate memory for all blocks - ensure proper alignment
        total_memory_size = input_num * block_size
        memory = (Memblock * total_memory_size)()
        
        # Fill memory following the pattern from psb2.c
        for i in range(input_num):
            base_idx = i * block_size
            
            # Fill ROM with features (X) - this corresponds to the problem data
            for j in range(rom_size_val):
                memory[base_idx + j].f64 = float(X[i, j])
            
            # Fill RAM with targets (y) at the beginning of RAM section
            # This follows the pattern where solutions are stored at rom_size offset
            for j in range(res_size):
                memory[base_idx + rom_size_val + j].f64 = float(y[i, j])
            
            # Initialize rest of RAM to 0.0 if ram_size > res_size
            for j in range(res_size, ram_size):
                memory[base_idx + rom_size_val + j].f64 = 0.0
        
        # Create and return instance - all fields should match C struct exactly
        instance = cls()
        instance.input_num = input_num
        instance.rom_size = rom_size_val
        instance.res_size = res_size
        instance.ram_size = ram_size
        instance.instr_set = instruction_set
        instance.memory = ctypes.cast(memory, POINTER(Memblock))
        
        # Store a reference to the memory array to prevent garbage collection
        instance._memory_array = memory
        
        return instance


class LGPResult(Structure):
    """Corresponds to struct LGPResult in genetics.h with integrated user interface"""
    _fields_ = [
        ("pop", Population),
        ("evaluations", c_uint64),
        ("generations", c_uint64),
        ("best_individ", c_uint64)
    ]
    


class VectorDistance(LGPInput):
    """Subclass of LGPInput for vector distance problems"""
    
    def __init__(self, instruction_set: InstructionSet, vector_len: int, instances: int):
        """
        Initialize a vector distance problem
        
        Args:
            instruction_set: Available instruction set
            vector_len: Vector length
            instances: Number of problem instances
        """
        # Initialize parent class with c_allocated=True
        super().__init__(c_allocated=True)
        
        result = liblgp.vector_distance(
            ctypes.byref(instruction_set),
            c_uint64(vector_len),
            c_uint64(instances)
        )
        
        # Copy fields from result
        self.input_num = result.input_num
        self.rom_size = result.rom_size
        self.res_size = result.res_size
        self.ram_size = result.ram_size
        self.instr_set = result.instr_set
        self.memory = result.memory


class BouncingBalls(LGPInput):
    """Subclass of LGPInput for bouncing balls problems"""
    
    def __init__(self, instruction_set: InstructionSet, instances: int):
        """
        Initialize a bouncing balls problem
        
        Args:
            instruction_set: Available instruction set
            instances: Number of problem instances
        """
        # Initialize parent class with c_allocated=True
        super().__init__(c_allocated=True)
        
        result = liblgp.bouncing_balls(
            ctypes.byref(instruction_set),
            c_uint64(instances)
        )
        
        # Copy fields from result
        self.input_num = result.input_num
        self.rom_size = result.rom_size
        self.res_size = result.res_size
        self.ram_size = result.ram_size
        self.instr_set = result.instr_set
        self.memory = result.memory


class DiceGame(LGPInput):
    """Subclass of LGPInput for dice game problems"""
    
    def __init__(self, instruction_set: InstructionSet, instances: int):
        """
        Initialize a dice game problem
        
        Args:
            instruction_set: Available instruction set
            instances: Number of problem instances
        """
        # Initialize parent class with c_allocated=True
        super().__init__(c_allocated=True)
        
        result = liblgp.dice_game(
            ctypes.byref(instruction_set),
            c_uint64(instances)
        )
        
        # Copy fields from result
        self.input_num = result.input_num
        self.rom_size = result.rom_size
        self.res_size = result.res_size
        self.ram_size = result.ram_size
        self.instr_set = result.instr_set
        self.memory = result.memory


class ShoppingList(LGPInput):
    """Subclass of LGPInput for shopping list problems"""
    
    def __init__(self, instruction_set: InstructionSet, num_of_items: int, instances: int):
        """
        Initialize a shopping list problem
        
        Args:
            instruction_set: Available instruction set
            num_of_items: Number of items in the shopping list
            instances: Number of problem instances
        """
        # Initialize parent class with c_allocated=True
        super().__init__(c_allocated=True)
        
        result = liblgp.shopping_list(
            ctypes.byref(instruction_set),
            c_uint64(num_of_items),
            c_uint64(instances)
        )
        
        # Copy fields from result
        self.input_num = result.input_num
        self.rom_size = result.rom_size
        self.res_size = result.res_size
        self.ram_size = result.ram_size
        self.instr_set = result.instr_set
        self.memory = result.memory


class SnowDay(LGPInput):
    """Subclass of LGPInput for snow day problems"""
    
    def __init__(self, instruction_set: InstructionSet, instances: int):
        """
        Initialize a snow day problem
        
        Args:
            instruction_set: Available instruction set
            instances: Number of problem instances
        """
        # Initialize parent class with c_allocated=True
        super().__init__(c_allocated=True)
        
        result = liblgp.snow_day(
            ctypes.byref(instruction_set),
            c_uint64(instances)
        )
        
        # Copy fields from result
        self.input_num = result.input_num
        self.rom_size = result.rom_size
        self.res_size = result.res_size
        self.ram_size = result.ram_size
        self.instr_set = result.instr_set
        self.memory = result.memory

__all__ = ['Individual', 'Population', 'InstructionSet', 'LGPInput', 'LGPResult', 'VectorDistance', 'BouncingBalls', 'DiceGame', 'ShoppingList', 'SnowDay']
