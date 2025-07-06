"""
Genetics structures and types - corresponds to genetics.h
"""

from .base import Structure, POINTER, c_uint64, c_double, List, Optional, pd, ctypes, liblgp
from .vm import ProgramWrapper, MemblockWrapper, OperationWrapper

class IndividualWrapper(Structure):
    """Corrisponde a struct Individual in genetics.h"""
    _fields_ = [
        ("prog", ProgramWrapper),
        ("fitness", c_double)
    ]


class PopulationWrapper(Structure):
    """Corrisponde a struct Population in genetics.h"""
    _fields_ = [
        ("individual", POINTER(IndividualWrapper)),
        ("size", c_uint64)
    ]


class InstructionSetWrapper(Structure):
    """Corrisponde a struct InstructionSet in genetics.h (wrapper C)"""
    _fields_ = [
        ("size", c_uint64),
        ("op", POINTER(OperationWrapper))
    ]


class InstructionSet:
    """InstructionSet che prende una lista di Operation nel costruttore"""
    
    def __init__(self, operations):
        """
        Crea un InstructionSet da una lista di Operation
        
        Args:
            operations: Lista di Operation
        """
        # Converti in lista se necessario
        if not isinstance(operations, (list, tuple)):
            operations = list(operations)
        
        # Crea array di OperationWrapper
        op_array = (OperationWrapper * len(operations))()
        
        # Riempi l'array con gli OperationWrapper delle Operation
        for i, op in enumerate(operations):
            op_array[i] = op.c_wrapper
        
        # Crea InstructionSetWrapper
        self._wrapper = InstructionSetWrapper(
            size=len(operations),
            op=ctypes.cast(op_array, POINTER(OperationWrapper))
        )
    
    @property
    def size(self) -> int:
        """Numero di operazioni nell'instruction set"""
        return self._wrapper.size
    
    @property
    def c_wrapper(self) -> InstructionSetWrapper:
        """Restituisce la struttura C sottostante"""
        return self._wrapper


class LGPInputWrapper(Structure):
    """Corrisponde a struct LGPInput in genetics.h"""
    _fields_ = [
        ("input_num", c_uint64),
        ("rom_size", c_uint64),
        ("res_size", c_uint64),
        ("ram_size", c_uint64),
        ("instr_set", InstructionSetWrapper),
        ("memory", POINTER(MemblockWrapper))
    ]


class LGPResultWrapper(Structure):
    """Corrisponde a struct LGPResult in genetics.h"""
    _fields_ = [
        ("pop", PopulationWrapper),
        ("evaluations", c_uint64),
        ("generations", c_uint64),
        ("best_individ", c_uint64)
    ]


# High-level wrapper classes (previously in wrapper.py)

class LGPInput:
    """Wrapper per struct LGPInput che mantiene la nomenclatura C"""
    
    def __init__(self, c_lgp_input: LGPInputWrapper):
        self._c_lgp_input = c_lgp_input
    
    @property
    def input_num(self) -> int:
        """Numero di input"""
        return self._c_lgp_input.input_num
    
    @property
    def rom_size(self) -> int:
        """Dimensione ROM"""
        return self._c_lgp_input.rom_size
    
    @property
    def res_size(self) -> int:
        """Dimensione risultato"""
        return self._c_lgp_input.res_size
    
    @property
    def ram_size(self) -> int:
        """Dimensione RAM"""
        return self._c_lgp_input.ram_size
    
    @property
    def c_wrapper(self) -> LGPInputWrapper:
        """Restituisce la struttura C sottostante"""
        return self._c_lgp_input
    
    @classmethod
    def from_df(cls, df, y: List[str], instruction_set: InstructionSet, ram_size: Optional[int] = None):
        """
        Crea un'istanza LGPInput da un pandas DataFrame
        
        Args:
            df: DataFrame contenente i dati
            y: Lista delle colonne da considerare come target (y)
            instruction_set: Set di istruzioni disponibili
            ram_size: Dimensione RAM (default: res_size)
        """
        if pd is None:
            raise ImportError("pandas is required for from_df method. Install with: pip install pandas")
        
        # Validazione parametri
        if len(y) == 0:
            raise ValueError("y list cannot be empty")
        
        if len(df) == 0:
            raise ValueError("DataFrame cannot be empty")
        
        # Calcola dimensioni
        input_num = len(df)
        rom_size = len(df.columns) - len(y)
        res_size = len(y)
        
        if rom_size <= 0:
            raise ValueError("rom_size must be positive (insufficient input columns)")
        
        if ram_size is None:
            ram_size = res_size
        
        if ram_size < res_size:
            raise ValueError("ram_size cannot be less than res_size")
        
        # Calcola block_size: rom + ram
        block_size = rom_size + ram_size
        
        # Alloca memoria per tutti i blocchi
        total_memory_size = input_num * block_size
        memory = (MemblockWrapper * total_memory_size)()
        
        # Separa feature (X) e target (y) colonne
        feature_cols = [col for col in df.columns if col not in y]
        target_cols = y
        
        # Riempi la memoria seguendo il pattern di psb2.c
        for i in range(input_num):
            base_idx = i * block_size
            
            # Riempi ROM con le feature (X)
            for j, col in enumerate(feature_cols):
                memory[base_idx + j].f64 = float(df.iloc[i][col])
            
            # Riempi la parte iniziale di RAM con i target (y)
            for j, col in enumerate(target_cols):
                memory[base_idx + rom_size + j].f64 = float(df.iloc[i][col])
            
            # Inizializza il resto di RAM a 0.0 se ram_size > res_size
            for j in range(res_size, ram_size):
                memory[base_idx + rom_size + j].f64 = 0.0
        
        # Crea la struttura LGPInputWrapper
        lgp_input_wrapper = LGPInputWrapper(
            input_num=input_num,
            rom_size=rom_size,
            res_size=res_size,
            ram_size=ram_size,
            instr_set=instruction_set.c_wrapper,
            memory=ctypes.cast(memory, POINTER(MemblockWrapper))
        )
        
        return cls(lgp_input_wrapper)
    
    @classmethod
    def from_numpy(cls, X, y, instruction_set: InstructionSet, ram_size: Optional[int] = None):
        """
        Crea un'istanza LGPInput da array numpy/liste
        
        Args:
            X: Array/matrice delle feature di input (n_samples, n_features)
            y: Array/lista dei target - può essere 1D o lista di valori
            instruction_set: Set di istruzioni disponibili
            ram_size: Dimensione RAM (default: len(y) se y è 1D, altrimenti numero di colonne target)
        """
        import numpy as np
        
        # Converti X in array numpy se non lo è già
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Converti y in array numpy e gestisci il caso 1D o lista
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Validazione dimensioni
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X e y devono avere lo stesso numero di campioni: {X.shape[0]} vs {y.shape[0]}")
        
        # Calcola dimensioni
        input_num = X.shape[0]
        rom_size = X.shape[1]
        res_size = y.shape[1]
        
        if rom_size <= 0:
            raise ValueError("rom_size deve essere positivo (numero di feature di input)")
        
        if ram_size is None:
            ram_size = res_size
        
        if ram_size < res_size:
            raise ValueError("ram_size non può essere minore di res_size")
        
        # Calcola block_size: rom + ram
        block_size = rom_size + ram_size
        
        # Alloca memoria per tutti i blocchi
        total_memory_size = input_num * block_size
        memory = (MemblockWrapper * total_memory_size)()
        
        # Riempi la memoria
        for i in range(input_num):
            base_idx = i * block_size
            
            # Riempi ROM con le feature (X)
            for j in range(rom_size):
                memory[base_idx + j].f64 = float(X[i, j])
            
            # Riempi la parte iniziale di RAM con i target (y)
            for j in range(res_size):
                memory[base_idx + rom_size + j].f64 = float(y[i, j])
            
            # Inizializza il resto di RAM a 0.0 se ram_size > res_size
            for j in range(res_size, ram_size):
                memory[base_idx + rom_size + j].f64 = 0.0
        
        # Crea la struttura LGPInputWrapper
        lgp_input_wrapper = LGPInputWrapper(
            input_num=input_num,
            rom_size=rom_size,
            res_size=res_size,
            ram_size=ram_size,
            instr_set=instruction_set.c_wrapper,
            memory=ctypes.cast(memory, POINTER(MemblockWrapper))
        )
        
        return cls(lgp_input_wrapper)


class Individual:
    """Wrapper per struct Individual che mantiene la nomenclatura C"""
    
    def __init__(self, c_individual: IndividualWrapper):
        self._c_individual = c_individual
    
    @property
    def prog(self) -> ProgramWrapper:
        """Accesso al programma"""
        return self._c_individual.prog
    
    @property
    def fitness(self) -> float:
        """Accesso al fitness"""
        return self._c_individual.fitness
    
    def print_program(self) -> None:
        """Stampa il programma di questo individuo"""
        from .utils import print_program
        print_program(self)


class Population:
    """Wrapper per struct Population che mantiene la nomenclatura C"""
    
    def __init__(self, c_population: PopulationWrapper):
        self._c_population = c_population
    
    @property
    def size(self) -> int:
        """Dimensione della popolazione"""
        return self._c_population.size
    
    @property
    def c_wrapper(self) -> PopulationWrapper:
        """Restituisce la struttura C sottostante"""
        return self._c_population
    
    def get(self, index: int) -> Individual:
        """
        Ottiene un individuo dalla popolazione per indice
        
        Args:
            index: Indice dell'individuo (0-based)
            
        Returns:
            Individual dell'individuo richiesto
        """
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range for population of size {self.size}")
        
        return Individual(self._c_population.individual[index])


class VectorDistance(LGPInput):
    """Sottoclasse di LGPInput per problemi di distanza vettoriale"""
    
    def __init__(self, instruction_set: InstructionSet, vector_len: int, instances: int):
        """
        Inizializza un problema di distanza vettoriale
        
        Args:
            instruction_set: Set di istruzioni disponibili
            vector_len: Lunghezza dei vettori
            instances: Numero di istanze del problema
        """
        liblgp.vector_distance.argtypes = [POINTER(InstructionSetWrapper), c_uint64, c_uint64]
        liblgp.vector_distance.restype = LGPInputWrapper
        
        result = liblgp.vector_distance(
            ctypes.byref(instruction_set.c_wrapper),
            c_uint64(vector_len),
            c_uint64(instances)
        )
        
        # Inizializza il wrapper con il risultato
        super().__init__(result)

__all__ = ['IndividualWrapper', 'PopulationWrapper', 'InstructionSetWrapper', 'InstructionSet', 'LGPInputWrapper', 'LGPResultWrapper', 'LGPInput', 'Individual', 'Population', 'VectorDistance']
