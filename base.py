from abc import ABC
from utils import timer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import traceback
from joblib import Parallel, delayed
from loguru import logger
from typing import Generic, Iterable, TypeVar

T = TypeVar('T', list, Iterable)
M = TypeVar('M', dict, None)

class Batch(ABC, Generic[T]):  
    content:T
    metadata:dict

    def __init__(self, content:T, metadata:M):
        self.content = content
        self.metadata = metadata

    def __len__(self):
        return len(self.content)

class Reader(ABC):
    @timer()
    def scheduler(self) -> list:
        raise NotImplementedError

    @timer()
    def read(self) -> Batch:
        raise NotImplementedError

class Step(ABC):
    name:str
    threads:int

    @timer()
    def step(self, data):
        raise NotImplementedError
    
    @timer()
    def _step(self, data):
        try:
            return self.step(data)
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            logger.error(traceback.format_exc())
            return None

    @timer()
    def run(self, batch:Batch) -> Batch:
        start = len(batch.content)
        logger.info(f"{self.name} started processing {start} samples")
        updates = Parallel(self.threads)(delayed(self._step)(sample) for sample in batch.content)
        batch.content = [sample for sample in updates if sample is not None]
        end = len(batch.content)
        if end <= start // 2:
            logger.warning(f"{self.name} filtered more than half of the samples")
        logger.info(f"{self.name} finished processing {end} samples")

        return batch

class FilterStep(Step):
    name:str    

class LocalPipeline:
    def __init__(self, steps:list[Step], reader:Reader):
        self.steps = steps
        self.reader = reader

    def _run(self, batch:Batch):
        batch = self.reader.read(batch)
        for step in self.steps:
            batch = step.run(batch)
        return batch

    @timer()
    def run(self, limit:int=None, workers:int=None):
        batches = self.reader.scheduler()
        for batch in batches[:limit]:
            self._run(batch)