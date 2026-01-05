import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    Function,
    FunctionType,
)
from settings import (
    STATE_VECTOR_SIZE, 
    OUTPUT_VECTOR_SIZE,
    DECAY_ENABLED,
    DECAY_FUNCTION,
    DECAY_OFFSET_IDS,
    DECAY_SCALE_IDS,
    DECAY_VALUE,
)
from typing import Dict, Optional, List

class Engram:
    def __init__(
            self, 
            vector: list[float],
            action: int, 
            outcome: float,
            trial_number: int = 0,
            trial_final_success: float = 0.0,
            trial_raw_reward: float = 0.0,
            trial_episode_length: int = 0,
            trial_is_success: bool = False,
        ):
        self.vector = vector
        self.action = action
        self.outcome = outcome
        self.trial_number = trial_number
        self.trial_final_success = trial_final_success
        self.trial_raw_reward = trial_raw_reward
        self.trial_episode_length = trial_episode_length
        self.trial_is_success = trial_is_success

    @staticmethod
    def from_record(record: list):
        return Engram(
            vector=record.fields[EngramField.vector],
            action=record.fields[EngramField.action],
            outcome=record.fields[EngramField.outcome],
            trial_number=record.fields.get(EngramField.trial_number, 0),
            trial_final_success=record.fields.get(EngramField.trial_final_success, 0.0),
            trial_raw_reward=record.fields.get(EngramField.trial_raw_reward, 0.0),
            trial_episode_length=record.fields.get(EngramField.trial_episode_length, 0),
            trial_is_success=record.fields.get(EngramField.trial_is_success, False),
        )
    
    @staticmethod
    def random(input_size: int, action_options: int):
        return Engram(
            vector=np.random.random(input_size).tolist(),
            
            # random int
            action=np.random.randint(0, action_options-1),
            #outcome=np.random.rand() # Should be between -1 and 1
            outcome=np.random.uniform(-1, 1)
        )

# enum class for field names
class EngramField:
    id = "id"
    insertion_index = "insertion_index"
    vector = "vector"
    action = "action"
    outcome = "outcome"
    trial_number = "trial_number"
    trial_final_success = "trial_final_success"
    trial_raw_reward = "trial_raw_reward"
    trial_episode_length = "trial_episode_length"
    trial_is_success = "trial_is_success"

class EngramStore:


    # static method
    @staticmethod
    def schema():
        fields = [
            FieldSchema(name=EngramField.id, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=EngramField.insertion_index, dtype=DataType.INT64, description='Sequential insertion index for decay ranking'),
            FieldSchema(name=EngramField.vector, dtype=DataType.FLOAT_VECTOR, dim=STATE_VECTOR_SIZE, description='The state embedding'),
            FieldSchema(name=EngramField.action, dtype=DataType.INT16, description='The action taken'),
            FieldSchema(name=EngramField.outcome, dtype=DataType.FLOAT, description="Negative means penalty, positive means reward"),
            FieldSchema(name=EngramField.trial_number, dtype=DataType.INT64, description='Trial number this engram came from'),
            FieldSchema(name=EngramField.trial_final_success, dtype=DataType.FLOAT, description='Normalized final success value of the trial'),
            FieldSchema(name=EngramField.trial_raw_reward, dtype=DataType.FLOAT, description='Raw reward from the trial'),
            FieldSchema(name=EngramField.trial_episode_length, dtype=DataType.INT64, description='Length of the episode in steps'),
            FieldSchema(name=EngramField.trial_is_success, dtype=DataType.BOOL, description='Whether the trial was successful (reward >= 200)'),
        ]
        schema = CollectionSchema(fields=fields, description="Collection of states")
        return schema

    def __init__(self, name: str = "lander", reset: bool = False):
        self.collection_name = name
        self._insertion_counter = 0  # Track sequential insertion index
        self.connect_to_db(reset)   
          
        #self.collection=Collection(name=self.collection_name)

    def connect_to_db(self, reset:bool = False):
        connections.connect(alias="default") 
        if reset and utility.has_collection(self.collection_name):
            print(f"Dropping collection {self.collection_name}")
            utility.drop_collection(self.collection_name)
            self._insertion_counter = 0
        else:
            print(f"Using existing collection {self.collection_name}")
        self.make_collection()
        # Initialize counter from existing collection if it exists and wasn't reset
        if not reset and utility.has_collection(self.collection_name):
            self._initialize_insertion_counter()      

    # def init(self):
    #     self.connect_to_db()
    #     if utility.has_collection(self.collection_name):
    #         utility.drop_collection(self.collection_name)
    #     self.make_collection()


    def make_collection(self):
        self.collection=Collection(name=self.collection_name, schema=self.schema())
        index = {
            "index_type": "IVF_FLAT", # Inverted File Flat: balanced between memory and speed
            "metric_type": "L2", # Euclidean distance
            "params": { "nlist": 1024 }, # 128 clusters, for speed of lookup
        }

        self.collection.create_index("vector", index)

    def _initialize_insertion_counter(self):
        """Initialize the insertion counter from the existing collection's max insertion_index."""
        try:
            self.collection.load()
            num_entities = self.collection.num_entities
            if num_entities == 0:
                self._insertion_counter = 0
                return
            
            # Query for the maximum insertion_index
            import numpy as np
            max_index = None
            # Sample multiple searches to find max insertion_index
            for _ in range(5):
                dummy_vector = np.random.random(STATE_VECTOR_SIZE).tolist()
                search_results = self.collection.search(
                    data=[dummy_vector],
                    limit=min(1000, num_entities),
                    param={"metric_type": "L2", "params": {"nprobe": 16}},
                    anns_field=EngramField.vector,
                    output_fields=[EngramField.insertion_index]
                )
                if search_results and len(search_results[0]) > 0:
                    indices = [hit.entity.get(EngramField.insertion_index) for hit in search_results[0] if EngramField.insertion_index in hit.entity]
                    if indices:
                        sample_max = max(indices)
                        if max_index is None or sample_max > max_index:
                            max_index = sample_max
            
            if max_index is not None:
                self._insertion_counter = max_index
            else:
                # If field doesn't exist in old collections, start from num_entities
                self._insertion_counter = num_entities
        except Exception:
            # On error, use num_entities as fallback
            self._insertion_counter = self.collection.num_entities

    def insert(self, record: Engram, trial_number: int = 0, trial_final_success: float = 0.0, 
               trial_raw_reward: float = 0.0, trial_episode_length: int = 0, trial_is_success: bool = False):       
        # Increment counter and assign insertion index
        self._insertion_counter += 1
        insertion_index = self._insertion_counter
        
        # Insert fields in schema order (excluding auto_id): insertion_index, vector, action, outcome, 
        # trial_number, trial_final_success, trial_raw_reward, trial_episode_length, trial_is_success
        self.collection.insert([
            [insertion_index],
            [record.vector], 
            [record.action], 
            [record.outcome],
            [trial_number],
            [trial_final_success],
            [trial_raw_reward],
            [trial_episode_length],
            [trial_is_success]
        ], 0.0001)

    def _get_max_insertion_index(self) -> int:
        """Get the current maximum insertion_index from the collection."""
        # Use the counter which tracks the max index
        return self._insertion_counter

    def _create_decay_ranker(self) -> Optional[Function]:
        """Create a decay ranker function based on current settings and collection state."""
        if not DECAY_ENABLED:
            return None
        
        # Get current max insertion_index to use as origin
        max_index = self._get_max_insertion_index()
        
        # If collection is empty, no decay needed
        if max_index == 0:
            return None
        
        # Create decay ranker function using insertion_index field
        ranker = Function(
            name="insertion_index_decay_ranker",
            input_field_names=[EngramField.insertion_index],
            function_type=FunctionType.RERANK,
            params={
                "reranker": "decay",
                "function": DECAY_FUNCTION,
                "origin": max_index,
                "offset": DECAY_OFFSET_IDS,
                "decay": DECAY_VALUE,
                "scale": DECAY_SCALE_IDS,
            }
        )
        return ranker
    
    def nearest(self, vector: list[float], limit: int) -> list[Engram, float]:
        self.collection.load()

        # Create decay ranker if enabled
        ranker = self._create_decay_ranker()

        # Prepare search parameters
        search_params = {
            "metric_type": "L2", # Euclidean distance
            "params": { 
                "nprobe": 16 # 16 clusters to search
            }
        }

        # Perform search with optional decay ranker
        output_fields = ["id", "insertion_index", "vector", "action", "outcome", EngramField.trial_number,
                        EngramField.trial_final_success, EngramField.trial_raw_reward, 
                        EngramField.trial_episode_length, EngramField.trial_is_success]
        if ranker is not None:
            records = self.collection.search(
                data=[vector],
                limit=limit,
                param=search_params,
                anns_field=EngramField.vector,
                output_fields=output_fields,
                ranker=ranker
            )
        else:
            records = self.collection.search(
                data=[vector],
                limit=limit,
                param=search_params,
                anns_field=EngramField.vector,
                output_fields=output_fields
            )
        
        # Map results into Record types
        result: list[tuple[Engram, float]] = []
        for record in records[0]:
            result.append((Engram.from_record(record), record.distance))
        return result
    
    def get_count(self) -> int:
        """Return the total number of engrams in the collection."""
        self.collection.load()
        return self.collection.num_entities
    
    def get_outcome_stats(self, sample_size: int = 1000) -> Dict[str, float]:
        """
        Sample engrams and return outcome distribution statistics.
        Returns a dict with 'positive_ratio', 'negative_ratio', 'mean_outcome'.
        """
        self.collection.load()
        total_count = self.collection.num_entities
        
        if total_count == 0:
            return {'positive_ratio': 0.0, 'negative_ratio': 0.0, 'mean_outcome': 0.0}
        
        # Sample up to sample_size engrams
        sample_limit = min(sample_size, total_count)
        
        # Query random sample by using a random vector and getting nearest
        import random
        random_vector = [[random.uniform(-1, 1) for _ in range(STATE_VECTOR_SIZE)]]
        
        records = self.collection.search(
            data=random_vector,
            limit=sample_limit,
            param={
                "metric_type": "L2",
                "params": {"nprobe": 16}
            },
            anns_field=EngramField.vector,
            output_fields=["outcome"]
        )
        
        outcomes = [record.fields["outcome"] for record in records[0]]
        positive_count = sum(1 for o in outcomes if o > 0)
        negative_count = sum(1 for o in outcomes if o < 0)
        mean_outcome = sum(outcomes) / len(outcomes) if outcomes else 0.0
        
        return {
            'positive_ratio': positive_count / len(outcomes) if outcomes else 0.0,
            'negative_ratio': negative_count / len(outcomes) if outcomes else 0.0,
            'mean_outcome': mean_outcome
        }

   
def test():
    store = EngramStore()
    akash = Engram(id=1, vector=[1.0, 2.0, 3.0], action=1, outcome=1.0)
    store.insert(akash)
    store.insert(Engram(id=2, vector=[-0.101, 0.001, 0.993], action=1, outcome=-1.0))
    store.insert(Engram(id=3, vector=[-0.2, 0.882, 0.303], action=2, outcome=0.5))
    store.insert(Engram(id=4, vector=[0.1, 0.45, 0.01], action=3, outcome=0.2))

    for record, distance in store.nearest(vector=[0.5, 0.5, 0.5]):
        print(record.action, record.outcome, distance)        
    




