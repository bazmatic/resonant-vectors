import numpy as np
from abc import ABC, abstractmethod
from settings import (
    STATE_VECTOR_SIZE, 
    OUTPUT_VECTOR_SIZE,
    DECAY_ENABLED,
    DECAY_FUNCTION,
    DECAY_OFFSET_IDS,
    DECAY_SCALE_IDS,
    DECAY_VALUE,
    VECTOR_SAVE_RATE,
)
from typing import Dict, Optional, List, Tuple

# Import Milvus only when needed
try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema, CollectionSchema, DataType,
        Collection,
        Function,
        FunctionType,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    # Create dummy classes for type hints when Milvus is not available
    class Function:
        pass
    class CollectionSchema:
        pass
    class FieldSchema:
        pass
    class DataType:
        pass
    class Collection:
        pass

class Engram:
    def __init__(
            self, 
            vector: list[float],
            action: int, 
            outcome: float,
            trial_final_success: float = 0.0,
        ):
        self.vector = vector
        self.action = action
        self.outcome = outcome
        self.trial_final_success = trial_final_success

    @staticmethod
    def from_record(record: list):
        return Engram(
            vector=record.fields[EngramField.vector],
            action=record.fields[EngramField.action],
            outcome=record.fields[EngramField.outcome],
            trial_final_success=record.fields.get(EngramField.trial_final_success, 0.0),
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
    trial_final_success = "trial_final_success"

class BaseEngramStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    def insert(self, record: 'Engram', trial_final_success: float = 0.0) -> None:
        """Insert a single engram record."""
        pass
    
    @abstractmethod
    def batch_insert(self, records: List['Engram'], trial_final_successes: List[float] = None) -> None:
        """Insert multiple engram records in a batch."""
        pass
    
    @abstractmethod
    def nearest(self, vector: list[float], limit: int) -> List[Tuple['Engram', float]]:
        """Find nearest engrams to the given vector. Returns list of (Engram, distance) tuples."""
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """Return the total number of engrams in the store."""
        pass
    
    @abstractmethod
    def get_outcome_stats(self, sample_size: int = 1000) -> Dict[str, float]:
        """Sample engrams and return outcome distribution statistics."""
        pass
    
    @abstractmethod
    def delete_oldest_records(self, count: int) -> int:
        """Delete the N oldest records based on insertion_index. Returns number deleted."""
        pass
    
    @abstractmethod
    def delete_random_records(self, count: int) -> int:
        """Delete N randomly selected records. Returns number deleted."""
        pass

class MilvusEngramStore(BaseEngramStore):
    """Milvus-based vector store implementation."""
    
    if not MILVUS_AVAILABLE:
        def __init__(self, *args, **kwargs):
            raise ImportError("pymilvus is not installed. Install it with: pip install pymilvus")


    # static method
    @staticmethod
    def schema():
        try:
            from pymilvus import FieldSchema, CollectionSchema, DataType
        except ImportError:
            raise ImportError("pymilvus is not installed. Install it with: pip install pymilvus")
        fields = [
            FieldSchema(name=EngramField.id, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=EngramField.insertion_index, dtype=DataType.INT64, description='Sequential insertion index for decay ranking'),
            FieldSchema(name=EngramField.vector, dtype=DataType.FLOAT_VECTOR, dim=STATE_VECTOR_SIZE, description='The state embedding'),
            FieldSchema(name=EngramField.action, dtype=DataType.INT16, description='The action taken'),
            FieldSchema(name=EngramField.outcome, dtype=DataType.FLOAT, description="Negative means penalty, positive means reward"),
            FieldSchema(name=EngramField.trial_final_success, dtype=DataType.FLOAT, description='Normalized final success value of the trial'),
        ]
        schema = CollectionSchema(fields=fields, description="Collection of states")
        return schema

    def __init__(self, name: str = "lander", reset: bool = False):
        self.collection_name = name
        self._insertion_counter = 0  # Track sequential insertion index
        self._collection_loaded = False  # Track whether collection is loaded in memory
        self._pending_flush = False  # Track if we have unflushed inserts
        self._cached_count = None  # Cache for entity count
        self.connect_to_db(reset)   
          
        #self.collection=Collection(name=self.collection_name)

    def connect_to_db(self, reset:bool = False):
        try:
            from pymilvus import connections, utility
        except ImportError:
            raise ImportError("pymilvus is not installed. Install it with: pip install pymilvus")
        connections.connect(alias="default") 
        if reset and utility.has_collection(self.collection_name):
            print(f"Dropping collection {self.collection_name}")
            utility.drop_collection(self.collection_name)
            self._insertion_counter = 0
            self._collection_loaded = False  # Reset load state after dropping collection
            self._pending_flush = False  # Reset flush flag after dropping collection
            self._cached_count = None  # Reset cached count
        else:
            print(f"Using existing collection {self.collection_name}")
        self.make_collection()
        # Initialize counter from existing collection if it exists and wasn't reset
        if not reset and utility.has_collection(self.collection_name):
            # Simply use num_entities as the starting point - much faster than searching
            self._ensure_loaded()
            self._insertion_counter = self.collection.num_entities
        else:
            # Load collection after creating new one (if reset or new collection)
            self._ensure_loaded()      

    # def init(self):
    #     self.connect_to_db()
    #     if utility.has_collection(self.collection_name):
    #         utility.drop_collection(self.collection_name)
    #     self.make_collection()


    def make_collection(self):
        try:
            from pymilvus import Collection
        except ImportError:
            raise ImportError("pymilvus is not installed. Install it with: pip install pymilvus")
        self.collection=Collection(name=self.collection_name, schema=self.schema())
        index = {
            "index_type": "IVF_FLAT", # Inverted File Flat: balanced between memory and speed
            "metric_type": "L2", # Euclidean distance
            "params": { "nlist": 1024 }, # 128 clusters, for speed of lookup
        }

        self.collection.create_index("vector", index)

    def _ensure_loaded(self):
        """Ensure the collection is loaded in memory. Loads only if not already loaded."""
        # Flush any pending inserts before loading/searching
        if self._pending_flush:
            self.collection.flush()
            self._pending_flush = False
        
        if not self._collection_loaded:
            self.collection.load()
            self._collection_loaded = True

    def insert(self, record: Engram, trial_final_success: float = 0.0):       
        # Apply random sampling if VECTOR_SAVE_RATE < 1.0
        if VECTOR_SAVE_RATE < 1.0:
            # Randomly decide whether to save this vector
            if np.random.random() >= VECTOR_SAVE_RATE:
                # Skip this vector
                return
        
        # Increment counter and assign insertion index
        self._insertion_counter += 1
        insertion_index = self._insertion_counter
        
        # Insert fields in schema order (excluding auto_id): insertion_index, vector, action, outcome, trial_final_success
        self.collection.insert([
            [insertion_index],
            [record.vector], 
            [record.action], 
            [record.outcome],
            [trial_final_success]
        ], 0.0001)
        
        # Mark that we have pending inserts that need to be flushed
        # We'll flush before the next search/load operation
        self._pending_flush = True
        # Invalidate cached count since we've inserted a new record
        self._cached_count = None

    def batch_insert(self, records: List[Engram], trial_final_successes: List[float] = None):
        """
        Insert multiple engrams in a single batch operation for better performance.
        All lists should have the same length as records, or be None to use defaults.
        """
        if len(records) == 0:
            return
        
        # Use defaults if not provided
        if trial_final_successes is None:
            trial_final_successes = [0.0] * len(records)
        
        # Apply random sampling if VECTOR_SAVE_RATE < 1.0
        if VECTOR_SAVE_RATE < 1.0:
            # Randomly select which indices to keep
            num_to_keep = int(len(records) * VECTOR_SAVE_RATE)
            if num_to_keep == 0:
                # If rate is very low and no vectors would be saved, return early
                return
            # Randomly select indices to keep
            selected_indices = np.random.choice(len(records), size=num_to_keep, replace=False)
            selected_indices = sorted(selected_indices)  # Sort for consistent ordering
        else:
            # Save all vectors
            selected_indices = list(range(len(records)))
        
        # Prepare batch data only for selected vectors
        insertion_indices = []
        vectors = []
        actions = []
        outcomes = []
        trial_finals = []
        
        for i in selected_indices:
            self._insertion_counter += 1
            insertion_indices.append(self._insertion_counter)
            vectors.append(records[i].vector)
            actions.append(records[i].action)
            outcomes.append(records[i].outcome)
            trial_finals.append(trial_final_successes[i])
        
        # Only insert if we have vectors to save
        if len(insertion_indices) > 0:
            # Batch insert all at once
            self.collection.insert([
                insertion_indices,
                vectors,
                actions,
                outcomes,
                trial_finals
            ], 0.0001)
            
            # Mark that we have pending inserts that need to be flushed
            self._pending_flush = True
            # Invalidate cached count since we've inserted new records
            self._cached_count = None

    def _get_max_insertion_index(self) -> int:
        """Get the current maximum insertion_index from the collection."""
        # Use the counter which tracks the max index
        return self._insertion_counter

    def _create_decay_ranker(self):
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
        self._ensure_loaded()

        # Create decay ranker if enabled
        ranker = self._create_decay_ranker()

        # Prepare search parameters
        search_params = {
            "metric_type": "L2", # Euclidean distance
            "params": { 
                "nprobe": 8 # 12 clusters to search
            }
        }

        # Perform search with optional decay ranker
        output_fields = ["id", "insertion_index", "vector", "action", "outcome", EngramField.trial_final_success]
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
        """Return the total number of engrams in the collection. Uses cached value if available."""
        # Return cached count if available and collection hasn't been reset
        if self._cached_count is not None:
            return self._cached_count
        
        # Calculate and cache the count
        self._ensure_loaded()
        self._cached_count = self.collection.num_entities
        return self._cached_count
    
    def get_outcome_stats(self, sample_size: int = 1000) -> Dict[str, float]:
        """
        Sample engrams and return outcome distribution statistics.
        Returns a dict with 'positive_ratio', 'negative_ratio', 'mean_outcome'.
        """
        self._ensure_loaded()
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
                "params": {"nprobe": 12}
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
    
    def delete_oldest_records(self, count: int) -> int:
        """
        Delete the N oldest records from the collection based on insertion_index.
        Returns the number of records actually deleted.
        """
        if count <= 0:
            return 0
        
        self._ensure_loaded()
        
        # Get total count to check if we have enough records
        total_count = self.get_count()
        if total_count == 0:
            return 0
        
        # If we have fewer records than requested, delete all existing records
        records_to_delete = min(count, total_count)
        
        # Milvus has a maximum query result window of 16384 (offset + limit <= 16384)
        # Since insertion_index is sequential starting from 1, the oldest records will have
        # the lowest insertion_index values. Query for records with low insertion_index.
        MAX_QUERY_LIMIT = 16384
        
        # Estimate an upper bound for insertion_index to query
        # Account for VECTOR_SAVE_RATE: if we need N records and save rate is R,
        # we might need to look at approximately N/R insertion_index values
        from settings import VECTOR_SAVE_RATE
        save_rate = max(VECTOR_SAVE_RATE, 0.01)  # Avoid division by zero
        estimated_max_index = int(records_to_delete / save_rate * 2)  # Safety multiplier of 2
        
        # Query records with insertion_index up to the estimate
        # Use limit that respects Milvus's 16384 constraint
        query_limit = min(MAX_QUERY_LIMIT, estimated_max_index)
        
        all_results = []
        current_max_index = min(estimated_max_index, self._insertion_counter)
        
        # Query and expand range if we don't have enough results
        while len(all_results) < records_to_delete and current_max_index <= self._insertion_counter:
            # Query a batch
            batch_results = self.collection.query(
                expr=f"{EngramField.insertion_index} >= 0 && {EngramField.insertion_index} <= {current_max_index}",
                output_fields=[EngramField.id, EngramField.insertion_index],
                limit=query_limit
            )
            
            if len(batch_results) == 0:
                # No records in this range, expand and try again
                if current_max_index >= self._insertion_counter:
                    break  # Reached the max, no more records
                current_max_index = min(current_max_index * 2, self._insertion_counter)
                continue
            
            all_results.extend(batch_results)
            
            # If we got fewer results than requested and haven't hit the limit, we might have enough
            # or need to expand. Check if we have enough records to delete.
            if len(all_results) >= records_to_delete:
                break
            
            # If we didn't hit the query limit, we've exhausted this range
            if len(batch_results) < query_limit:
                # Expand range if we still need more records
                if current_max_index >= self._insertion_counter:
                    break  # Reached the max
                current_max_index = min(current_max_index * 2, self._insertion_counter)
            else:
                # We hit the query limit, need to query in chunks using offset
                # But Milvus limits offset+limit <= 16384, so we need a different approach
                # Instead, expand the insertion_index range for next query
                current_max_index = min(current_max_index * 2, self._insertion_counter)
        
        if len(all_results) == 0:
            return 0
        
        # Sort by insertion_index (ascending) to get oldest first
        all_results.sort(key=lambda x: x[EngramField.insertion_index])
        
        # Remove duplicates (in case we queried overlapping ranges)
        seen_ids = set()
        unique_results = []
        for record in all_results:
            record_id = record[EngramField.id]
            if record_id not in seen_ids:
                seen_ids.add(record_id)
                unique_results.append(record)
        
        # Take only the number we need
        oldest_records = unique_results[:records_to_delete]
        
        if len(oldest_records) == 0:
            return 0
        
        # Extract IDs
        ids_to_delete = [record[EngramField.id] for record in oldest_records]
        
        # Delete in batches if we have too many IDs (to avoid expression length limits)
        BATCH_SIZE = 1000  # Conservative batch size for delete expressions
        total_deleted = 0
        
        for i in range(0, len(ids_to_delete), BATCH_SIZE):
            batch_ids = ids_to_delete[i:i + BATCH_SIZE]
            id_list_str = ",".join(str(id) for id in batch_ids)
            delete_expr = f"{EngramField.id} in [{id_list_str}]"
            self.collection.delete(delete_expr)
            total_deleted += len(batch_ids)
        
        # Flush to ensure deletion is persisted
        self.collection.flush()
        
        # Invalidate cached count
        self._cached_count = None
        
        return total_deleted
    
    def delete_random_records(self, count: int) -> int:
        """
        Delete N randomly selected records from the collection.
        Returns the number of records actually deleted.
        """
        if count <= 0:
            return 0
        
        self._ensure_loaded()
        
        # Get total count to check if we have enough records
        total_count = self.get_count()
        if total_count == 0:
            return 0
        
        # If we have fewer records than requested, delete all existing records
        records_to_delete = min(count, total_count)
        
        # Use random vector search to find random records
        # Generate random query vectors and collect unique records
        import random
        all_results = []
        seen_ids = set()
        max_attempts = 100  # Limit attempts to avoid infinite loop
        attempts = 0
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 8}
        }
        
        # Keep searching with random vectors until we have enough unique records
        while len(all_results) < records_to_delete and attempts < max_attempts:
            # Generate random query vector
            random_vector = [[random.uniform(-1, 1) for _ in range(STATE_VECTOR_SIZE)]]
            
            # Search for nearest vectors (this gives us a somewhat random sample)
            search_results = self.collection.search(
                data=random_vector,
                limit=min(records_to_delete * 2, 100),  # Get more than needed for randomness
                param=search_params,
                anns_field=EngramField.vector,
                output_fields=[EngramField.id]
            )
            
            # Add unique records to our collection
            for record in search_results[0]:
                record_id = record.fields[EngramField.id]
                if record_id not in seen_ids:
                    seen_ids.add(record_id)
                    all_results.append(record_id)
                    if len(all_results) >= records_to_delete:
                        break
            
            attempts += 1
        
        # If we still don't have enough, fall back to querying all IDs and randomly selecting
        if len(all_results) < records_to_delete:
            # Query all records for their IDs
            all_ids_results = self.collection.query(
                expr="",
                output_fields=[EngramField.id],
                limit=16384  # Milvus max
            )
            
            all_ids = [record[EngramField.id] for record in all_ids_results]
            
            # If we got some from random search, remove those from consideration
            candidate_ids = [id for id in all_ids if id not in seen_ids]
            
            # Randomly sample from remaining IDs
            needed = records_to_delete - len(all_results)
            if len(candidate_ids) > 0:
                selected_ids = random.sample(candidate_ids, min(needed, len(candidate_ids)))
                all_results.extend(selected_ids)
        
        # Take only the number we need
        ids_to_delete = all_results[:records_to_delete]
        
        if len(ids_to_delete) == 0:
            return 0
        
        # Delete in batches if we have too many IDs (to avoid expression length limits)
        BATCH_SIZE = 1000  # Conservative batch size for delete expressions
        total_deleted = 0
        
        for i in range(0, len(ids_to_delete), BATCH_SIZE):
            batch_ids = ids_to_delete[i:i + BATCH_SIZE]
            id_list_str = ",".join(str(id) for id in batch_ids)
            delete_expr = f"{EngramField.id} in [{id_list_str}]"
            self.collection.delete(delete_expr)
            total_deleted += len(batch_ids)
        
        # Flush to ensure deletion is persisted
        self.collection.flush()
        
        # Invalidate cached count
        self._cached_count = None
        
        return total_deleted

def create_engram_store(name: str = "lander", reset: bool = False) -> BaseEngramStore:
    """
    Factory function to create an EngramStore instance based on VECTOR_STORE_TYPE setting.
    
    Args:
        name: Name for the store instance
        reset: Whether to reset/clear existing data
        
    Returns:
        BaseEngramStore instance (either MilvusEngramStore or FAISSEngramStore)
    """
    from settings import VECTOR_STORE_TYPE
    
    if VECTOR_STORE_TYPE == "faiss":
        from faiss_store import FAISSEngramStore
        return FAISSEngramStore(name, reset)
    elif VECTOR_STORE_TYPE == "milvus":
        return MilvusEngramStore(name, reset)
    else:
        raise ValueError(f"Unknown VECTOR_STORE_TYPE: {VECTOR_STORE_TYPE}. Must be 'milvus' or 'faiss'")

# Backward compatibility: EngramStore class that delegates to factory
class EngramStore:
    """Backward compatibility wrapper that creates the appropriate store type based on settings."""
    def __new__(cls, name: str = "lander", reset: bool = False):
        return create_engram_store(name, reset)

def test():
    store = create_engram_store()
    akash = Engram(id=1, vector=[1.0, 2.0, 3.0], action=1, outcome=1.0)
    store.insert(akash)
    store.insert(Engram(id=2, vector=[-0.101, 0.001, 0.993], action=1, outcome=-1.0))
    store.insert(Engram(id=3, vector=[-0.2, 0.882, 0.303], action=2, outcome=0.5))
    store.insert(Engram(id=4, vector=[0.1, 0.45, 0.01], action=3, outcome=0.2))

    for record, distance in store.nearest(vector=[0.5, 0.5, 0.5]):
        print(record.action, record.outcome, distance)        
    




