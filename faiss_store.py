import numpy as np
import faiss
from typing import Dict, Optional, List, Tuple
from engram import Engram, EngramField, BaseEngramStore
from settings import (
    STATE_VECTOR_SIZE,
    DECAY_ENABLED,
    DECAY_FUNCTION,
    DECAY_OFFSET_IDS,
    DECAY_SCALE_IDS,
    DECAY_VALUE,
    VECTOR_SAVE_RATE,
)


class FAISSEngramStore(BaseEngramStore):
    """In-memory vector store implementation using FAISS."""
    
    def __init__(self, name: str = "lander", reset: bool = False):
        """
        Initialize FAISS store.
        
        Args:
            name: Store name (used for identification, FAISS doesn't persist)
            reset: If True, start with empty store (always True for FAISS)
        """
        self.name = name
        self._insertion_counter = 0
        self._cached_count = None
        
        # Initialize FAISS index for L2 (Euclidean) distance
        # Note: IndexFlatL2 returns squared L2 distances, but we convert to actual
        # L2 distances in nearest() for consistency with Milvus backend
        self.index = faiss.IndexFlatL2(STATE_VECTOR_SIZE)
        
        # Parallel data structures for metadata and vectors
        # Each index in these lists corresponds to the same FAISS vector ID
        self.vectors = []  # List of vectors (stored for deletion support)
        self.metadata = []  # List of dicts: {action, outcome, trial_final_success}
        self.insertion_indices = []  # List of insertion_index values
        
        if reset:
            print(f"Initializing new FAISS store: {name}")
        else:
            print(f"FAISS store '{name}' is always empty on startup (in-memory only)")
    
    def insert(self, record: Engram, trial_final_success: float = 0.0) -> None:
        """Insert a single engram record."""
        # Apply random sampling if VECTOR_SAVE_RATE < 1.0
        if VECTOR_SAVE_RATE < 1.0:
            if np.random.random() >= VECTOR_SAVE_RATE:
                return
        
        # Increment counter and assign insertion index
        self._insertion_counter += 1
        insertion_index = self._insertion_counter
        
        # Convert vector to numpy array and add to FAISS index
        vector_array = np.array([record.vector], dtype=np.float32)
        self.index.add(vector_array)
        
        # Store vector, metadata in parallel structures
        self.vectors.append(record.vector.copy())  # Store copy of vector
        self.metadata.append({
            EngramField.action: record.action,
            EngramField.outcome: record.outcome,
            EngramField.trial_final_success: trial_final_success,
        })
        self.insertion_indices.append(insertion_index)
        
        # Invalidate cached count
        self._cached_count = None
    
    def batch_insert(self, records: List[Engram], trial_final_successes: List[float] = None) -> None:
        """Insert multiple engrams in a single batch operation."""
        if len(records) == 0:
            return
        
        # Use defaults if not provided
        if trial_final_successes is None:
            trial_final_successes = [0.0] * len(records)
        
        # Apply random sampling if VECTOR_SAVE_RATE < 1.0
        if VECTOR_SAVE_RATE < 1.0:
            num_to_keep = int(len(records) * VECTOR_SAVE_RATE)
            if num_to_keep == 0:
                return
            selected_indices = np.random.choice(len(records), size=num_to_keep, replace=False)
            selected_indices = sorted(selected_indices)
        else:
            selected_indices = list(range(len(records)))
        
        # Prepare batch data
        vectors_to_add = []
        metadata_to_add = []
        insertion_indices_to_add = []
        
        for i in selected_indices:
            self._insertion_counter += 1
            vectors_to_add.append(records[i].vector)
            metadata_to_add.append({
                EngramField.action: records[i].action,
                EngramField.outcome: records[i].outcome,
                EngramField.trial_final_success: trial_final_successes[i],
            })
            insertion_indices_to_add.append(self._insertion_counter)
        
        if len(vectors_to_add) > 0:
            # Add all vectors at once to FAISS
            vectors_array = np.array(vectors_to_add, dtype=np.float32)
            self.index.add(vectors_array)
            
            # Store vectors and metadata
            self.vectors.extend([v.copy() for v in vectors_to_add])
            self.metadata.extend(metadata_to_add)
            self.insertion_indices.extend(insertion_indices_to_add)
            
            # Invalidate cached count
            self._cached_count = None
    
    def _get_max_insertion_index(self) -> int:
        """Get the current maximum insertion_index."""
        return self._insertion_counter
    
    def _apply_decay_ranking(
        self, 
        distances: np.ndarray, 
        indices: np.ndarray, 
        limit: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply decay ranking to search results.
        
        Args:
            distances: Array of actual L2 (Euclidean) distances (not squared)
            indices: Array of FAISS indices corresponding to distances
            limit: Maximum number of results to return
            
        Returns:
            Tuple of (sorted_distances, sorted_indices) after decay ranking
        """
        if not DECAY_ENABLED:
            # Return top results without decay
            sorted_order = np.argsort(distances[0])[:limit]
            return distances[0][sorted_order], indices[0][sorted_order]
        
        max_index = self._get_max_insertion_index()
        if max_index == 0:
            # Empty collection
            return distances[0][:limit], indices[0][:limit]
        
        # Get insertion indices for all candidates
        insertion_vals = np.array([self.insertion_indices[idx] for idx in indices[0]])
        dists = distances[0]
        
        # Calculate decay multipliers based on insertion_index distance from max
        id_distances = max_index - insertion_vals
        
        # Apply decay function
        if DECAY_FUNCTION == "gauss":
            # Gaussian decay
            scale_factor = id_distances / DECAY_SCALE_IDS if DECAY_SCALE_IDS > 0 else 0
            decay_multipliers = np.where(
                id_distances <= DECAY_OFFSET_IDS,
                1.0,  # No decay within offset
                np.exp(-0.5 * ((scale_factor - DECAY_OFFSET_IDS / DECAY_SCALE_IDS) ** 2)) * (1 - DECAY_VALUE) + DECAY_VALUE
            )
        elif DECAY_FUNCTION == "exp":
            # Exponential decay
            effective_distance = np.maximum(0, id_distances - DECAY_OFFSET_IDS)
            decay_multipliers = np.where(
                id_distances <= DECAY_OFFSET_IDS,
                1.0,  # No decay within offset
                DECAY_VALUE + (1.0 - DECAY_VALUE) * np.exp(-effective_distance / DECAY_SCALE_IDS)
            )
        elif DECAY_FUNCTION == "linear":
            # Linear decay
            effective_distance = np.maximum(0, id_distances - DECAY_OFFSET_IDS)
            decay_multipliers = np.where(
                id_distances <= DECAY_OFFSET_IDS,
                1.0,  # No decay within offset
                np.maximum(DECAY_VALUE, 1.0 - effective_distance / DECAY_SCALE_IDS)
            )
        else:
            # Default: no decay
            decay_multipliers = np.ones_like(id_distances)
        
        # Adjust distances (multiply by inverse of decay, so higher decay = worse score = higher distance)
        adjusted_distances = dists / (decay_multipliers + 1e-10)  # Small epsilon to avoid division by zero
        
        # Sort by adjusted distance and return top limit
        sorted_order = np.argsort(adjusted_distances)[:limit]
        return adjusted_distances[sorted_order], indices[0][sorted_order]
    
    def nearest(self, vector: list[float], limit: int) -> List[Tuple[Engram, float]]:
        """Find nearest engrams to the given vector."""
        if self.index.ntotal == 0:
            return []
        
        # Convert query vector to numpy array
        query_vector = np.array([vector], dtype=np.float32)
        
        # Search in FAISS - search for more candidates if decay is enabled
        # (decay ranking may reorder results, so we need more candidates)
        search_limit = limit * 2 if DECAY_ENABLED else limit
        search_limit = min(search_limit, self.index.ntotal)
        
        if search_limit == 0:
            return []
        
        distances, indices = self.index.search(query_vector, search_limit)
        
        # FAISS IndexFlatL2 returns squared L2 distances. Convert to actual L2 distances
        # by taking square root for consistency with Milvus (which returns actual L2 distances)
        # This ensures distance-based weighting behaves the same across backends
        actual_distances = np.sqrt(distances)
        
        # Apply decay ranking if enabled
        if DECAY_ENABLED and len(indices[0]) > 0:
            adjusted_distances, final_indices = self._apply_decay_ranking(actual_distances, indices, limit)
        else:
            # Take top results
            final_indices = indices[0][:limit]
            adjusted_distances = actual_distances[0][:limit]
        
        # Build result list
        result: List[Tuple[Engram, float]] = []
        for idx, dist in zip(final_indices, adjusted_distances):
            idx_int = int(idx)
            if idx_int >= len(self.metadata) or idx_int < 0:
                continue  # Safety check
            meta = self.metadata[idx_int]
            stored_vector = self.vectors[idx_int]  # Get stored vector
            engram = Engram(
                vector=stored_vector,
                action=meta[EngramField.action],
                outcome=meta[EngramField.outcome],
                trial_final_success=meta[EngramField.trial_final_success],
            )
            result.append((engram, float(dist)))
        
        return result
    
    def get_count(self) -> int:
        """Return the total number of engrams in the store."""
        if self._cached_count is not None:
            return self._cached_count
        self._cached_count = self.index.ntotal
        return self._cached_count
    
    def get_outcome_stats(self, sample_size: int = 1000) -> Dict[str, float]:
        """Sample engrams and return outcome distribution statistics."""
        total_count = self.get_count()
        
        if total_count == 0:
            return {'positive_ratio': 0.0, 'negative_ratio': 0.0, 'mean_outcome': 0.0}
        
        # Sample up to sample_size engrams using random vector search
        sample_limit = min(sample_size, total_count)
        
        # Generate random query vector
        import random
        random_vector = [random.uniform(-1, 1) for _ in range(STATE_VECTOR_SIZE)]
        
        # Search for nearest to get a sample
        query_vector = np.array([random_vector], dtype=np.float32)
        distances, indices = self.index.search(query_vector, sample_limit)
        
        # Extract outcomes
        outcomes = [self.metadata[idx][EngramField.outcome] for idx in indices[0] if idx < len(self.metadata)]
        
        if not outcomes:
            return {'positive_ratio': 0.0, 'negative_ratio': 0.0, 'mean_outcome': 0.0}
        
        positive_count = sum(1 for o in outcomes if o > 0)
        negative_count = sum(1 for o in outcomes if o < 0)
        mean_outcome = sum(outcomes) / len(outcomes)
        
        return {
            'positive_ratio': positive_count / len(outcomes),
            'negative_ratio': negative_count / len(outcomes),
            'mean_outcome': mean_outcome
        }
    
    def delete_oldest_records(self, count: int) -> int:
        """Delete the N oldest records based on insertion_index."""
        if count <= 0:
            return 0
        
        total_count = self.get_count()
        if total_count == 0:
            return 0
        
        records_to_delete = min(count, total_count)
        
        # Create list of (index, insertion_index) pairs
        index_pairs = [(i, self.insertion_indices[i]) for i in range(len(self.insertion_indices))]
        
        # Sort by insertion_index to get oldest first
        index_pairs.sort(key=lambda x: x[1])
        
        # Get indices to delete
        indices_to_delete = [idx for idx, _ in index_pairs[:records_to_delete]]
        indices_to_delete_set = set(indices_to_delete)
        
        # Rebuild index and all data structures excluding deleted records
        # FAISS doesn't support deletion, so we rebuild the entire index
        new_index = faiss.IndexFlatL2(STATE_VECTOR_SIZE)
        new_vectors = []
        new_metadata = []
        new_insertion_indices = []
        
        for i in range(len(self.metadata)):
            if i not in indices_to_delete_set:
                new_vectors.append(self.vectors[i])
                new_metadata.append(self.metadata[i])
                new_insertion_indices.append(self.insertion_indices[i])
        
        # Rebuild FAISS index with remaining vectors
        if len(new_vectors) > 0:
            vectors_array = np.array(new_vectors, dtype=np.float32)
            new_index.add(vectors_array)
        
        # Replace old structures with new ones
        self.index = new_index
        self.vectors = new_vectors
        self.metadata = new_metadata
        self.insertion_indices = new_insertion_indices
        
        # Invalidate cached count
        self._cached_count = None
        
        return records_to_delete
    
    def delete_lowest_score_records(self, count: int) -> int:
        """Delete N records with the lowest outcome scores."""
        if count <= 0:
            return 0
        
        total_count = self.get_count()
        if total_count == 0:
            return 0
        
        records_to_delete = min(count, total_count)
        
        # Create list of (index, outcome) pairs
        index_outcome_pairs = [
            (i, self.metadata[i][EngramField.outcome]) 
            for i in range(len(self.metadata))
        ]
        
        # Sort by outcome (ascending) to get lowest scores first
        index_outcome_pairs.sort(key=lambda x: x[1])
        
        # Get indices to delete (lowest scores)
        indices_to_delete = [idx for idx, _ in index_outcome_pairs[:records_to_delete]]
        indices_to_delete_set = set(indices_to_delete)
        
        # Rebuild index and all data structures excluding deleted records
        # FAISS doesn't support deletion, so we rebuild the entire index
        new_index = faiss.IndexFlatL2(STATE_VECTOR_SIZE)
        new_vectors = []
        new_metadata = []
        new_insertion_indices = []
        
        for i in range(len(self.metadata)):
            if i not in indices_to_delete_set:
                new_vectors.append(self.vectors[i])
                new_metadata.append(self.metadata[i])
                new_insertion_indices.append(self.insertion_indices[i])
        
        # Rebuild FAISS index with remaining vectors
        if len(new_vectors) > 0:
            vectors_array = np.array(new_vectors, dtype=np.float32)
            new_index.add(vectors_array)
        
        # Replace old structures with new ones
        self.index = new_index
        self.vectors = new_vectors
        self.metadata = new_metadata
        self.insertion_indices = new_insertion_indices
        
        # Invalidate cached count
        self._cached_count = None
        
        return records_to_delete
    
    def delete_smallest_absolute_reward_records(self, count: int) -> int:
        """Delete N records with the smallest absolute outcome values (closest to zero)."""
        if count <= 0:
            return 0
        
        total_count = self.get_count()
        if total_count == 0:
            return 0
        
        records_to_delete = min(count, total_count)
        
        # Create list of (index, abs(outcome)) pairs
        index_abs_outcome_pairs = [
            (i, abs(self.metadata[i][EngramField.outcome])) 
            for i in range(len(self.metadata))
        ]
        
        # Sort by absolute outcome (ascending) to get smallest absolute values first
        index_abs_outcome_pairs.sort(key=lambda x: x[1])
        
        # Get indices to delete (smallest absolute outcomes)
        indices_to_delete = [idx for idx, _ in index_abs_outcome_pairs[:records_to_delete]]
        indices_to_delete_set = set(indices_to_delete)
        
        # Rebuild index and all data structures excluding deleted records
        # FAISS doesn't support deletion, so we rebuild the entire index
        new_index = faiss.IndexFlatL2(STATE_VECTOR_SIZE)
        new_vectors = []
        new_metadata = []
        new_insertion_indices = []
        
        for i in range(len(self.metadata)):
            if i not in indices_to_delete_set:
                new_vectors.append(self.vectors[i])
                new_metadata.append(self.metadata[i])
                new_insertion_indices.append(self.insertion_indices[i])
        
        # Rebuild FAISS index with remaining vectors
        if len(new_vectors) > 0:
            vectors_array = np.array(new_vectors, dtype=np.float32)
            new_index.add(vectors_array)
        
        # Replace old structures with new ones
        self.index = new_index
        self.vectors = new_vectors
        self.metadata = new_metadata
        self.insertion_indices = new_insertion_indices
        
        # Invalidate cached count
        self._cached_count = None
        
        return records_to_delete
    
    def delete_random_records(self, count: int) -> int:
        """Delete N randomly selected records."""
        if count <= 0:
            return 0
        
        total_count = self.get_count()
        if total_count == 0:
            return 0
        
        records_to_delete = min(count, total_count)
        
        # Randomly select indices to delete
        indices_to_delete = np.random.choice(
            len(self.metadata), 
            size=records_to_delete, 
            replace=False
        ).tolist()
        indices_to_delete_set = set(indices_to_delete)
        
        # Rebuild index and all data structures excluding deleted records
        # FAISS doesn't support deletion, so we rebuild the entire index
        new_index = faiss.IndexFlatL2(STATE_VECTOR_SIZE)
        new_vectors = []
        new_metadata = []
        new_insertion_indices = []
        
        for i in range(len(self.metadata)):
            if i not in indices_to_delete_set:
                new_vectors.append(self.vectors[i])
                new_metadata.append(self.metadata[i])
                new_insertion_indices.append(self.insertion_indices[i])
        
        # Rebuild FAISS index with remaining vectors
        if len(new_vectors) > 0:
            vectors_array = np.array(new_vectors, dtype=np.float32)
            new_index.add(vectors_array)
        
        # Replace old structures with new ones
        self.index = new_index
        self.vectors = new_vectors
        self.metadata = new_metadata
        self.insertion_indices = new_insertion_indices
        
        # Invalidate cached count
        self._cached_count = None
        
        return records_to_delete