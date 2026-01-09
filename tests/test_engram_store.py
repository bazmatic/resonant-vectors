import pytest
import numpy as np
from unittest.mock import MagicMock, Mock, patch, call
from engram import Engram, EngramStore, EngramField, MilvusEngramStore, create_engram_store
from pymilvus import FieldSchema, CollectionSchema, DataType, Function, FunctionType
from settings import STATE_VECTOR_SIZE


class TestMilvusEngramStore:
    """Tests for EngramStore class with mocked Milvus."""
    
    @pytest.fixture
    def mock_collection(self):
        """Mock Milvus collection."""
        collection = MagicMock()
        collection.num_entities = 0
        collection.load = MagicMock()
        collection.insert = MagicMock()
        collection.search = MagicMock(return_value=[[], []])
        collection.create_index = MagicMock()
        return collection
    
    @pytest.fixture
    def mock_utility(self):
        """Mock Milvus utility."""
        utility = MagicMock()
        utility.has_collection = MagicMock(return_value=False)
        utility.drop_collection = MagicMock()
        return utility
    
    @pytest.fixture
    def mock_connections(self):
        """Mock Milvus connections."""
        connections = MagicMock()
        connections.connect = MagicMock()
        return connections
    
    @pytest.fixture
    def mock_collection_class(self, mock_collection):
        """Mock Collection class."""
        with patch('engram.Collection') as mock_collection_class:
            mock_collection_class.return_value = mock_collection
            yield mock_collection_class
    
    def test_schema_creation(self, mock_settings):
        """Test schema creation with correct fields."""
        schema = MilvusEngramStore.schema()
        
        assert isinstance(schema, CollectionSchema)
        fields = schema.fields
        
        # Check all required fields exist
        field_names = [field.name for field in fields]
        assert EngramField.id in field_names
        assert EngramField.insertion_index in field_names
        assert EngramField.vector in field_names
        assert EngramField.action in field_names
        assert EngramField.outcome in field_names
        assert EngramField.trial_final_success in field_names
        
        # Check primary key
        id_field = next(f for f in fields if f.name == EngramField.id)
        assert id_field.is_primary == True
        assert id_field.auto_id == True
        
        # Check vector dimension
        vector_field = next(f for f in fields if f.name == EngramField.vector)
        assert vector_field.dtype == DataType.FLOAT_VECTOR
        assert vector_field.params['dim'] == STATE_VECTOR_SIZE
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_init_creates_collection(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test that __init__ creates collection and connects."""
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_collection.load = MagicMock()
        mock_collection.create_index = MagicMock()
        mock_collection_class.return_value = mock_collection
        mock_utility.has_collection.return_value = False
        
        store = MilvusEngramStore("test_collection", reset=True)
        
        mock_connections.connect.assert_called_once_with(alias="default")
        mock_collection_class.assert_called_once()
        mock_collection.create_index.assert_called_once()
        assert store.collection_name == "test_collection"
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_init_with_reset_drops_collection(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test that __init__ with reset=True drops existing collection."""
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_collection.load = MagicMock()
        mock_collection.create_index = MagicMock()
        mock_collection_class.return_value = mock_collection
        mock_utility.has_collection.return_value = True
        
        store = MilvusEngramStore("test_collection", reset=True)
        
        mock_utility.drop_collection.assert_called_once_with("test_collection")
        assert store._insertion_counter == 0
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_insert_tracks_insertion_index(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test that insert increments and tracks insertion_index."""
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_collection.load = MagicMock()
        mock_collection.create_index = MagicMock()
        mock_collection_class.return_value = mock_collection
        mock_utility.has_collection.return_value = False
        
        store = MilvusEngramStore("test_collection", reset=True)
        engram = Engram(vector=[0.1] * STATE_VECTOR_SIZE, action=1, outcome=0.5)
        
        store.insert(engram, trial_final_success=0.8)
        
        assert store._insertion_counter == 1
        mock_collection.insert.assert_called_once()
        call_args = mock_collection.insert.call_args[0][0]
        # Check field order: insertion_index, vector, action, outcome, trial_final_success
        assert call_args[0] == [1]  # insertion_index
        assert call_args[1] == [engram.vector]  # vector
        assert call_args[2] == [engram.action]  # action
        assert call_args[3] == [engram.outcome]  # outcome
        assert call_args[4] == [0.8]  # trial_final_success
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_nearest_without_decay_ranker(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test nearest search without decay ranker when decay is disabled."""
        with patch('engram.DECAY_ENABLED', False):
            mock_collection = MagicMock()
            mock_collection.num_entities = 0
            mock_collection.load = MagicMock()
            mock_collection.create_index = MagicMock()
            
            # Mock search results
            mock_record = MagicMock()
            mock_record.fields = {
                EngramField.vector: [0.1] * STATE_VECTOR_SIZE,
                EngramField.action: 1,
                EngramField.outcome: 0.5,
                EngramField.trial_final_success: 0.0,
            }
            mock_record.distance = 0.123
            mock_collection.search.return_value = [[mock_record]]
            
            mock_collection_class.return_value = mock_collection
            mock_utility.has_collection.return_value = False
            
            store = MilvusEngramStore("test_collection", reset=True)
            vector = [0.5] * STATE_VECTOR_SIZE
            limit = 10
            
            results = store.nearest(vector, limit)
            
            mock_collection.load.assert_called()
            mock_collection.search.assert_called_once()
            call_kwargs = mock_collection.search.call_args[1]
            assert 'ranker' not in call_kwargs or call_kwargs.get('ranker') is None
            assert len(results) == 1
            assert isinstance(results[0][0], Engram)
            assert results[0][1] == 0.123
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_nearest_with_decay_ranker(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test nearest search with decay ranker when decay is enabled."""
        with patch('engram.DECAY_ENABLED', True):
            with patch('engram.DECAY_FUNCTION', 'gauss'):
                with patch('engram.DECAY_OFFSET_IDS', 40000):
                    with patch('engram.DECAY_SCALE_IDS', 200000):
                        with patch('engram.DECAY_VALUE', 0.9):
                            mock_collection = MagicMock()
                            mock_collection.num_entities = 0
                            mock_collection.load = MagicMock()
                            mock_collection.create_index = MagicMock()
                            mock_collection.search.return_value = [[], []]
                            mock_collection_class.return_value = mock_collection
                            mock_utility.has_collection.return_value = False
                            
                            store = MilvusEngramStore("test_collection", reset=True)
                            store._insertion_counter = 1000  # Set counter so decay ranker is created
                            vector = [0.5] * STATE_VECTOR_SIZE
                            limit = 10
                            
                            results = store.nearest(vector, limit)
                            
                            mock_collection.search.assert_called_once()
                            call_kwargs = mock_collection.search.call_args[1]
                            assert 'ranker' in call_kwargs
                            assert call_kwargs['ranker'] is not None
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_nearest_empty_collection_no_decay(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test that empty collection doesn't create decay ranker."""
        with patch('engram.DECAY_ENABLED', True):
            mock_collection = MagicMock()
            mock_collection.num_entities = 0
            mock_collection.load = MagicMock()
            mock_collection.create_index = MagicMock()
            mock_collection.search.return_value = [[], []]
            mock_collection_class.return_value = mock_collection
            mock_utility.has_collection.return_value = False
            
            store = MilvusEngramStore("test_collection", reset=True)
            assert store._insertion_counter == 0
            
            vector = [0.5] * STATE_VECTOR_SIZE
            limit = 10
            
            results = store.nearest(vector, limit)
            
            # Should not create decay ranker for empty collection
            call_kwargs = mock_collection.search.call_args[1]
            assert 'ranker' not in call_kwargs or call_kwargs.get('ranker') is None
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_get_count(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test get_count returns correct entity count."""
        mock_collection = MagicMock()
        mock_collection.num_entities = 42
        mock_collection.load = MagicMock()
        mock_collection.create_index = MagicMock()
        mock_collection_class.return_value = mock_collection
        mock_utility.has_collection.return_value = False
        
        store = MilvusEngramStore("test_collection", reset=True)
        
        count = store.get_count()
        
        assert count == 42
        mock_collection.load.assert_called()
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_get_outcome_stats_empty_collection(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test get_outcome_stats with empty collection."""
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_collection.load = MagicMock()
        mock_collection.create_index = MagicMock()
        mock_collection.search.return_value = [[], []]
        mock_collection_class.return_value = mock_collection
        mock_utility.has_collection.return_value = False
        
        store = MilvusEngramStore("test_collection", reset=True)
        
        stats = store.get_outcome_stats()
        
        assert stats == {'positive_ratio': 0.0, 'negative_ratio': 0.0, 'mean_outcome': 0.0}
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_get_outcome_stats_with_data(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test get_outcome_stats calculation."""
        mock_collection = MagicMock()
        mock_collection.num_entities = 100
        mock_collection.load = MagicMock()
        mock_collection.create_index = MagicMock()
        
        # Mock search results with outcomes
        mock_records = []
        outcomes = [0.8, 0.5, -0.3, 0.2, -0.1]
        for outcome in outcomes:
            mock_record = MagicMock()
            mock_record.fields = {"outcome": outcome}
            mock_records.append(mock_record)
        
        mock_collection.search.return_value = [mock_records]
        mock_collection_class.return_value = mock_collection
        mock_utility.has_collection.return_value = False
        
        store = MilvusEngramStore("test_collection", reset=True)
        
        stats = store.get_outcome_stats(sample_size=1000)
        
        # outcomes = [0.8, 0.5, -0.3, 0.2, -0.1]
        # Positive: 0.8, 0.5, 0.2 = 3 out of 5 = 0.6
        # Negative: -0.3, -0.1 = 2 out of 5 = 0.4
        assert stats['positive_ratio'] == pytest.approx(0.6)  # 3 out of 5 positive
        assert stats['negative_ratio'] == pytest.approx(0.4)  # 2 out of 5 negative
        assert stats['mean_outcome'] == pytest.approx(0.22)  # (0.8 + 0.5 - 0.3 + 0.2 - 0.1) / 5
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_create_decay_ranker_when_enabled(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test _create_decay_ranker creates ranker when decay is enabled."""
        with patch('engram.DECAY_ENABLED', True):
            with patch('engram.DECAY_FUNCTION', 'gauss'):
                with patch('engram.DECAY_OFFSET_IDS', 40000):
                    with patch('engram.DECAY_SCALE_IDS', 200000):
                        with patch('engram.DECAY_VALUE', 0.9):
                            mock_collection = MagicMock()
                            mock_collection.num_entities = 0
                            mock_collection.load = MagicMock()
                            mock_collection.create_index = MagicMock()
                            mock_collection_class.return_value = mock_collection
                            mock_utility.has_collection.return_value = False
                            
                            store = MilvusEngramStore("test_collection", reset=True)
                            store._insertion_counter = 5000
                            
                            ranker = store._create_decay_ranker()
                            
                            assert ranker is not None
                            assert isinstance(ranker, Function)
                            assert ranker.name == "insertion_index_decay_ranker"
                            # Function object may not expose function_type directly, but we can check params
                            assert hasattr(ranker, 'params') or hasattr(ranker, 'input_field_names')
    
    @patch('engram.connections')
    @patch('engram.utility')
    @patch('engram.Collection')
    def test_create_decay_ranker_when_disabled(self, mock_collection_class, mock_utility, mock_connections, mock_settings):
        """Test _create_decay_ranker returns None when decay is disabled."""
        with patch('engram.DECAY_ENABLED', False):
            mock_collection = MagicMock()
            mock_collection.num_entities = 0
            mock_collection.load = MagicMock()
            mock_collection.create_index = MagicMock()
            mock_collection_class.return_value = mock_collection
            mock_utility.has_collection.return_value = False
            
            store = MilvusEngramStore("test_collection", reset=True)
            store._insertion_counter = 5000
            
            ranker = store._create_decay_ranker()
            
            assert ranker is None


class TestFAISSEngramStore:
    """Tests for FAISSEngramStore class."""
    
    @pytest.fixture
    def faiss_store(self):
        """Create a FAISSEngramStore instance for testing."""
        from faiss_store import FAISSEngramStore
        return FAISSEngramStore("test_faiss", reset=True)
    
    def test_insert(self, faiss_store):
        """Test inserting an engram."""
        engram = Engram(vector=[0.1] * STATE_VECTOR_SIZE, action=1, outcome=0.5)
        faiss_store.insert(engram, trial_final_success=0.8)
        
        assert faiss_store._insertion_counter == 1
        assert faiss_store.get_count() == 1
    
    def test_batch_insert(self, faiss_store):
        """Test batch inserting engrams."""
        engrams = [
            Engram(vector=[0.1] * STATE_VECTOR_SIZE, action=1, outcome=0.5),
            Engram(vector=[0.2] * STATE_VECTOR_SIZE, action=2, outcome=0.6),
            Engram(vector=[0.3] * STATE_VECTOR_SIZE, action=3, outcome=0.7),
        ]
        faiss_store.batch_insert(engrams)
        
        assert faiss_store.get_count() == 3
        assert faiss_store._insertion_counter == 3
    
    def test_nearest(self, faiss_store):
        """Test nearest neighbor search."""
        # Insert some test data - create distinct vectors of correct size
        vec1 = [1.0] + [0.0] * (STATE_VECTOR_SIZE - 1)
        vec2 = [0.0, 1.0] + [0.0] * (STATE_VECTOR_SIZE - 2)
        vec3 = [0.0, 0.0, 1.0] + [0.0] * (STATE_VECTOR_SIZE - 3)
        engram1 = Engram(vector=vec1, action=1, outcome=0.5)
        engram2 = Engram(vector=vec2, action=2, outcome=0.6)
        engram3 = Engram(vector=vec3, action=3, outcome=0.7)
        faiss_store.insert(engram1)
        faiss_store.insert(engram2)
        faiss_store.insert(engram3)
        
        # Search for nearest to first vector
        results = faiss_store.nearest(vec1, limit=2)
        
        assert len(results) == 2
        assert isinstance(results[0][0], Engram)
        assert results[0][0].action == 1  # Should be closest
        assert results[0][1] >= 0  # Distance should be non-negative
    
    def test_get_count(self, faiss_store):
        """Test getting count."""
        assert faiss_store.get_count() == 0
        
        engram = Engram(vector=[0.1] * STATE_VECTOR_SIZE, action=1, outcome=0.5)
        faiss_store.insert(engram)
        
        assert faiss_store.get_count() == 1
    
    def test_get_outcome_stats_empty(self, faiss_store):
        """Test outcome stats with empty store."""
        stats = faiss_store.get_outcome_stats()
        assert stats == {'positive_ratio': 0.0, 'negative_ratio': 0.0, 'mean_outcome': 0.0}
    
    def test_get_outcome_stats_with_data(self, faiss_store):
        """Test outcome stats calculation."""
        # Insert engrams with various outcomes
        engrams = [
            Engram(vector=[0.1] * STATE_VECTOR_SIZE, action=1, outcome=0.8),
            Engram(vector=[0.2] * STATE_VECTOR_SIZE, action=2, outcome=0.5),
            Engram(vector=[0.3] * STATE_VECTOR_SIZE, action=3, outcome=-0.3),
            Engram(vector=[0.4] * STATE_VECTOR_SIZE, action=1, outcome=0.2),
            Engram(vector=[0.5] * STATE_VECTOR_SIZE, action=2, outcome=-0.1),
        ]
        faiss_store.batch_insert(engrams)
        
        stats = faiss_store.get_outcome_stats(sample_size=1000)
        
        # Should have some positive and negative outcomes
        assert 'positive_ratio' in stats
        assert 'negative_ratio' in stats
        assert 'mean_outcome' in stats
        assert 0 <= stats['positive_ratio'] <= 1
        assert 0 <= stats['negative_ratio'] <= 1
    
    def test_delete_oldest_records(self, faiss_store):
        """Test deleting oldest records."""
        # Insert multiple engrams
        for i in range(5):
            engram = Engram(vector=[float(i)] * STATE_VECTOR_SIZE, action=i, outcome=0.5)
            faiss_store.insert(engram)
        
        assert faiss_store.get_count() == 5
        
        # Delete 2 oldest
        deleted = faiss_store.delete_oldest_records(2)
        
        assert deleted == 2
        assert faiss_store.get_count() == 3


class TestEngramStoreFactory:
    """Tests for factory function and backward compatibility."""
    
    @patch('engram.create_engram_store')
    def test_engram_store_delegates_to_factory(self, mock_factory):
        """Test that EngramStore class delegates to factory."""
        mock_store = MagicMock()
        mock_factory.return_value = mock_store
        
        store = EngramStore("test", reset=True)
        
        mock_factory.assert_called_once_with("test", True)
        assert store == mock_store

