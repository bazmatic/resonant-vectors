import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from settings import STATE_VECTOR_SIZE, OUTPUT_VECTOR_SIZE

class Engram:
    def __init__(
            self, 
            vector: list[float],
            action: int, 
            outcome: float,
        ):
        self.vector = vector
        self.action = action
        self.outcome = outcome

    @staticmethod
    def from_record(record: list):
        return Engram(
            vector=record.fields[EngramField.vector],
            action=record.fields[EngramField.action],
            outcome=record.fields[EngramField.outcome],
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
    vector = "vector"
    action = "action"
    outcome = "outcome"

class EngramStore:


    # static method
    @staticmethod
    def schema():
        fields = [
            FieldSchema(name=EngramField.id, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=EngramField.vector, dtype=DataType.FLOAT_VECTOR, dim=STATE_VECTOR_SIZE, description='The state embedding'),
            FieldSchema(name=EngramField.action, dtype=DataType.INT16, description='The action taken'),
            FieldSchema(name=EngramField.outcome, dtype=DataType.FLOAT, description="Negative means penalty, positive means reward")
        ]
        schema = CollectionSchema(fields=fields, description="Collection of states")
        return schema

    def __init__(self, name: str = "lander", reset: bool = False):
        self.collection_name = name  
        self.connect_to_db(reset)   
          
        #self.collection=Collection(name=self.collection_name)

    def connect_to_db(self, reset:bool = False):
        connections.connect(alias="default") 
        if reset and utility.has_collection(self.collection_name):
            print(f"Dropping collection {self.collection_name}")
            utility.drop_collection(self.collection_name)
        else:
            print(f"Using existing collection {self.collection_name}")
        self.make_collection()      

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

    def insert(self, record: Engram):       
        self.collection.insert([[record.vector], [record.action], [record.outcome]], 0.0001)

    
    def nearest(self, vector: list[float], limit: int) -> list[Engram, float]:
        self.collection.load()

        records = self.collection.search(
            data=[vector],
            limit=limit,
            param={
                "metric_type": "L2", # Euclidean distance
                "params": { 
                    "nprobe": 16 # 16 clusters to search
                }
            },
            anns_field=EngramField.vector,
            output_fields=["id", "vector", "action", "outcome"]
        )
        # Map results into Record types
        result: list[Engram, float] = []
        for record in records[0]:
            result.append((Engram.from_record(record), record.distance))
        return result

   
def test():
    store = EngramStore()
    akash = Engram(id=1, vector=[1.0, 2.0, 3.0], action=1, outcome=1.0)
    store.insert(akash)
    store.insert(Engram(id=2, vector=[-0.101, 0.001, 0.993], action=1, outcome=-1.0))
    store.insert(Engram(id=3, vector=[-0.2, 0.882, 0.303], action=2, outcome=0.5))
    store.insert(Engram(id=4, vector=[0.1, 0.45, 0.01], action=3, outcome=0.2))

    for record, distance in store.nearest(vector=[0.5, 0.5, 0.5]):
        print(record.action, record.outcome, distance)        
    




