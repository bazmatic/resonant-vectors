from pymilvus import connections, utility, Collection
from engram import EngramStore, EngramField

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Get the collection name (adjust if different)
collection_name = "lander3"  # Based on your training metrics files

# Check if collection exists
if not utility.has_collection(collection_name):
    print(f"Collection '{collection_name}' does not exist.")
    # Try default name
    collection_name = "lander"
    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' also does not exist.")
        exit(1)

# Load the collection
collection = Collection(collection_name)
collection.load()

num_entities = collection.num_entities
print(f"Total entities in collection '{collection_name}': {num_entities}")

if num_entities == 0:
    print("Collection is empty.")
else:
    # Query for min and max IDs
    try:
        # Query a sample of IDs to find min and max
        # Get first 1000 entities to sample IDs
        sample_results = collection.query(
            expr="",
            output_fields=[EngramField.id],
            limit=min(1000, num_entities)
        )
        
        if sample_results:
            ids = [result[EngramField.id] for result in sample_results]
            min_id = min(ids)
            max_id = max(ids)
            print(f"Query sample size: {len(ids)}")
            print(f"  Lowest ID: {min_id}")
            print(f"  Highest ID: {max_id}")
            
            # Query multiple random samples to get better coverage
            print("\nQuerying multiple samples for better coverage...")
            all_ids = set(ids)
            import numpy as np
            for i in range(5):
                dummy_vector = np.random.random(9).tolist()
                search_results = collection.search(
                    data=[dummy_vector],
                    limit=min(1000, num_entities),
                    param={"metric_type": "L2", "params": {"nprobe": 16}},
                    anns_field=EngramField.vector,
                    output_fields=[EngramField.id]
                )
                if search_results and len(search_results[0]) > 0:
                    search_ids = [hit.entity.get(EngramField.id) for hit in search_results[0]]
                    all_ids.update(search_ids)
            
            overall_min = min(all_ids)
            overall_max = max(all_ids)
            print(f"\nFrom {len(all_ids)} unique IDs sampled:")
            print(f"  Lowest ID: {overall_min}")
            print(f"  Highest ID: {overall_max}")
            print(f"  ID range: {overall_max - overall_min}")
            print(f"\nNote: With {num_entities} total entities,")
            print(f"the actual range may be larger than sampled.")
            print(f"Estimated ID span: ~{overall_max - overall_min} (from sample)")
        else:
            print("Could not query IDs.")
    except Exception as e:
        print(f"Query approach failed: {e}")
        import traceback
        traceback.print_exc()

