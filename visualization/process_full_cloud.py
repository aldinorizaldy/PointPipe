import requests
import numpy as np
import io

def process_full_file(input_path, output_path, chunk_size=1024):
    print(f"🚀 Starting full processing of {input_path}...")
    
    # 1. Load the whole file
    data = np.loadtxt(input_path)
    xyz = data[:, :3]
    total_points = len(xyz)
    print(f"📊 Total points detected: {total_points}")

    all_predictions = []
    url = "http://localhost:8000/predict"

    # 2. Loop through the file
    # Ensure range goes from 0 to total_points in steps of chunk_size
    for i in range(0, total_points, chunk_size):
        chunk = xyz[i : i + chunk_size]
        actual_chunk_len = len(chunk)
        
        # PointNet usually needs exactly 1024 or 4096. 
        # If your model expects 1024, use that.
        # We pad the last chunk if it's too small.
        if actual_chunk_len < chunk_size:
            padding = np.tile(chunk[0], (chunk_size - actual_chunk_len, 1))
            chunk_to_send = np.vstack([chunk, padding])
        else:
            chunk_to_send = chunk

        # Send to API
        buf = io.BytesIO()
        np.savetxt(buf, chunk_to_send, fmt='%.3f')
        buf.seek(0)

        try:
            r = requests.post(url, files={'file': ('chunk.txt', buf, 'text/plain')})
            if r.status_code == 200:
                labels = r.json()['predictions']
                # ONLY take the labels for the points we actually sent
                # This prevents adding the "padding" labels to our list
                all_predictions.extend(labels[:actual_chunk_len])
            else:
                print(f"\n❌ API Error at index {i}: {r.text}")
                break
        except Exception as e:
            print(f"\n❌ Connection error at index {i}: {e}")
            break
        
        # Update progress on one line
        print(f"✅ Progress: {len(all_predictions)} / {total_points} (Chunk starting at {i})", end='\r')

    # Create a reverse map to turn names back into IDs
    name_to_id = {
        "Powerline": 1, "Low Veg": 2, "Impervious": 3, 
        "Car": 4, "Fence": 5, "Roof": 6, 
        "Facade": 7, "Shrub": 8, "Tree": 9, "Unknown": 0
}
    
    # 3. Final Save
    print(f"\n💾 Saving {len(all_predictions)} results to {output_path}...")
    with open(output_path, 'w') as f:
        for j in range(len(all_predictions)):
            label_name = all_predictions[j]
            label_id = name_to_id.get(label_name, 0)
            # Format: X Y Z ID
            f.write(f"{xyz[j,0]:.3f} {xyz[j,1]:.3f} {xyz[j,2]:.3f} {label_id}\n")
            
    print("✨ Finished!")

if __name__ == "__main__":
    # Match this chunk_size to what your PointNet model expects!
    process_full_file("data/Vaihingen3D/test.txt", "full_classified_cloud.txt", chunk_size=1024)