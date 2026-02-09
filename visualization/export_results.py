import requests
import numpy as np

def export_to_txt(input_txt, output_txt):
    # 1. Load the original points
    # We load 4096 to match the PointNet processing size
    try:
        data = np.loadtxt(input_txt, max_rows=4096)
        xyz = data[:, :3]
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Get predictions from your Docker API
    url = "http://localhost:8000/predict"
    print(f"📡 Sending data to API...")
    
    try:
        with open(input_txt, 'rb') as f:
            r = requests.post(url, files={'file': f})
        
        if r.status_code != 200:
            print(f"❌ API Error: {r.text}")
            return
            
        predictions = r.json()['predictions']
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    # 3. Write to TXT file
    print(f"💾 Exporting to {output_txt}...")
    with open(output_txt, 'w') as f:
        # Header for clarity
        f.write("# X Y Z Predicted_Label\n")
        
        # Use zip to stay safe from IndexError
        count = 0
        for point, label in zip(xyz, predictions):
            # Format: 3 floats and 1 string
            f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f} {label}\n")
            count += 1

    print(f"✅ Success! Exported {count} labeled points.")

if __name__ == "__main__":
    # Ensure this path is correct relative to where you run the script
    export_to_txt("data/Vaihingen3D/test.txt", "classification_results.txt")