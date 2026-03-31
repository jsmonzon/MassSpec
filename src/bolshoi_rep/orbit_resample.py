import sys
import concurrent.futures
import os

location = "local"

if location == "server":
    parentdir = "/home/jsm99/SatGen/src/"
    master_path = "/netb/vdbosch/jsm99/data/bolshoi_rep/"
    save_path = "/netb/vdbosch/jsm99/data/bolshoi_rep/bound/"

elif location == "local":
    parentdir = "/Users/jsmonzon/Research/SatGen/src/"
    master_path = "/Users/jsmonzon/Research/MassSpec/data/local_trees/test/"
    save_path = "/Users/jsmonzon/Research/MassSpec/data/local_trees/test/bound/"


sys.path.insert(0, parentdir)
import orbit

files = [os.path.join(master_path, filename) for filename in os.listdir(master_path)
            if filename.startswith('tree') and not filename.endswith('evo.npz')]

def process_tree(file):
    base_dir, filename = os.path.split(file) 
    orbit.resample_orbit(file, save_path + filename)

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(process_tree, files)  # Skip master tree (tree_0.npz)
