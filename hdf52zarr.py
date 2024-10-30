import os
import h5py
import zarr

# Specify the root directory containing HDF5 files
hdf5_root_dir = '/home/yixin/Downloads/robomimic/datasets'

# Recursively find and convert HDF5 files
def convert_hdf5_to_zarr(hdf5_file, zarr_file):
    """Convert HDF5 file to Zarr format"""
    zarr_root = zarr.open_group(zarr_file, mode='w')

    def hdf5_to_zarr(h5_group, zarr_group):
        """Recursively convert HDF5 groups and datasets"""
        for key, item in h5_group.items():
            if isinstance(item, h5py.Dataset):
                # If it's a dataset, copy it to Zarr
                zarr_group.create_dataset(key, data=item[...])
            elif isinstance(item, h5py.Group):
                # If it's a group, process it recursively
                zarr_subgroup = zarr_group.create_group(key)
                hdf5_to_zarr(item, zarr_subgroup)

    # Open the HDF5 file and start conversion
    with h5py.File(hdf5_file, 'r') as f:
        hdf5_to_zarr(f, zarr_root)

def traverse_and_convert(root_dir):
    """Recursively traverse directories to find and convert HDF5 files"""
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".h5") or file.endswith(".hdf5"):
                hdf5_file = os.path.join(root, file)
                zarr_file = os.path.splitext(hdf5_file)[0] + ".zarr"
                
                print(f"Converting {hdf5_file} to {zarr_file} ...")
                convert_hdf5_to_zarr(hdf5_file, zarr_file)
                print(f"Conversion completed for {hdf5_file}.")

# Start conversion
traverse_and_convert(hdf5_root_dir)