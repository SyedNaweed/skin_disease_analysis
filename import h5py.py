import h5py

filename = "mymodel2.h5"

h5 = h5py.File(filename, 'r')

# Check the available dataset names in the HDF5 file
print("Available datasets:", list(h5.keys()))
# Access the correct dataset
if "futures_data" in h5:
    futures_data = h5["futures_data"]
    print(futures_data)
else:
    print("Dataset 'futures_data' not found in the HDF5 file.")

q

h5.close()
