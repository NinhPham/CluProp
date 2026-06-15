import numpy as np

def txt_to_bin(txt_path, bin_path, dtype=np.float32):
    # data = np.loadtxt(txt_path, dtype=dtype, delimiter=",")
    data = np.loadtxt(txt_path, dtype=dtype)
    data.astype(dtype).tofile(bin_path)

def npy_to_bin(txt_path, bin_path, dtype=np.float32):
    data = np.load(txt_path)
    data.astype(dtype).tofile(bin_path)

def read_bin(bin_path, num_rows, num_cols, dtype=np.float32):
    return np.fromfile(bin_path, dtype=dtype).reshape((num_rows, num_cols))

def mmap_bin(bin_path, num_rows, num_cols, dtype=np.float32):
    return np.memmap(bin_path, dtype=dtype, mode='r', shape=(num_rows, num_cols))

if __name__ == '__main__':

    path = "/shared/Dataset/Clustering/"

    txt_file = path + "mnist_X_60K_780"    # Assume this file already exists
    bin_file = path + "mnist_X_60K_780.bin"
    num_rows = 60000
    num_cols = 780

    # txt_file = path + "covtype_X"  # Assume this file already exists
    # bin_file = path + "covtype_X.bin"
    # num_rows = 581012
    # num_cols = 54

    # txt_file = path + "kddCup_X_6class"  # Assume this file already exists
    # bin_file = path + "kddCup_X_6class.bin"
    # num_rows = 4891470
    # num_cols = 38

    # txt_file = path + "pamap2_X_no_0"  # Assume this file already exists
    # bin_file = path + "pamap2_X_no_0.bin"
    # num_rows = 1770131
    # num_cols = 51

    # txt_file = path + "tinyimagenet100_resnet50_embs.npy"  # Assume this file already exists
    # bin_file = path + "tinyimagenet100_resnet50_embs.bin"
    # num_rows = 98179
    # num_cols = 2048

    # txt_file = path + "tinyimagenet100_resnet10_embs.npy"  # Assume this file already exists
    # bin_file = path + "tinyimagenet100_resnet10_embs.bin"
    # num_rows = 98179
    # num_cols = 512

    # txt_file = path + "svhn_extra_resnet50_embs.npy"  # Assume this file already exists
    # bin_file = path + "svhn_extra_resnet50_embs.bin"
    # num_rows = 531131
    # num_cols = 2048

    # txt_file = path + "svhn_extra_resnet10_embs.npy"  # Assume this file already exists
    # bin_file = path + "svhn_extra_resnet10_embs.bin"
    # num_rows = 531131
    # num_cols = 512

    # path1 = "/home/npha145/Uni of Auckland Dropbox/soptics-dbhd/sVDC/"
    # txt_file = path1 + "L2_MinPts_32"  # Assume this file already exists
    # bin_file = path + "mnist_all_X_kNN_L2_32.bin"
    # num_rows = 70000
    # num_cols = 32

    # Convert .txt to .bin
    txt_to_bin(txt_file, bin_file)

    # Convert .npy to .bin
    # npy_to_bin(txt_file, bin_file)

    # ==== STEP 2   : Load the binary file fully into memory ====

    data = read_bin(bin_file, num_rows, num_cols)
    print("Loaded full binary into memory")
    print(data.shape)        # Should print: (8100000, 784)
    print(data[0, :10])      # First 10 values of the first row

    # ==== STEP 3: Use memory-mapped version ====

    data_mmap = mmap_bin(bin_file, num_rows, num_cols)
    print("Memory-mapped binary")
    print(data_mmap.shape)   # Same shape
    print(data_mmap[0, :10])  # Access 124th row, first 10 values
