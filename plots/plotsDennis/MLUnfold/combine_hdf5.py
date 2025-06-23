import h5py
import numpy as np

def merge_h5_files(input_files, output_file):
    # Open output file in write mode
    with h5py.File(output_file, 'w') as f_out:
        datasets_initialized = {}

        for i, file in enumerate(input_files):
            with h5py.File(file, 'r') as f_in:
                for key in f_in.keys():
                    data = f_in[key][...]

                    if key not in datasets_initialized:
                        # Create extendable dataset in output file
                        maxshape = (None,) + data.shape[1:]  # make it extendable in the first dimension
                        f_out.create_dataset(
                            key,
                            data=data,
                            maxshape=maxshape,
                            chunks=True,
                            compression='gzip'
                        )
                        datasets_initialized[key] = data.shape[0]
                    else:
                        old_size = datasets_initialized[key]
                        new_size = old_size + data.shape[0]
                        f_out[key].resize((new_size,) + data.shape[1:])
                        f_out[key][old_size:new_size] = data
                        datasets_initialized[key] = new_size

    print(f"Merged {len(input_files)} files into: {output_file}")


train_sample_list = [f'/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/TTToSemiLeptonic_{i}.h5' for i in range(0,60) ]
validation_sample_list = [f'/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/TTToSemiLeptonic_{i}.h5' for i in range(60,80) ]
pseudodata_sample_list = [f'/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/TTToSemiLeptonic_{i}.h5' for i in range(80,100) ]

print("Making validation file...")
merge_h5_files(validation_sample_list, "/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/TTToSemiLeptonic_val.h5")
print("Making pseudo data file...")
merge_h5_files(pseudodata_sample_list, "/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/TTToSemiLeptonic_pseudo.h5")
print("Making training file...")
merge_h5_files(train_sample_list, "/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/TTToSemiLeptonic_train.h5")
