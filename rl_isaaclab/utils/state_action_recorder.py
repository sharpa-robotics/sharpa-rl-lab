import numpy as np
import h5py


class H5StateActionRecorder:
    def __init__(self):
        self.is_open = False

    def new(self, output: str):
        self.close()
        self.filepath = output
        self.h5f = h5py.File(self.filepath, "w")
        self.datasets_created: bool = False
        self.is_open = True

    def _create_datasets(self, input_dict) -> None:
        for name, value in input_dict.items():
            if isinstance(value, np.ndarray):
                self.h5f.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape), dtype=np.float64, chunks=True)
            else:
                self.h5f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.float64, chunks=True)
        self.datasets_created = True
    
    def append(self, input_dict) -> None:
        if not self.datasets_created:
            self._create_datasets(input_dict)
        cur_len = self.h5f[list(input_dict.keys())[0]].shape[0]
        for name, value in input_dict.items():
            if isinstance(value, np.ndarray):
                self.h5f[name].resize((cur_len+1, *value.shape))
            else:
                self.h5f[name].resize((cur_len+1,))
            self.h5f[name][cur_len] = np.asarray(value, dtype=np.float64)
        self.h5f.flush()

    def close(self) -> None:
        try:
            self.h5f.close()
            self.datasets_created = False
            self.is_open = False
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
