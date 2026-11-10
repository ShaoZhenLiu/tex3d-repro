"""Read this study's trusted float32 torch archive without installing torch."""
import collections
import io
import pickle
import zipfile
import numpy as np


def load(path):
    archive = zipfile.ZipFile(path)
    prefix = next(n[:-8] for n in archive.namelist() if n.endswith('data.pkl'))
    storage = {}

    def rebuild(values, offset, shape, stride, requires_grad, hooks):
        return np.ndarray(tuple(shape), dtype=values.dtype, buffer=values,
                          offset=int(offset) * values.dtype.itemsize,
                          strides=tuple(int(s) * values.dtype.itemsize for s in stride))

    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ('collections', 'OrderedDict'):
                return collections.OrderedDict
            if (module, name) == ('torch', 'FloatStorage'):
                return np.dtype('<f4')
            if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
                return rebuild
            raise ValueError('Unsupported cache object: %s.%s' % (module, name))

        def persistent_load(self, pid):
            kind, dtype, key, device, count = pid
            assert kind == 'storage' and dtype == np.dtype('<f4')
            if key not in storage:
                storage[key] = np.frombuffer(archive.read(prefix + 'data/' + key), dtype=dtype)
                assert storage[key].size == count
            return storage[key]

    result = Reader(io.BytesIO(archive.read(prefix + 'data.pkl'))).load()
    archive.close()
    return result


if __name__ == '__main__':
    import sys
    data = load(sys.argv[1])
    print(type(data).__name__, len(data))
    def describe(obj, depth=0):
        if isinstance(obj, dict):
            print('  '*depth, 'keys', list(obj)[:15])
            if depth < 2:
                for key in list(obj)[:2]:
                    print('  '*depth, key)
                    describe(obj[key], depth+1)
        elif isinstance(obj, np.ndarray):
            print('  '*depth, obj.shape, obj.dtype)
        else:
            print('  '*depth, type(obj).__name__, str(obj)[:200])
    describe(data)
