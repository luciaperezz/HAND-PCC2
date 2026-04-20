import scipy.io
mat = scipy.io.loadmat('/Users/nayasarianasr/Desktop/HAND-PCC2/data/stroke/HS01.mat')
s = mat['s'][0, 0]
print("HS fields:", s.dtype.names)
for field in s.dtype.names:
    val = s[field]
    print(f"  {field}: shape={getattr(val, 'shape', 'N/A')}")