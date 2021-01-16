from torch.utils.cpp_extension import load

fwht_cpp = load(name="fwht_cpp", sources=["fwht.cpp"], verbose=True)
help(fwht_cpp)
