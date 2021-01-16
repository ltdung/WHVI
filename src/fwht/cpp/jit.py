from torch.utils.cpp_extension import load

fwht_cpp = load(name="fwhtm_cpp", sources=["fwhtm.cpp"], verbose=True)
help(fwht_cpp)
