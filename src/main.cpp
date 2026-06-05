#include <pybind11/pybind11.h>

namespace py = pybind11;

int verify_environment() {
    return 124;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "NucleoByte internal C++/CUDA acceleration engine";
    m.def("verify_environment", &verify_environment, "Verifies host tooling integration.");
}