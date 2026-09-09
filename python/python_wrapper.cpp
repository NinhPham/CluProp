#include <cluprop.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace python {

namespace py = pybind11;

PYBIND11_MODULE(cluprop, m) { // Must be the same name with class Dbscan
    py::class_<cluprop>(m, "cluprop")

        //.def(py::init<const int&, const int&>(),  py::arg("n_points"), py::arg("n_features"))
        .def(py::init<>())

        .def_readonly("labels_", &cluprop::labels) // must be def_readonly

        .def_property(
        "min_cluster_size",
        [](const cluprop& self) {
            return self.min_cluster_size;
        },
        [](cluprop& self, int value) {
            self.set_min_cluster_size(value);
        },
        "Minimum initial cluster size.")

        .def_property(
            "n_threads",
            [](const cluprop& self) {
                return self.n_threads;
            },
            [](cluprop& self, int value) {
                self.set_threads(value);
            },
        "Number of OpenMP threads used by CluProp.")

        // .def("set_propagation_cutoff", &cluprop::set_propagation_cutoff, py::arg("propagation_cutoff")=true)

        .def("clear", &cluprop::clear)

        // DANE from pre-computed kNN graph
        .def("knn_dane", &cluprop::knn_dane, py::arg("indices"), py::arg("distances"), py::arg("k"), py::arg("k_expand")=-1)
        ;

} // namespace cluprop
} // namespace python
