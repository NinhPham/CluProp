#include <cluprop.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <sstream>

namespace python {

namespace py = pybind11;

PYBIND11_MODULE(cluprop, m) { // Must be the same name with class Dbscan
    py::class_<cluprop>(m, "cluprop")
        .def(py::init<const int&, const int&>(),  py::arg("n_points"), py::arg("n_features"))
        
        .def_readonly("labels_", &cluprop::labels) // must be def_readonly
//        .def_readonly("indices_", &cluprop::indices_) // must be def_readonly
//        .def_readonly("distances_", &cluprop::distances_) // must be def_readonly

//        .def_readonly("flat_indices_", &cluprop::flat_indices_) // must be def_readonly
//        .def_readonly("flat_distances_", &cluprop::flat_distances_) // must be def_readonly
//        .def_readonly("flat_offset_", &cluprop::flat_offset_) // must be def_readonly

//    .def_readonly("matrix_flat_indices_", &cluprop::matrix_flat_indices_,
//         py::return_value_policy::reference_internal) // must be def_readonly
//    .def_readonly("matrix_flat_distances_", &cluprop::matrix_flat_distances_,
//       py::return_value_policy::reference_internal) // Risk of copy!
//        .def("matrix_flat_indices_", &cluprop::get_indices,
//           py::return_value_policy::reference_internal)  // ← NO COPY
//        .def("matrix_flat_distances_", &cluprop::get_distances,
//           py::return_value_policy::reference_internal)  // ← NO COPY


        .def_readonly("n_clusters_", &cluprop::n_clusters) // must be def_readonly
        .def_readwrite("min_cluster_size", &cluprop::min_cluster_size, "Change minimum of initialized cluster size.")
        .def_readwrite("propagation_cutoff", &cluprop::propagation_cutoff, "Change neighbor_cutoff flag to reduce the time/space complexity.")

        .def("set_min_cluster_size", &cluprop::set_min_cluster_size, py::arg("min_cluster_size"))
        .def("set_propagation_cutoff", &cluprop::set_propagation_cutoff, py::arg("propagation_cutoff")=true)

        .def("set_threads", &cluprop::set_threads, py::arg("n_threads"), "Change number of threads.")
        .def("clear", &cluprop::clear)

        // DANE from pre-computed kNN graph
        .def("knn_dane", &cluprop::knn_dane, py::arg("indices"), py::arg("distances"), py::arg("k"), py::arg("k_expand")=-1)


        ;

} // namespace cluprop
} // namespace python
