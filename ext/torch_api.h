// Handwritten C shim over libtorch (C++ API), ported from tch-rs
// torch-sys/libtch/torch_api.h. C++ objects cross the boundary as opaque
// pointers. Exceptions are caught by PROTECT and stored in a thread-local
// buffer, retrievable via get_and_reset_last_err.
#ifndef TORCHCR_API_H
#define TORCHCR_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include <torch/torch.h>
#include <torch/script.h>
#include <stdexcept>

extern thread_local char *torch_last_err;

extern "C" {
typedef torch::Tensor *tensor;
typedef torch::Scalar *scalar;
typedef torch::jit::script::Module *module;
typedef torch::jit::IValue *ivalue;

#define PROTECT(x)                                                             \
  try {                                                                        \
    x                                                                          \
  } catch (const std::exception &e) {                                          \
    torch_last_err = strdup(e.what());                                         \
  }
#else
typedef void *tensor;
typedef void *scalar;
typedef void *module;
typedef void *ivalue;
#endif

/* ---- errors ---- */
char *get_and_reset_last_err();

/* ---- misc ---- */
void at_manual_seed(int64_t seed);
int at_get_num_threads();
void at_set_num_threads(int n_threads);

/* ---- tensor lifecycle ---- */
tensor at_new_tensor();
tensor at_tensor_of_blob(void *data, int64_t *dims, size_t ndims,
                         int64_t *strides, size_t nstrides, int type,
                         int device);
tensor at_tensor_of_data(void *vs, int64_t *dims, size_t ndims,
                         size_t element_size_in_bytes, int type);
void at_copy_data(tensor t, void *vs, size_t numel, size_t elt_size_in_bytes);
tensor at_shallow_clone(tensor t);
void at_free(tensor t);

/* ---- tensor introspection ---- */
void *at_data_ptr(tensor t);
int at_defined(tensor t);
int at_is_sparse(tensor t);
int at_is_contiguous(tensor t);
int at_device(tensor t);
size_t at_dim(tensor t);
void at_shape(tensor t, int64_t *dims);
void at_stride(tensor t, int64_t *dims);
int at_scalar_type(tensor t);

/* ---- element access ---- */
tensor at_get(tensor t, int index);
void at_fill_double(tensor t, double v);
void at_fill_int64(tensor t, int64_t v);
double at_double_value_at_indexes(tensor t, int64_t *indexes, int indexes_len);
int64_t at_int64_value_at_indexes(tensor t, int64_t *indexes,
                                  int indexes_len);
void at_set_double_value_at_indexes(tensor t, int *indexes, int indexes_len,
                                    double v);
void at_set_int64_value_at_indexes(tensor t, int *indexes, int indexes_len,
                                   int64_t v);
void at_copy_(tensor dst, tensor src);

/* ---- printing / serialization ---- */
void at_print(tensor t);
char *at_to_string(tensor t, int line_size);
void at_save(tensor t, char *filename);
tensor at_load(char *filename);

/* ---- autograd ---- */
void at_backward(tensor t, int keep_graph, int create_graph);
int at_requires_grad(tensor t);
int at_grad_set_enabled(int b);

/* ---- device transfer ---- */
tensor at_to_device(tensor t, int device);
tensor at_to_type(tensor t, int type);

/* ---- device / context ---- */
int atc_cuda_device_count();
int atc_cuda_is_available();
int atc_cudnn_is_available();
void atc_manual_seed(uint64_t seed);
void atc_synchronize(int64_t device_index);

/* ---- scalar ---- */
scalar ats_int(int64_t v);
scalar ats_float(double v);
int64_t ats_to_int(scalar s);
double ats_to_float(scalar s);
char *ats_to_string(scalar s);
void ats_free(scalar s);

/* ---- torchscript module ---- */
module atm_load(char *filename);
module atm_load_on_device(char *filename, int device);
tensor atm_forward(module m, tensor *tensors, int ntensors);
ivalue atm_forward_(module m, ivalue *ivalues, int nivalues);
ivalue atm_method_(module m, char *method_name, ivalue *ivalues,
                   int nivalues);
void atm_eval(module m);
void atm_train(module m);
void atm_free(module m);
void atm_to(module m, int device, int dtype, int non_blocking);
void atm_save(module m, char *filename);
void atm_named_parameters(module m, void *data,
                          void (*f)(void *, char *, tensor));

/* ---- ivalue boxing ---- */
ivalue ati_none();
ivalue ati_tensor(tensor t);
ivalue ati_int(int64_t i);
ivalue ati_double(double d);
ivalue ati_bool(int i);
ivalue ati_string(char *s);
ivalue ati_tuple(ivalue *is, int nvalues);
ivalue ati_generic_list(ivalue *is, int nvalues);
ivalue ati_generic_dict(ivalue *is, int nvalues);
ivalue ati_int_list(int64_t *is, int nvalues);
ivalue ati_double_list(double *is, int nvalues);
ivalue ati_bool_list(char *is, int nvalues);
ivalue ati_string_list(char **is, int nvalues);
ivalue ati_tensor_list(tensor *is, int nvalues);
ivalue ati_device(int device_idx);
int ati_tag(ivalue i);
tensor ati_to_tensor(ivalue i);
int64_t ati_to_int(ivalue i);
double ati_to_double(ivalue i);
int ati_to_bool(ivalue i);
char *ati_to_string(ivalue i);
int ati_length(ivalue i);
int ati_tuple_length(ivalue i);
void ati_to_tuple(ivalue i, ivalue *outputs, int noutputs);
void ati_to_generic_list(ivalue i, ivalue *outputs, int noutputs);
void ati_to_generic_dict(ivalue i, ivalue *outputs, int noutputs);
void ati_to_int_list(ivalue i, int64_t *outputs, int noutputs);
void ati_to_double_list(ivalue i, double *outputs, int noutputs);
void ati_to_bool_list(ivalue i, char *outputs, int noutputs);
void ati_to_tensor_list(ivalue i, tensor *outputs, int noutputs);
ivalue ati_object_method_(ivalue i, char *method_name, ivalue *ivalues,
                          int nivalues);
ivalue ati_object_getattr_(ivalue i, char *attr_name);
ivalue ati_clone(ivalue i);
void ati_free(ivalue i);

#ifdef __cplusplus
} /* extern "C" */

std::vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len);
at::Device device_of_int(int d);
#endif

#endif /* TORCHCR_API_H */
