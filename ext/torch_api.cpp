// Implementation of the handwritten C shim over libtorch.
// Ported from tch-rs torch-sys/libtch/torch_api.cpp (inference-oriented
// subset: tensor lifecycle/data IO, autograd basics, TorchScript, IValues).
#include "torch_api.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

thread_local char *torch_last_err = nullptr;

char *get_and_reset_last_err() {
  char *tmp = torch_last_err;
  torch_last_err = nullptr;
  return tmp;
}

void at_manual_seed(int64_t seed) { torch::manual_seed(seed); }

int at_get_num_threads() {
  PROTECT(return at::get_num_threads();)
  return -1;
}

void at_set_num_threads(int n_threads) { PROTECT(at::set_num_threads(n_threads);) }

std::vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  std::vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i)
    result.push_back(*(vs[i]));
  return result;
}

at::Device device_of_int(int d) {
  if (d == -3)
    return at::Device(at::kVulkan);
  if (d == -2)
    return at::Device(at::kMPS);
  if (d < 0)
    return at::Device(at::kCPU);
  return at::Device(at::kCUDA, /*index=*/d);
}

tensor at_new_tensor() {
  PROTECT(return new torch::Tensor();)
  return nullptr;
}

tensor at_tensor_of_blob(void *data, int64_t *dims, size_t ndims,
                         int64_t *strides, size_t nstrides, int type,
                         int device) {
  PROTECT(
      at::TensorOptions blobOptions =
          at::TensorOptions()
              .device(device_of_int(device))
              .dtype(torch::ScalarType(type));
      if (nstrides == 0) {
        return new torch::Tensor(
            torch::for_blob(data, torch::IntArrayRef(dims, ndims))
                .options(blobOptions)
                .make_tensor());
      } else {
        return new torch::Tensor(torch::from_blob(
            data, torch::IntArrayRef(dims, ndims),
            torch::IntArrayRef(strides, nstrides), blobOptions));
      })
  return nullptr;
}

tensor at_tensor_of_data(void *vs, int64_t *dims, size_t ndims,
                         size_t element_size_in_bytes, int type) {
  PROTECT(
      torch::Tensor tensor =
          torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
      if ((int64_t)element_size_in_bytes != tensor.element_size()) throw std::
          invalid_argument("incoherent element sizes in bytes");
      void *tensor_data = tensor.data_ptr();
      memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
      return new torch::Tensor(tensor);)
  return nullptr;
}

void at_copy_data(tensor t, void *vs, size_t numel, size_t elt_size_in_bytes) {
  PROTECT(
      if ((int64_t)elt_size_in_bytes != t->element_size()) throw std::
          invalid_argument("incoherent element sizes in bytes");
      if ((int64_t)numel > t->numel()) throw std::invalid_argument(
          "target numel is larger than tensor numel");
      if (t->device().type() != at::kCPU) {
        torch::Tensor tmp_tensor = t->to(at::kCPU).contiguous();
        void *tensor_data = tmp_tensor.data_ptr();
        memcpy(vs, tensor_data, numel * elt_size_in_bytes);
      } else {
        auto tmp_tensor = t->contiguous();
        void *tensor_data = tmp_tensor.data_ptr();
        memcpy(vs, tensor_data, numel * elt_size_in_bytes);
      })
}

tensor at_shallow_clone(tensor t) {
  PROTECT(return new torch::Tensor(*t);)
  return nullptr;
}

void at_free(tensor t) { delete (t); }

void *at_data_ptr(tensor t) {
  PROTECT(return t->data_ptr();)
  return nullptr;
}

int at_defined(tensor t) {
  PROTECT(return t->defined();)
  return -1;
}

int at_is_sparse(tensor t) {
  PROTECT(return t->is_sparse();)
  return -1;
}

int at_is_contiguous(tensor t) {
  PROTECT(return t->is_contiguous();)
  return -1;
}

int at_device(tensor t) {
  PROTECT(auto device = t->device(); if (device.type() == at::kCPU) return -1;
          if (device.type() == at::kMPS) return -2;
          if (device.type() == at::kVulkan) return -3;
          if (device.type() == at::kCUDA) return device.index();)
  return -2;
}

size_t at_dim(tensor t) {
  PROTECT(return t->dim();)
  return -1;
}

void at_shape(tensor t, int64_t *dims) {
  PROTECT(int i = 0; for (int64_t dim : t->sizes()) dims[i++] = dim;)
}

void at_stride(tensor t, int64_t *dims) {
  PROTECT(int i = 0; for (int64_t dim : t->strides()) dims[i++] = dim;)
}

int at_scalar_type(tensor t) {
  PROTECT(return static_cast<int>(t->scalar_type());)
  return -1;
}

tensor at_get(tensor t, int index) {
  PROTECT(return new torch::Tensor((*t)[index]);)
  return nullptr;
}

template <typename T>
T at_value_at_indexes(tensor t, int64_t *indexes, int indexes_len) {
  PROTECT(torch::Tensor tensor = *t; for (int i = 0; i < indexes_len; ++i) {
    tensor = tensor[indexes[i]];
  } return tensor.item<T>();)
  return T();
}

double at_double_value_at_indexes(tensor t, int64_t *indexes,
                                  int indexes_len) {
  return at_value_at_indexes<double>(t, indexes, indexes_len);
}

int64_t at_int64_value_at_indexes(tensor t, int64_t *indexes,
                                  int indexes_len) {
  return at_value_at_indexes<int64_t>(t, indexes, indexes_len);
}

template <typename T>
void at_set_value_at_indexes(tensor t, int *indexes, int indexes_len, T v) {
  PROTECT(torch::Tensor tensor = *t; for (int i = 0; i < indexes_len; ++i) {
    tensor = tensor[indexes[i]];
  } tensor.fill_(v);)
}

void at_set_double_value_at_indexes(tensor t, int *indexes, int indexes_len,
                                    double v) {
  at_set_value_at_indexes<double>(t, indexes, indexes_len, v);
}

void at_set_int64_value_at_indexes(tensor t, int *indexes, int indexes_len,
                                   int64_t v) {
  at_set_value_at_indexes<int64_t>(t, indexes, indexes_len, v);
}

void at_fill_double(tensor t, double v) { PROTECT(t->fill_(v);) }

void at_fill_int64(tensor t, int64_t v) { PROTECT(t->fill_(v);) }

void at_copy_(tensor dst, tensor src) { PROTECT(dst->copy_(*src);) }

void at_print(tensor t) {
  PROTECT(std::cout << *t << std::endl;)
}

char *at_to_string(tensor t, int line_size) {
  PROTECT(std::ostringstream oss; torch::print(oss, *t, line_size);
          return strdup(oss.str().c_str());)
  return nullptr;
}

void at_save(tensor t, char *filename) { PROTECT(torch::save(*t, filename);) }

tensor at_load(char *filename) {
  PROTECT(torch::Tensor tensor; torch::load(tensor, filename);
          return new torch::Tensor(tensor);)
  return nullptr;
}

void at_backward(tensor t, int keep_graph, int create_graph) {
  PROTECT(t->backward({}, keep_graph, create_graph);)
}

int at_requires_grad(tensor t) {
  PROTECT(return t->requires_grad();)
  return -1;
}

int at_grad_set_enabled(int b) {
  PROTECT(bool is_enabled = torch::autograd::GradMode::is_enabled();
          torch::autograd::GradMode::set_enabled(b); return is_enabled;)
  return -1;
}

tensor at_to_device(tensor t, int device) {
  PROTECT(return new torch::Tensor(t->to(device_of_int(device)));)
  return nullptr;
}

tensor at_to_type(tensor t, int type) {
  PROTECT(return new torch::Tensor(t->to(at::ScalarType(type)));)
  return nullptr;
}

int atc_cuda_device_count() {
  PROTECT(return torch::cuda::device_count();)
  return -1;
}

int atc_cuda_is_available() {
  PROTECT(return torch::cuda::is_available();)
  return -1;
}

int atc_cudnn_is_available() {
  PROTECT(return torch::cuda::cudnn_is_available();)
  return -1;
}

void atc_manual_seed(uint64_t seed) {
  PROTECT(return torch::cuda::manual_seed(seed);)
}

void atc_synchronize(int64_t device_index) {
  PROTECT(return torch::cuda::synchronize(device_index);)
}

scalar ats_int(int64_t v) {
  PROTECT(return new torch::Scalar(v);)
  return nullptr;
}

scalar ats_float(double v) {
  PROTECT(return new torch::Scalar(v);)
  return nullptr;
}

int64_t ats_to_int(scalar s) {
  PROTECT(return s->toLong();)
  return -1;
}

double ats_to_float(scalar s) {
  PROTECT(return s->toDouble();)
  return 0.;
}

char *ats_to_string(scalar s) {
  PROTECT(std::ostringstream oss; oss << (*s);
          return strdup(oss.str().c_str());)
  return nullptr;
}

void ats_free(scalar s) { delete (s); }

module atm_load(char *filename) {
  PROTECT(return new torch::jit::script::Module(torch::jit::load(filename));)
  return nullptr;
}

module atm_load_on_device(char *filename, int device) {
  PROTECT(return new torch::jit::script::Module(
                     torch::jit::load(filename, device_of_int(device)));)
  return nullptr;
}

tensor atm_forward(module m, tensor *tensors, int ntensors) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          for (int i = 0; i < ntensors; ++i) inputs.push_back(*(tensors[i]));
          torch::jit::IValue output = m->forward(std::move(inputs));
          if (!output.isTensor()) throw std::invalid_argument(
              "forward did not return a tensor");
          return new torch::Tensor(output.toTensor());)
  return nullptr;
}

ivalue atm_forward_(module m, ivalue *ivalues, int nivalues) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          for (int i = 0; i < nivalues; ++i) inputs.push_back(*(ivalues[i]));
          torch::jit::IValue output = m->forward(std::move(inputs));
          return new torch::jit::IValue(output);)
  return nullptr;
}

ivalue atm_method_(module m, char *method_name, ivalue *ivalues,
                   int nivalues) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          for (int i = 0; i < nivalues; ++i) inputs.push_back(*(ivalues[i]));
          torch::jit::IValue output =
              m->get_method(method_name)(std::move(inputs));
          return new torch::jit::IValue(output);)
  return nullptr;
}

void atm_eval(module m) { PROTECT(m->eval();) }

void atm_train(module m) { PROTECT(m->train();) }

void atm_free(module m) { delete (m); }

void atm_to(module m, int device, int dtype, int non_blocking) {
  PROTECT(m->to(device_of_int(device), at::ScalarType(dtype), non_blocking);)
}

void atm_save(module m, char *filename) { PROTECT(m->save(filename);) }

void atm_named_parameters(module m, void *data,
                          void (*f)(void *, char *, tensor)) {
  PROTECT(for (const auto &p : m->named_parameters()) {
    auto v = p.value;
    f(data, (char *)p.name.c_str(), new torch::Tensor(v));
  })
}

ivalue ati_none() {
  PROTECT(return new torch::jit::IValue();)
  return nullptr;
}

ivalue ati_tensor(tensor t) {
  PROTECT(return new torch::jit::IValue(*t);)
  return nullptr;
}

ivalue ati_int(int64_t i) {
  PROTECT(return new torch::jit::IValue(i);)
  return nullptr;
}

ivalue ati_double(double d) {
  PROTECT(return new torch::jit::IValue(d);)
  return nullptr;
}

ivalue ati_bool(int i) {
  PROTECT(return new torch::jit::IValue((bool)i);)
  return nullptr;
}

ivalue ati_string(char *s) {
  PROTECT(std::string str(s); return new torch::jit::IValue(str);)
  return nullptr;
}

ivalue ati_tuple(ivalue *is, int nvalues) {
  PROTECT(std::vector<torch::jit::IValue> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
          return new torch::jit::IValue(torch::ivalue::Tuple::create(vec));)
  return nullptr;
}

ivalue ati_generic_list(ivalue *is, int nvalues) {
  PROTECT(c10::List<torch::jit::IValue> vec(c10::AnyType::get());
          for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
          return new torch::jit::IValue(c10::List<torch::jit::IValue>(vec));)
  return nullptr;
}

using generic_dict = c10::Dict<torch::jit::IValue, torch::jit::IValue>;

ivalue ati_generic_dict(ivalue *is, int nvalues) {
  PROTECT(
      bool all_keys_are_str = true; for (int i = 0; i < nvalues; ++i) {
        if (!is[2 * i]->isString())
          all_keys_are_str = false;
      } bool all_values_are_tensor = true;
      for (int i = 0; i < nvalues; ++i) {
        if (!is[2 * i + 1]->isTensor())
          all_values_are_tensor = false;
      } if (all_keys_are_str && all_values_are_tensor) {
        generic_dict dict(c10::StringType::get(), c10::TensorType::get());
        for (int i = 0; i < nvalues; ++i)
          dict.insert(is[2 * i]->toStringRef(), is[2 * i + 1]->toTensor());
        return new torch::jit::IValue(dict);
      } else if (all_keys_are_str) {
        generic_dict dict(c10::StringType::get(), c10::AnyType::get());
        for (int i = 0; i < nvalues; ++i)
          dict.insert(is[2 * i]->toStringRef(), *(is[2 * i + 1]));
        return new torch::jit::IValue(dict);
      } else {
        generic_dict dict(c10::AnyType::get(), c10::AnyType::get());
        for (int i = 0; i < nvalues; ++i)
          dict.insert(*(is[2 * i]), *(is[2 * i + 1]));
        return new torch::jit::IValue(dict);
      })
  return nullptr;
}

ivalue ati_int_list(int64_t *is, int nvalues) {
  PROTECT(c10::List<int64_t> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_double_list(double *is, int nvalues) {
  PROTECT(c10::List<double> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_bool_list(char *is, int nvalues) {
  PROTECT(c10::List<bool> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i] != 0);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_string_list(char **is, int nvalues) {
  PROTECT(c10::List<std::string> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(std::string(is[i]));
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_tensor_list(tensor *is, int nvalues) {
  PROTECT(c10::List<at::Tensor> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_device(int device_idx) {
  PROTECT(return new torch::jit::IValue(device_of_int(device_idx));)
  return nullptr;
}

int ati_tag(ivalue i) {
  PROTECT(
      if (i->isNone()) return 0; else if (i->isTensor()) return 1;
      else if (i->isDouble()) return 2; else if (i->isInt()) return 3;
      else if (i->isBool()) return 4; else if (i->isTuple()) return 5;
      else if (i->isIntList()) return 6; else if (i->isDoubleList()) return 7;
      else if (i->isBoolList()) return 8; else if (i->isString()) return 9;
      else if (i->isTensorList()) return 10; else if (i->isList()) return 12;
      else if (i->isGenericDict()) return 13; else if (i->isObject()) return 14;
      throw std::invalid_argument(("unsupported tag " + i->tagKind()).c_str());)
  return -1;
}

tensor ati_to_tensor(ivalue i) {
  PROTECT(return new torch::Tensor(i->toTensor());)
  return nullptr;
}

int64_t ati_to_int(ivalue i) {
  PROTECT(return i->toInt();)
  return -1;
}

double ati_to_double(ivalue i) {
  PROTECT(return i->toDouble();)
  return 0;
}

int ati_to_bool(ivalue i) {
  PROTECT(return i->toBool();)
  return -1;
}

char *ati_to_string(ivalue i) {
  PROTECT(auto str = i->toStringRef(); return strdup(str.c_str());)
  return nullptr;
}

int ati_length(ivalue i) {
  PROTECT(if (i->isTuple()) return i->toTuple()->elements().size();
          else if (i->isIntList()) return i->toIntList().size();
          else if (i->isDoubleList()) return i->toDoubleList().size();
          else if (i->isBoolList()) return i->toBoolList().size();
          else if (i->isString()) return i->toStringRef().size();
          else if (i->isTensorList()) return i->toTensorList().size();
          else if (i->isList()) return i->toList().size();
          else if (i->isGenericDict()) return i->toGenericDict().size();
          throw std::invalid_argument(
              ("unsupported tag for length " + i->tagKind()).c_str());)
  return -1;
}

int ati_tuple_length(ivalue i) {
  PROTECT(return i->toTuple()->elements().size();)
  return -1;
}

void ati_to_tuple(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(auto vec = i->toTuple()->elements(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected tuple size");
  } for (int j = 0; j < noutputs;
         ++j) outputs[j] = new torch::jit::IValue(vec[j]);)
}

void ati_to_generic_list(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(auto vec = i->toList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list size");
  } for (int j = 0; j < noutputs; ++j) outputs[j] =
                                            new torch::jit::IValue(vec[j]);)
}

void ati_to_generic_dict(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(
      auto dict = i->toGenericDict(); if (dict.size() != noutputs) {
        throw std::invalid_argument("unexpected dict size");
      } int k = 0;
      for (auto it = dict.begin(); it != dict.end(); ++it) {
        outputs[k++] = new torch::jit::IValue(it->key());
        outputs[k++] = new torch::jit::IValue(it->value());
      })
}

void ati_to_int_list(ivalue i, int64_t *outputs, int noutputs) {
  PROTECT(auto vec = i->toIntList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<int> size");
  } for (int j = 0; j < noutputs; ++j) outputs[j] = vec[j];)
}

void ati_to_double_list(ivalue i, double *outputs, int noutputs) {
  PROTECT(auto vec = i->toDoubleList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<double> size");
  } for (int j = 0; j < noutputs; ++j) outputs[j] = vec[j];)
}

void ati_to_bool_list(ivalue i, char *outputs, int noutputs) {
  PROTECT(auto vec = i->toBoolList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<bool> size");
  } for (int j = 0; j < noutputs; ++j) outputs[j] = vec[j];)
}

void ati_to_tensor_list(ivalue i, tensor *outputs, int noutputs) {
  PROTECT(auto vec = i->toTensorList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<tensor> size");
  } for (int j = 0; j < noutputs; ++j) outputs[j] =
                                             new torch::Tensor(vec[j]);)
}

ivalue ati_object_method_(ivalue i, char *method_name, ivalue *ivalues,
                          int nivalues) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          inputs.push_back(*i); // self parameter
          for (int j = 0; j < nivalues; ++j) inputs.push_back(*(ivalues[j]));
          torch::jit::IValue output = i->toObjectRef().type()->getMethod(
              method_name)(std::move(inputs));
          return new torch::jit::IValue(output);)
  return nullptr;
}

ivalue ati_object_getattr_(ivalue i, char *attr_name) {
  PROTECT(torch::jit::IValue output = i->toObjectRef().getAttr(attr_name);
          return new torch::jit::IValue(output);)
  return nullptr;
}

ivalue ati_clone(ivalue i) {
  PROTECT(return new torch::jit::IValue(*i);)
  return nullptr;
}

void ati_free(ivalue i) { delete (i); }
