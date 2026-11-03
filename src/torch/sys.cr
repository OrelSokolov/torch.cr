# C ABI declarations for the handwritten shim (ext/torch_api.h).
# C++ objects cross the boundary as opaque pointers.
@[Link("tch")]
lib TorchSys
  type CTensor  = Void*
  type CScalar  = Void*
  type CModule  = Void*
  type CIValue  = Void*

  # ---- errors ----
  fun get_and_reset_last_err : UInt8*

  # ---- misc ----
  fun at_manual_seed(seed : Int64)
  fun at_get_num_threads : Int32
  fun at_set_num_threads(n_threads : Int32)

  # ---- tensor lifecycle ----
  fun at_new_tensor : CTensor
  fun at_tensor_of_blob(data : Void*, dims : Int64*, ndims : LibC::SizeT,
                        strides : Int64*, nstrides : LibC::SizeT, type : Int32,
                        device : Int32) : CTensor
  fun at_tensor_of_data(vs : Void*, dims : Int64*, ndims : LibC::SizeT,
                        element_size_in_bytes : LibC::SizeT, type : Int32) : CTensor
  fun at_copy_data(t : CTensor, vs : Void*, numel : LibC::SizeT,
                   elt_size_in_bytes : LibC::SizeT)
  fun at_shallow_clone(t : CTensor) : CTensor
  fun at_free(t : CTensor)

  # ---- tensor introspection ----
  fun at_data_ptr(t : CTensor) : Void*
  fun at_defined(t : CTensor) : Int32
  fun at_is_sparse(t : CTensor) : Int32
  fun at_is_contiguous(t : CTensor) : Int32
  fun at_device(t : CTensor) : Int32
  fun at_dim(t : CTensor) : LibC::SizeT
  fun at_shape(t : CTensor, dims : Int64*)
  fun at_stride(t : CTensor, dims : Int64*)
  fun at_scalar_type(t : CTensor) : Int32

  # ---- element access ----
  fun at_get(t : CTensor, index : Int32) : CTensor
  fun at_fill_double(t : CTensor, v : Float64)
  fun at_fill_int64(t : CTensor, v : Int64)
  fun at_double_value_at_indexes(t : CTensor, indexes : Int64*, indexes_len : Int32) : Float64
  fun at_int64_value_at_indexes(t : CTensor, indexes : Int64*, indexes_len : Int32) : Int64
  fun at_set_double_value_at_indexes(t : CTensor, indexes : Int32*, indexes_len : Int32, v : Float64)
  fun at_set_int64_value_at_indexes(t : CTensor, indexes : Int32*, indexes_len : Int32, v : Int64)
  fun at_copy_(dst : CTensor, src : CTensor)

  # ---- printing / serialization ----
  fun at_print(t : CTensor)
  fun at_to_string(t : CTensor, line_size : Int32) : UInt8*
  fun at_save(t : CTensor, filename : UInt8*)
  fun at_load(filename : UInt8*) : CTensor

  # ---- autograd ----
  fun at_backward(t : CTensor, keep_graph : Int32, create_graph : Int32)
  fun at_requires_grad(t : CTensor) : Int32
  fun at_grad_set_enabled(b : Int32) : Int32

  # ---- device transfer ----
  fun at_to_device(t : CTensor, device : Int32) : CTensor
  fun at_to_type(t : CTensor, type : Int32) : CTensor

  # ---- device / context ----
  fun atc_cuda_device_count : Int32
  fun atc_cuda_is_available : Int32
  fun atc_cudnn_is_available : Int32
  fun atc_manual_seed(seed : UInt64)
  fun atc_synchronize(device_index : Int64)

  # ---- scalar ----
  fun ats_int(v : Int64) : CScalar
  fun ats_float(v : Float64) : CScalar
  fun ats_to_int(s : CScalar) : Int64
  fun ats_to_float(s : CScalar) : Float64
  fun ats_to_string(s : CScalar) : UInt8*
  fun ats_free(s : CScalar)

  # ---- torchscript module ----
  fun atm_load(filename : UInt8*) : CModule
  fun atm_load_on_device(filename : UInt8*, device : Int32) : CModule
  fun atm_forward(m : CModule, tensors : CTensor*, ntensors : Int32) : CTensor
  fun atm_forward_(m : CModule, ivalues : CIValue*, nivalues : Int32) : CIValue
  fun atm_method_(m : CModule, method_name : UInt8*, ivalues : CIValue*,
                  nivalues : Int32) : CIValue
  fun atm_eval(m : CModule)
  fun atm_train(m : CModule)
  fun atm_free(m : CModule)
  fun atm_to(m : CModule, device : Int32, dtype : Int32, non_blocking : Int32)
  fun atm_save(m : CModule, filename : UInt8*)
  fun atm_named_parameters(m : CModule, data : Void*, f : Void*, UInt8*, CTensor ->)

  # ---- ivalue boxing ----
  fun ati_none : CIValue
  fun ati_tensor(t : CTensor) : CIValue
  fun ati_int(i : Int64) : CIValue
  fun ati_double(d : Float64) : CIValue
  fun ati_bool(i : Int32) : CIValue
  fun ati_string(s : UInt8*) : CIValue
  fun ati_tuple(is : CIValue*, nvalues : Int32) : CIValue
  fun ati_generic_list(is : CIValue*, nvalues : Int32) : CIValue
  fun ati_generic_dict(is : CIValue*, nvalues : Int32) : CIValue
  fun ati_int_list(is : Int64*, nvalues : Int32) : CIValue
  fun ati_double_list(is : Float64*, nvalues : Int32) : CIValue
  fun ati_bool_list(is : UInt8*, nvalues : Int32) : CIValue
  fun ati_string_list(is : UInt8**, nvalues : Int32) : CIValue
  fun ati_tensor_list(is : CTensor*, nvalues : Int32) : CIValue
  fun ati_device(device_idx : Int32) : CIValue
  fun ati_tag(i : CIValue) : Int32
  fun ati_to_tensor(i : CIValue) : CTensor
  fun ati_to_int(i : CIValue) : Int64
  fun ati_to_double(i : CIValue) : Float64
  fun ati_to_bool(i : CIValue) : Int32
  fun ati_to_string(i : CIValue) : UInt8*
  fun ati_length(i : CIValue) : Int32
  fun ati_tuple_length(i : CIValue) : Int32
  fun ati_to_tuple(i : CIValue, outputs : CIValue*, noutputs : Int32)
  fun ati_to_generic_list(i : CIValue, outputs : CIValue*, noutputs : Int32)
  fun ati_to_generic_dict(i : CIValue, outputs : CIValue*, noutputs : Int32)
  fun ati_to_int_list(i : CIValue, outputs : Int64*, noutputs : Int32)
  fun ati_to_double_list(i : CIValue, outputs : Float64*, noutputs : Int32)
  fun ati_to_bool_list(i : CIValue, outputs : UInt8*, noutputs : Int32)
  fun ati_to_tensor_list(i : CIValue, outputs : CTensor*, noutputs : Int32)
  fun ati_object_method_(i : CIValue, method_name : UInt8*, ivalues : CIValue*,
                         nivalues : Int32) : CIValue
  fun ati_object_getattr_(i : CIValue, attr_name : UInt8*) : CIValue
  fun ati_clone(i : CIValue) : CIValue
  fun ati_free(i : CIValue)
end
