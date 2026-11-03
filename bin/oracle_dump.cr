# Dumps the "current" oracle tree for twinshot comparison: writes plain-text
# files whose byte content must match the golden tree produced by
# spec/support/oracle_dump.py (python torch).
#
#   ./bin/oracle_dump [output_dir]   (default: oracle/current)

require "json"
require "../src/torch"

out_dir = ARGV[0]? || "oracle/current"
Dir.mkdir_p(out_dir)

fixtures = File.expand_path("#{__DIR__}/../spec/fixtures")
golden = JSON.parse(File.read("#{fixtures}/golden.json")) if File.exists?("#{fixtures}/golden.json")
x_data = golden.try &.["x"].as_a.map(&.as_a.map(&.as_f32))

def fmt(values : Array(Float64)) : Array(String)
  values.map { |v| sprintf("%0.6f", v) }
end

def grid_rows(values : Array(Float64), width : Int32) : Array(String)
  values.each_slice(width).map { |row| fmt(row).join(" ") }.to_a
end

# -- tensor io ---------------------------------------------------------------
t = Torch::Tensor.of([1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32], [2, 2])
File.write("#{out_dir}/tensor_io.txt",
  (["shape=#{t.shape.join('x')}", "kind=#{t.kind}"] +
   grid_rows(t.to_f32.map(&.to_f64), 2)).join("\n") + "\n")

ti = Torch::Tensor.of([10_i64, 20_i64, 30_i64])
File.write("#{out_dir}/tensor_i64.txt", ti.to_i64.join("\n") + "\n")

# -- traced module -----------------------------------------------------------
if x_data
  x = Torch::Tensor.of(x_data, [2, 3])
  m = Torch::Module.load("#{fixtures}/double.pt").eval
  File.write("#{out_dir}/forward_double.txt",
    grid_rows(m.forward(x).to_f32.map(&.to_f64), 3).join("\n") + "\n")

  multi = Torch::Module.load("#{fixtures}/multi.pt").eval
  out = multi.forward([Torch::IValue.of(x), Torch::IValue.of(3)])
  y, rowsum, ret = out.as_list
  File.write("#{out_dir}/forward_multi.txt",
    (grid_rows(y.as_tensor.to_f32.map(&.to_f64), 3) +
     fmt(rowsum.as_tensor.to_f32.map(&.to_f64)) +
     [ret.as_i64.to_s]).join("\n") + "\n")

  lin = Torch::Module.load("#{fixtures}/linear.pt").eval
  File.write("#{out_dir}/forward_linear.txt",
    grid_rows(lin.forward(x).to_f32.map(&.to_f64), 2).join("\n") + "\n")

  names = [] of String
  lin.each_named_parameter { |name, _| names << name }
  File.write("#{out_dir}/linear_params.txt", names.sort.join("\n") + "\n")
end

puts "oracle tree dumped to #{out_dir}"
