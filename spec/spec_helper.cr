require "json"
require "spec"
require "../src/torch"

FIXTURES = File.expand_path(__DIR__ + "/fixtures")
GOLDEN = JSON.parse(File.read("#{FIXTURES}/golden.json"))

def fixture_path(name : String) : String
  "#{FIXTURES}/#{name}"
end

def fmt_grid(values : Array(Array(Float64))) : Array(String)
  values.map { |row| row.map { |v| sprintf("%0.6f", v) }.join(" ") }
end

def expect_grid(actual : Array(Array(Float64)), expected_raw : Array(JSON::Any)) : Nil
  expected = expected_raw.map { |row| row.as_a.map(&.as_f) }
  actual.map(&.map { |v| v.round(6) }).should eq expected.map(&.map { |v| v.round(6) })
end
