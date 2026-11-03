# torch.cr — libtorch bindings for Crystal

Тонкий явный C-шим поверх libtorch (порт рукописного слоя
[tch-rs](https://github.com/LaurentMazare/tch-rs)) + безопасные Crystal-обёртки:
тензоры, TorchScript-модули, IValue. Полный генератор поверхности ATen — по
плану в `PLAN.md`.

## Быстрый старт

```bash
make ext        # собирает ext/libtch.so (STATIC=1 — libtch.a)
make fixture    # python-torch генерит TorchScript-фикстуры + golden.json
make spec       # crystal spec
make verify     # twinshot review: python-golden vs crystal-current, 6/6
```

Подключение к своему проекту:

```crystal
require "torch"

m = Torch::Module.load("model.pt").eval
x = Torch::Tensor.of([1.0_f32, 2.0_f32, 3.0_f32], [1, 3])
y = m.forward(x)
puts y.to_f32
```

Сборка бинарника (`TORCH` — каталог пакета torch, см. `bin/detect_torch.sh`):

```bash
crystal build app.cr --link-flags "-L/path/to/torch.cr/ext -Wl,-rpath,/path/to/torch.cr/ext -lstdc++"
```

Шим собирается как самодостаточный `libtch.so` (rpath в libtorch). Вариант
`STATIC=1 ./ext/build.sh` даёт `libtch.a`, но из-за фиксированного порядка
флагов у Crystal удобен только при ручной линковке (`LINK_FLAGS_STATIC` в
Makefile); libtorch в pip-сборке всё равно только динамический.

## Поиск libtorch

`bin/detect_torch.sh` ищет в порядке: `$LIBTORCH` → `python3` с torch →
`$TORCH_VENV/bin/python` → известные venv. ABI (`_GLIBCXX_USE_CXX11_ABI`)
подбирается под установленную сборку автоматически.

## Приложение: `bin/infer`

TorchScript-раннер с бенчмарком и проверкой против Python-золота:

```bash
make ext && make fixture
crystal build --release -o bin/infer --link-flags "$(make -s print-link-flags)" bin/infer.cr   # или см. Makefile

./bin/infer spec/fixtures/mlp.pt --device cpu --iters 100 --golden mlp
./bin/infer spec/fixtures/mlp.pt --device cuda:0 --iters 200 --golden mlp
```

Пример вывода (RTX 5080):

```
device:  cuda:0
latency: avg=0.119ms  min=0.048ms  p50=0.057ms  max=1.910ms
out[0..3]: -0.133337 0.265934 0.157105 0.191246
golden:  OK (mlp, max_err=2.38e-07)
```

CUDA-путь полностью рабочий: тензор и модуль переносятся на девайс
(`Tensor#to`, `Module#to`), инференс с синхронизацией, результат копируется
обратно на CPU и сверяется.

## Оракул

Два независимых источника правды для результатов:

1. `spec/fixtures/golden.json` — значения из python, проверяются в спеках.
2. twinshot-деревья: `spec/support/oracle_dump.py` пишет `oracle/golden`,
   `bin/oracle_dump.cr` пишет `oracle/current`; `make verify` = regenerate +
   compare. Текущий статус: 6/6 файлов идентичны (txt-адаптер, по строкам).

## Статус

- [x] явный C-шим (tensor/io/jit/ivalue) — `ext/torch_api.{h,cpp}`
- [x] Crystal-обёртки Tensor / IValue / Module (12 спеков зелёные)
- [x] twinshot-оракул: python vs crystal, 6/6
- [x] CUDA: перенос тензоров/модулей на девайс, инференс на RTX 5080,
      parity с Python (max_err ~2e-7), бенчмарк-CLI `bin/infer`
- [ ] генератор полной поверхности ATen (`gen/generator.cr`, см. PLAN.md)
- [ ] оптимизаторы, автоген при `shards install`
