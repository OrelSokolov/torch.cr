# torch.cr — план

Биндинги libtorch для Crystal по архитектуре [tch-rs](https://github.com/LaurentMazare/tch-rs).
Выбранный объём: **сразу полный генератор** всей поверхности ATen (~2000 ops),
плюс рукописное ядро и полная автоматизация подключения («по щелчку пальцев»).

## Целевая среда (зафиксировано)

- torch **2.11.0+cu128**, `_GLIBCXX_USE_CXX11_ABI = 1` (CXX11 ABI включён!)
- venv: `~/dataset-vox-cpm2/.venv` — отсюда берём include/lib пути (режим как
  `LIBTORCH_USE_PYTORCH` в build.rs у tch)
- Альтернатива: скачанный zip с download.pytorch.org или `LIBTORCH=...`
- Конечная цель — инференс VoxCPM2 (TorchScript) из Crystal без Python

## Архитектура — 4 слоя (как в tch-rs)

```
torch.cr/
├── shard.yml                  # postinstall: bin/setup  (см. «Автоматизация»)
├── Rakefile                   # gen / build / test / regen-схем
├── bin/setup                  # установка: найти libtorch → сгенерить → собрать libtch.so
├── ext/
│   ├── torch_api.h / .cpp     # слой 1: рукописный C-шим (порт из tch, ~ дословно)
│   ├── torch_api_generated.*  # слой 2: atg_<op> на каждый ATen op (генерится)
│   └── build.sh               # clang++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 ...
├── gen/
│   ├── generator.cr           # порт gen.ml (OCaml) на Crystal — OFFLINE-кодоген
│   └── codegen.cr             # то же, но для {{ run() }} compile-time режима
├── vendor/
│   └── schemas-2.11.yaml      # дамп API (генерится tools/dump_schemas.py, коммитится)
├── tools/
│   └── dump_schemas.py        # torch._C._jit_get_all_schemas() → YAML
├── spec/                      # parity-тесты против python-torch
└── src/
    ├── torch.cr               # точка входа
    ├── sys.cr                 # слой 3: lib TorchSys — fun at_/atm_/ati_/atg_
    ├── loader.cr              # слой 3b: dlopen-режим (нуль-конфиг линковки)
    ├── tensor.cr              # слой 4: class Tensor (finalize → at_free)
    ├── module.cr              # class Module (TorchScript), IValue
    ├── kind.cr / device.cr    # enum Kind, Device, Scalar, Layout
    └── generated/             # слой 4b: generated bindings (генерится)
```

### Слой 1 — рукописный шим (порт `torch-sys/libtch/torch_api.h/.cpp`)

- Opaque-указатели: `typedef torch::Tensor *tensor; ... *module; ... *ivalue;`
- `PROTECT(x)`: catch → `thread_local char *torch_last_err`, наружу
  `char *get_and_reset_last_err()`
- Жизнь/данные: `at_new_tensor`, `at_free`, `at_tensor_of_data`, `at_copy_data`,
  `at_shape`, `at_stride`, `at_scalar_type`, `at_device`
- TorchScript: `atm_load[_on_device]`, `atm_forward_` (IValue-вариант), `atm_eval`,
  `atm_to(device, dtype)`, `atm_named_parameters`
- IValue-боксинг: `ati_tensor/int/double/bool/string/tuple/..._list`,
  обратные `ati_to_*`, `ati_tag`
- Optimizer-набор (Adam/SGD/...) — можно отложить, для инференса не нужен

### Слой 2 — генерируемый шим

По каждому op: `atg_<name>(...)` с C ABI. Таблица типов (из gen.ml):

| C++                         | C ABI                                   |
|-----------------------------|-----------------------------------------|
| `Tensor`                    | `tensor` (= указатель)                  |
| `optional<Tensor>`          | `tensor` + null-соглашение (nullptr)    |
| `IntArrayRef`               | `int64_t *data, int len`                |
| `ArrayRef<double>`          | `double *data, int len`                 |
| `optional<int64_t/double>`  | `T v, uint8_t null`                     |
| `string_view`               | `char *ptr, int len`                    |
| `TensorOptions`             | `int kind, int device`                  |
| `Scalar`                    | `scalar`                                |
| возврат N тензоров          | out-парам `tensor *out__`               |
| возврат TensorList          | `tensor**` (nullptr-terminated)         |

### Слой 3 — `lib TorchSys` + опциональный dlopen-загрузчик

### Слой 4 — безопасные обёртки

- `Tensor` оборачивает `c_tensor : Void*`; `finalize` зовёт `at_free`
  (GC-финалайзер; подробно в «Риски»)
- Генерятся два вида методов на каждый op (как `f_*` / обычные в tch):
  - `Tensor.add!(...)` → бросает `Tensor::Error`
  - `Tensor.add?(...)` → `Result`-стиль (возврат nil + `last_error`)

## Генератор на Crystal (`gen/generator.cr`)

Порт `gen.ml` (~700 строк OCaml). Crystal подходит идеально: YAML в stdlib.

Вход — `vendor/schemas-2.11.yaml`:
- дампит `tools/dump_schemas.py` из установленного torch
  (`torch._C._jit_get_all_schemas()` — имя, аргументы с типом/nullability/дефолтом,
  возвраты, schema kind)
- запасной вход: `Declarations-*.yaml` из tch-rs `third_party/` (формат тот же по смыслу)
- регенерация — одна rake-задача при апгрейде torch, артефакт коммитится

Логика (переносим из gen.ml):
1. Фильтры: excluded-функции, excluded-префиксы (`_foreach`, `_th_`, `sym_`...),
   суффиксы (`_forward`...)
2. Дизамбигуация перегрузок: `operator_name` + `overload_name` →
   `add_Scalar`, `add__Tensor` и т.п.
3. Дефолтные Scalar-аргументы отбрасываются (кроме whitelist), как в tch
4. Санитизация имён (`where`→`where_`, ведущий `_`→`internal_`)
5. Эмиссия: `ext/torch_api_generated.{h,cpp}` + `src/generated/sys.cr` +
   `src/generated/tensor.cr`

## «Щелчок пальцев» — автоматический режим

Проблема: у Crystal **нет** build.rs (postinstall-хуков компиляции нет, `@[Link]`
не знает про каталоги шарда). Решение — комбинация из трёх трюков:

1. **`postinstall` в shard.yml** — `shards` умеет выполнять команду после
   установки зависимости: `postinstall: bin/setup`.
   `bin/setup` (Crystal-скрипт):
   - находит libtorch: `$LIBTORCH` → venv с torch (`$TORCH_VENV`, автопоиск) →
     system → скачать zip с download.pytorch.org (как build.rs у tch)
   - запускает `gen/generator.cr` (если vendor-схемы новее артефактов)
   - компилит `libtch.so` через clang++ с правильным ABI-флагом
   - пишет `ext/manifest.json` (версия torch, пути) для последующих запусков

2. **Compile-time кодоген** — `{{ run("gen/codegen.cr", ...) }}`: Crystal умеет
   выполнить Crystal-программу во время компиляции и вставить её stdout как AST.
   `codegen.cr` проверяет кэш (`src/generated/*.cr`), при устаревании
   регенерирует. Итог: после `shards install` просто `crystal build` —
   биндинги самоподдерживающиеся, никаких ручных шагов.

3. **Линковка** — два режима:
   - **A (каноничный)**: `@[Link("tch")]`, пользователь передаёт
     `--link-flags "-L$(pwd)/ext"`. Просто, но не «щелчок».
   - **B (нуль-конфиг)**: dlopen-загрузчик — `libtch.so` грузится в рантайме
     (`LibC.dlopen`/`dlsym`, путь известен из манифеста postinstall),
     генератор эмитит ленивые Proc-указатели вместо `fun`. Ни одного флага
     линковки, никаких rpath-проблем, версионирование рантайм-библиотеки.
     Оверхед нулевой после первого dlsym (указатели кэшируются).
   - Режим выбирается флагом сборки (например, ` -Dtorch_dlopen`), дефолт — B.

Итого UX: в `shard.yml` добавил зависимость → `shards install` → `require "torch"`
→ работает. Обновился torch → `rake regen`.

## Этапы (порядок работ)

- [x] **0. Скелет**: репо, shard.yml, Makefile, README, PLAN (MIT как tch)
- [x] **1. Схемы**: TODO — `tools/dump_schemas.py` (отложено до генератора)
- [x] **2. Ядро ext/**: порт `torch_api.{h,cpp}` (tensor + jit + ivalue) —
      `make ext` собирает `libtch.so` (дефолт, самодостаточный с rpath;
      `STATIC=1` — `libtch.a`); ABI подбирается по установленному torch
- [ ] **2b. Смоук-тесты**: ✅ 12 спеков зелёные (данные, jit forward,
      IValue-туры, named_parameters, ошибки); twinshot-оракул 6/6 —
      python и crystal деревья побайтово совпадают
- [ ] **3. generator.cr v1 (C++ эмиссия)**: yaml → torch_api_generated.{h,cpp};
      чек: линкуется, кол-во atg_* ~ 2000
- [ ] **4. generator.cr v2 (Crystal эмиссия)**: lib-декларации + safe-обёртки;
      `{{ run }}`-кэш
- [ ] **5. CUDA/device**: atm_to(device), стресс-тест GC-финалайзеров
- [ ] **6. Интеграционный**: трейс VoxCPM2-кусков в TorchScript (python, разово)
      → грузим и гоняем из Crystal; сравнение с python-инференсом (twinshot)

## Риски / решения

| Риск | Решение |
|---|---|
| CXX11 ABI (у нас =1, у zip-сборок бывает 0) | ABI-флаг определяется из torch (`torch._C._GLIBCXX_USE_CXX11_ABI`) и передаётся в build.sh |
| GC-финалайзеры vs torch allocator | `at_free` из `finalize`; тензоры освобождаются и из GC-потока — c10 потокобезопасен; при проблемах — явный `#free` + финалайзер-страховка |
| Перегрузки/дефолты ATen | точный порт логики disambiguation из gen.ml (проверено годами в tch) |
| Размер сгенерённого .cr (~2000 fun + 2×методов) | компиляция может подтормаживать; кэш + измерить; при необходимости резать по подмножествам (`torch/core` vs `torch/all`) |
| Разные версии torch у пользователей | схемы версионируются; postinstall перегенерит под локальный torch; допускать минорные расхождения |
| macOS/Windows | первая версия Linux-only; macOS потом (в tch поддерживается, шим портативен) |

## Ссылки-первоисточники

- gen.ml (референс генератора): `tch-rs/gen/gen.ml`
- Рукописный шим: `tch-rs/torch-sys/libtch/torch_api.{h,cpp}`
- Сборка/поиск libtorch: `tch-rs/torch-sys/build.rs`
- Дамп схем: `torch._C._jit_get_all_schemas()` в python
