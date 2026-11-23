[English](README.md) | **Magyar**

# Echter XGBoost

![C++20 modules](https://img.shields.io/badge/C%2B%2B20-modules-blue)
![CUDA 13](https://img.shields.io/badge/CUDA-13-76B900)
![cuDF 26.02](https://img.shields.io/badge/cuDF-26.02-7400B8)
![CMake 3.28+](https://img.shields.io/badge/CMake-3.28%2B-064F8C)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue)

XGBoost regressziós modellek inferenciája GPU-n, C++20-ban. Az Echter XGBoost betölti az XGBoost által mentett modellt, egy CUDA-kernellel a GPU-n számítja ki a predikciókat, a CSV-fájlokat pedig a cuDF segítségével a GPU-n olvassa be és írja ki.

- **Ugyanazok az eredmények, mint az XGBoostnál:** a predikciók bitre azonosak az XGBoost GPU-s prediktorának eredményeivel.
- **Gyors:** egy H200-on másodpercenként 75–85 millió sor, ugyanazon a GPU-n 2,5–2,9× akkora áteresztőképesség, mint az XGBoost C API-jával, a host memóriájában lévő bemenettel 3,8–7×, a modell betöltése pedig 12× gyorsabb. Lásd: [Teljesítmény](#teljesítmény).
- **Végig a GPU-n:** CSV-bemenet, oszlopkiválasztás, predikció és CSV-kimenet anélkül, hogy az adatok megjárnák a host memóriáját.
- **Biztonságos használat:** a modellek a feltöltés előtt ellenőrzésen esnek át, az eszközmemóriát RAII-típusok birtokolják, az eszközadatként átadott, host memóriára mutató pointereket pedig a könyvtár még a kernel indítása előtt elutasítja.
- **Modern C++:** C++20-as nevesített modulok; a nyilvános modulok nem teszik láthatóvá a CUDA és a cuDF fejlécfájljait.

## Tartalom

- [Teljesítmény](#teljesítmény)
- [Követelmények](#követelmények)
- [Fordítás és telepítés](#fordítás-és-telepítés)
- [Parancssori predikció](#parancssori-predikció)
- [A könyvtár használata](#a-könyvtár-használata)
- [Támogatott modellek](#támogatott-modellek)
- [A predikció működése](#a-predikció-működése)
- [Példák](#példák)
- [Tesztek](#tesztek)
- [Összevetés az XGBoosttal](#összevetés-az-xgboosttal)
- [A projekt felépítése](#a-projekt-felépítése)
- [Licenc](#licenc)

## Teljesítmény

Az alább bemutatott benchmarkmodellel az Echter XGBoost egy H200-on **másodpercenként 75–85 millió sorra** számít predikciót, ha a bemenet már a GPU-memóriában van, a host memóriájából pedig másodpercenként 44–52 millió sorra. Az XGBoost ugyanezekben a helyzetekben másodpercenként 26–33 millió, illetve 7–12 millió sort ér el. A cuML-hez tartozó Forest Inference Library (FIL) a legkisebb kötegnél 3%-kal gyorsabb, a nagyobbaknál 5–9%-kal lassabb, és a predikciói eltérnek az XGBoost predikcióitól.

### Mérési környezet

- **Hardver:** NVIDIA H200 GPU, Intel Xeon Platinum 8480C CPU
- **Szoftver:** CUDA 13.1, cuDF 26.02, XGBoost 3.4.1 `device="cuda"` beállítással
- **XGBoost-interfészek:** a Python-csomag (`Booster.inplace_predict` NumPy-tömbön) és a C++-ból hívott C API (`XGBoosterPredictFromDense` a host memóriájában lévő bemenethez, `XGBoosterPredictFromCudaArray` a GPU-n lévő bemenethez)
- **cuML FIL:** cuML 26.02 Pythonból, CuPy-bemenettel, egyszeres pontossággal, a három csomópont-elrendezés és a hat chunkméret közül a leggyorsabb kombinációval
- **Adatok:** 489 046 valódi sor 19 jellemzővel, a nagyobb kötegekhez 10×, illetve 20× megismételve
- **Módszer:** futásonként 9 hívás mediánja (az XGBoostnál, a host memóriájában lévő bemenettel 5 hívásé), majd ezek mediánja az ismételt futásokon

Az Echter XGBoost mind a négy bemeneti útján, valamint az XGBoost C API-ja és Python-csomagja a benchmark mind a 15,2 M során bájtra azonos predikciókat adnak; a cuML FIL nem. A benchmarkprogramok a [`scripts/benchmarks`](#benchmarkok) könyvtárban találhatók.

### Benchmarkmodell

| Tulajdonság | Érték |
|---|---|
| Fák | 500 (`reg:squarederror`, 19 jellemző) |
| Csomópontok | Összesen 6 715 914: 3 357 707 elágazás és 3 358 207 levél, fánként nagyjából 13 400 |
| Mélység | Átlagos levélmélység 14,7, maximális mélység 16 |
| Modellfájl (JSON) | 387 MB |
| Fák a GPU-memóriában | 54 MB, csomópontonként 8 bájt |

### Összehasonlítás az XGBoosttal és a cuML FIL-lel

Másodpercenként feldolgozott sorok, és az Echter XGBoost viszonya a többiekhez.

**A bemenet már a GPU-memóriában van**

| Sorok | XGBoost C API | cuML FIL | Echter | Gyorsulás az XGBoosthoz képest | A FIL-hez viszonyítva |
|---:|---:|---:|---:|---:|---:|
| 489 046 | 26,0 M sor/s | 77,3 M sor/s | 75,2 M sor/s | **2,9×** | 0,97× |
| 4 890 460 | 32,1 M sor/s | 77,7 M sor/s | 81,5 M sor/s | **2,5×** | **1,05×** |
| 9 780 920 | 32,7 M sor/s | 77,7 M sor/s | 84,6 M sor/s | **2,6×** | **1,09×** |

A cuML FIL is 8 bájtos csomópontokat használ, de a sorokat a megosztott memóriába másolja, és egy sor fáit több szál között osztja szét. A fák szétosztása megváltoztatja a levélértékek összegzési sorrendjét, ezért a predikciói nem bitre azonosak az XGBoost predikcióival, még kétszeres pontossággal sem:

| | Bitre azonos az XGBoosttal | Legnagyobb eltérés |
|---|---:|---:|
| Echter XGBoost | 100% | 0 ULP |
| cuML FIL | 14% | 272 ULP |

**A bemenet a host memóriájában van** (minden hívás a GPU-ra másolja a bemenetet, a predikciókat pedig vissza)

| Sorok | XGBoost Python | XGBoost C API | Echter | Gyorsulás a Pythonhoz képest | Gyorsulás a C API-hoz képest |
|---:|---:|---:|---:|---:|---:|
| 489 046 | 7,7 M sor/s | 7,4 M sor/s | 51,5 M sor/s | **6,7×** | **6,9×** |
| 4 890 460 | 11,3 M sor/s | 11,5 M sor/s | 47,7 M sor/s | **4,2×** | **4,2×** |
| 9 780 920 | 11,6 M sor/s | 11,1 M sor/s | 44,2 M sor/s | **3,8×** | **4,0×** |

A host memóriájában lévő bemenettel és GPU-s boosterrel az XGBoost először egy DMatrix objektumot épít az adatokból, ahogy azt a saját, teljesítményre vonatkozó figyelmeztetése is jelzi. A Python-csomagnak a GPU-n lévő bemenethez CuPy vagy cuDF kell, ezért a GPU-n lévő bemenettel végzett összehasonlítás a C API-t használja.

A mért hívások, amikor a bemenet már a GPU-n van:

```cpp
// XGBoost C API: the input is described by a JSON __cuda_array_interface__ string
XGBoosterPredictFromCudaArray(booster, array_interface, config, nullptr, &shape, &dim, &result);

// Echter XGBoost
const echter::xgb::DevicePrediction prediction = regression.predict(device_input);
```

```python
# cuML FIL
fil = ForestInference.load("model.json", model_type="xgboost_json", output_type="cupy")
prediction = fil.predict(cupy_input)
```

**Modellbetöltés** (a 387 MB-os JSON-fájl)

| XGBoost Python | XGBoost C API | cuML FIL | Echter | Gyorsulás az XGBoosthoz képest | Gyorsulás a FIL-hez képest |
|---:|---:|---:|---:|---:|---:|
| 10,0 s | 10,2 s | 5,9 s | 0,85 s | **12×** | **7×** |

### Az Echter XGBoost bemeneti útjai

Ugyanaz a modell és ugyanazok az adatok azon a négy úton, amelyen egy tábla a modellhez juthat, másodpercenként feldolgozott sorokban. Mindegyik úthoz tartozik egy példaprogram a `bin/examples/` könyvtárban.

| Bemenet | API | Példa | 489 046 sor | 4 890 460 sor | 9 780 920 sor |
|---|---|---|---:|---:|---:|
| CUDA-eszközpuffer | `predict(DeviceColumnarData)` | `echter_xgb_cuda_prediction` | 75,2 M/s | 81,5 M/s | 84,6 M/s |
| cuDF-tábla | `predict(regression, cudf::table_view)` | `echter_xgb_cudf_prediction` | 70,6 M/s | 79,9 M/s | 83,5 M/s |
| Tömb a host memóriájában | `predict(HostColumnarView)` | `echter_xgb_cpu_prediction` | 51,5 M/s | 47,7 M/s | 44,2 M/s |
| CSV-fájl | `io::read_csv`, `io::select_columns`, `predict` | `echter_xgb_csv_prediction` | 11,9 M/s | 13,8 M/s | 14,3 M/s |

A cuDF-es út magában foglalja a tábla oszlopainak egyetlen oszlopfolytonos pufferbe csomagolását. A host memóriájából induló út a host és a GPU közötti mindkét másolást tartalmazza. A CSV-s út a fájl GPU-n történő beolvasását és elemzését is magában foglalja; a 9 780 920 soros fájl mérete 3,0 GB.

A négy út kódban:

```cpp
namespace xgb = echter::xgb;

// CUDA device buffer: fill the library-owned buffer, the prediction stays on the GPU
auto input = regression.allocate_device(rows, features);
cudaMemcpy(input.mutable_data(), host_data, rows * features * sizeof(float), cudaMemcpyHostToDevice);
const xgb::DevicePrediction from_buffer = regression.predict(input);

// cuDF table: one numeric column per feature (#include "src/regression/regression_cudf_adapter.hpp")
const xgb::DevicePrediction from_table = xgb::predict(regression, table.view());

// Host array: column-major, host_data[feature * rows + row]
const xgb::Prediction from_host = regression.predict(xgb::HostColumnarView{host_data, rows, features});

// CSV file, parsed on the GPU; columns 0 and 1 are not features
const auto csv = xgb::io::read_csv("input.csv");
const xgb::DevicePrediction from_csv = regression.predict(xgb::io::select_columns(csv, {0, 1}));
```

### CSV-ből CSV-be

A teljes feldolgozási lánc a 489 046 soros, 21 oszlopos CSV-fájlon: a [`scripts/predict.py`](scripts/predict.py) (pandas és XGBoost) az `echter_xgb_predict` programmal szemben, három futás mediánja.

| Lépés | XGBoost (Python) | Echter | Gyorsulás |
|---|---:|---:|---:|
| Modellbetöltés | 10 356 ms | 849 ms | 12× |
| CSV-olvasás | 911 ms | 69 ms | 13× |
| Predikció | 294 ms (DMatrix 56,0 + predict 237,9) | 6 ms | 49× |
| CSV-írás | 420 ms | 26 ms | 16× |
| **Összesen** | **12,1 s** | **0,95 s** | **13×** |
| Sor másodpercenként, teljes lánc | 41 k | 516 k | |

A két mért parancs:

```bash
python scripts/predict.py model.json input.csv --target-column incident_proton_energy
./bin/echter_xgb_predict model.json input.csv --target-columns 1
```

## Követelmények

- Linux és NVIDIA GPU, amelynek compute capability értéke legalább 7.5
- CUDA Toolkit (a 13.1-es verzióval tesztelve)
- libcudf, a cuDF C++-könyvtára (a 26.02-es verzióval tesztelve, például a RAPIDS conda-csomagjaiból)
- CMake 3.28 vagy újabb és Ninja 1.11 vagy újabb (a C++20-as modulok nem működnek a Makefile-generátorokkal)
- Modulokat támogató C++20-fordító (GCC 15.2 és Clang 21.1 verzióval tesztelve)
- Clang esetén a GCC 13-as vagy újabb verziójából származó libstdc++, ahogy azt a cuDF megköveteli, például `-DCMAKE_CXX_FLAGS=--gcc-toolchain=/opt/gcc-15.2`

## Fordítás és telepítés

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
cmake --install build --strip
```

Ha a CMake nem találja a cuDF-et, adjuk meg a telepítési prefixét, például: `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`.

A könyvtár megosztott könyvtárként készül el, `libEchter_XGBoost.so` néven. A `cmake --install` a forrásfán belül ezt a `bin/lib/`, a parancssori alkalmazást a `bin/`, a tesztet a `bin/test/`, a példákat pedig a `bin/examples/` könyvtárba helyezi; máshová a `--prefix` kapcsolóval telepíthetünk. A futtatható állományok a `libEchter_XGBoost.so` fájlt a saját helyükhöz viszonyítva találják meg, a CUDA-t és a cuDF-et pedig azokban a könyvtárakban, amelyekből a linkeléskor származtak.

A repository `bin/` könyvtárában lévő futtatható állományok a benchmarkokat futtató gépen készültek, az alább felsorolt összes GPU-architektúrára. Olyan gépen, ahol a CUDA 13 és a cuDF 26.02 más könyvtárakba van telepítve, adjuk hozzá ezeket a könyvtárakat az `LD_LIBRARY_PATH` változóhoz:

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/local/cuda/lib64 ./bin/echter_xgb_predict model.json input.csv
```

| CMake-opció | Alapértelmezés | Cél |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `all-major` (sm_75-től sm_120-ig, valamint PTX) | `native` értékre állítva csak a helyi GPU-ra fordít; a `CUDAARCHS` környezeti változó is használható |
| `CMAKE_BUILD_TYPE` | `Release` | |
| `ECHTER_XGB_BUILD_PREDICT` | `ON` | Az `echter_xgb_predict` alkalmazás |
| `ECHTER_XGB_BUILD_TESTS` | `ON` | A futtatható teszt és a CTest-regisztráció |
| `ECHTER_XGB_BUILD_EXAMPLES` | `ON` | A négy példa |
| `ECHTER_XGB_TEST_MODEL` | üres | JSON-modell egy további, valódi modellel végzett CTest-futtatáshoz |

A három `BUILD` opció alapértelmezett értéke `OFF`, ha a projektet egy másik CMake-projekt építi be. Például ha csak a helyi GPU-ra szeretnénk fordítani, és a valódi modellel futó tesztet is fel szeretnénk venni a CTestbe:

```bash
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=native -DECHTER_XGB_TEST_MODEL=/path/to/model.json
```

## Parancssori predikció

```bash
./bin/echter_xgb_predict model.json input.csv --target-columns 1
```

| Argumentum | Jelentés |
|---|---|
| `model.json` | JSON formátumú XGBoost-modell |
| `input.csv` | Numerikus CSV fejlécsorral |
| `[key-column]`, `--key-column INDEX` | Azonosítóoszlop, `id` néven íródik ki (alapértelmezés: `0`) |
| `--target-columns INDEX[,INDEX...]` | Oszlopok, amelyek nem bemenetei a modellnek, de `target_<index>` néven átkerülnek a kimenetbe |
| `--ignore-columns INDEX[,INDEX...]` | Oszlopok, amelyek nem bemenetei a modellnek, és ki sem íródnak |

Az indexek nullától kezdődnek, az opciók ismételhetők, az ismétlődő indexeket pedig a program figyelmen kívül hagyja. Az összes többi oszlop jellemző, és számuknak meg kell egyeznie a modell jellemzőinek számával. Az eredmény az `outputs/<input-stem>_predictions.csv` fájlba íródik, `id,target_<index>...,prediction` oszlopokkal. A program kiírja a sorok és oszlopok számát, valamint minden lépés idejét.

```text
id,target_1,prediction
2.0,1.0,1.001273155
4.0,1.0,0.99904871
```

## A könyvtár használata

```cpp
#include <vector>

import echter.xgb;

int main()
{
    echter::xgb::Regression regression;
    regression.load_model("model.json");

    // Column-major input, data[feature * rows + row]; NaN marks a missing value.
    const std::size_t rows = 2;
    const std::size_t features = regression.num_features();
    std::vector<float> data(rows * features, 0.0f);

    const auto result = regression.predict(
        echter::xgb::HostColumnarView{data.data(), rows, features});
    // result.values holds one prediction per row.
}
```

Az adatok a bemenettől a kimenetig a GPU-n maradhatnak. Eszközmemóriában lévő bemenet esetén a `predict()` egy `DevicePrediction` objektumot ad vissza:

```cpp
namespace io = echter::xgb::io;

const auto table = io::read_csv("input.csv");             // numeric columns on the GPU
const auto features = io::select_columns(table, {0, 1});  // drop the id and target columns
const auto prediction = regression.predict(features);     // result stays on the GPU

io::write_csv(
    "predictions.csv",
    std::vector<echter::xgb::DeviceColumnarView>{
        {table.view().data, table.rows(), 1},             // column 0 of the table
        prediction.values.view()},
    {"id", "prediction"});
```

| Modul | Tartalom |
|---|---|
| `echter.xgb` | Az összes alábbi modul, valamint az `XGBoost` facade-osztály a `regression()` függvénnyel |
| `echter.xgb.reg` | `Regression`: `load_model`, `predict` a host és az eszköz memóriájában lévő bemenetre, `upload`, `allocate_device`, `num_features`, `num_trees` |
| `echter.xgb.data` | `HostColumnarView`, `DeviceColumnarView`, `DeviceColumnarData` (eszközmemóriát birtokol; `view()`, `mutable_data()`), `Prediction`, `DevicePrediction` |
| `echter.xgb.io` | `read_csv`, `select_columns`, `write_csv`, `read_json` |

Megjegyzések:

- Az `io::read_csv` minden oszlopot `float` típusúra alakít; az üres mezőkből `NaN` lesz. Az `io::write_csv` csak akkor ír fejlécet, ha nevek is meg vannak adva, oszloponként egy.
- Egy `DeviceColumnarView` objektumnak eszközmemóriára kell mutatnia. A host memóriára mutató pointereket a könyvtár kivétellel utasítja el.
- A hibákat kivételek jelzik, amelyek üzenete megnevezi a fájlt és a hiba okát.
- A mozgatás forrásaként szolgáló `Regression` és `DeviceColumnarData` objektumok üresek; egy ilyen modell használatkor „not loaded” kivételt dob.

```cpp
try
{
    regression.load_model("model.json");
}
catch (const std::exception& error)
{
    // for example: failed to load XGBoost regression model model.json: file not found
    std::cerr << error.what() << '\n';
}
```

Ha egy másik CMake-projektből szeretnénk használni a könyvtárat, vegyük fel alkönyvtárként. A C++20-as követelmény a könyvtárat használó projektre is átöröklődik:

```cmake
find_package(CUDAToolkit REQUIRED)
find_package(cudf CONFIG REQUIRED)
add_subdirectory(Echter_XGBoost)
target_link_libraries(my_app PRIVATE Echter_XGBoost::Echter_XGBoost)
```

## Támogatott modellek

A `Regression::load_model()` a `Booster.save_model("model.json")` hívással mentett modelleket olvassa be, ha azok a következőket használják:

- a `gbtree` boostert egyetlen célváltozóval,
- numerikus elágazásokat,
- kimeneti transzformáció nélküli célfüggvényt: `reg:squarederror`, `reg:squaredlogerror`, `reg:pseudohubererror`, `reg:absoluteerror` vagy `reg:quantileerror` egyetlen kvantilissel.

Más modelleket a könyvtár hibával elutasít, ahelyett, hogy hibás számokat adna. Ide tartoznak az olyan célfüggvények, mint a `reg:logistic`, a `count:poisson` vagy a `reg:gamma`, továbbá a kategorikus elágazások, a több célváltozós modellek és az osztályozó modellek. Feltöltés előtt minden fa ellenőrzésen esik át: az elágazási jellemzőknek létezniük kell, a gyermekhivatkozásoknak körök nélkül, a fán belül kell maradniuk, és minden csomópontnak bele kell férnie a [8 bájtos csomópont-elrendezésbe](#a-predikció-működése).

A hiányzó értékek (`NaN`) minden elágazásnál az elágazás alapértelmezett irányát követik. A levélértékek összegzése a fák sorrendjében történik, és az összeg a végén adódik hozzá a kiindulási értékhez (base score), pontosan úgy, ahogy az XGBoost GPU-s prediktora is teszi, így az eredmények bitre azonosak.

Támogatott modell mentése Pythonból, és a hibaüzenet nem támogatott modell esetén:

```python
model = xgboost.XGBRegressor(objective="reg:squarederror").fit(X, y)
model.save_model("model.json")
```

```text
failed to load XGBoost regression model model.json: unsupported objective 'reg:logistic': only regression objectives without an output transformation are supported
```

## A predikció működése

A betöltő minden fát 8 bájtos csomópontokká alakít. Az összes fa egymás után, egyetlen csomóponttömbben tárolódik, és az `entry_nodes[tree]` az egyes fák gyökerének indexe. Egy fán belül a betöltő minden elágazás két gyermekét egymás mellé helyezi, így egy csomópontnak csak a bal gyermekéig mért távolságot kell tárolnia ([`regression_model.hpp`](src/regression/regression_model.hpp)):

```cpp
struct alignas(8) Node
{
    std::uint32_t link;
    float value;
};
```

A `link` három mezőt tartalmaz. A jellemzőbitek száma a modell jellemzőinek számából következik; a benchmarkmodell 19 jellemzője esetén ez 5:

```text
bit:  31 ....................... 6 | 5 .......... 1 | 0
      distance to the left child   | feature index  | default_left
```

| Mező | Elágazási csomópont | Levél |
|---|---|---|
| `link` | A bal gyermek távolsága, a jellemzőindex, valamint az, hogy a hiányzó értékek balra mennek-e | 0 |
| `value` | Küszöbérték | Levélérték |

A jobb gyermek a bal gyermeket követő csomópont. Mivel a csomópont 8 bájtos és 8 bájtra igazított, a GPU egyetlen 64 bites betöltéssel olvas be egy teljes csomópontot, és a benchmarkmodell mind a 6,7 M csomópontja 54 MB GPU-memóriát foglal.

A kernel soronként egy szálat futtat, és minden szál végigjárja az összes fát ([`regression_kernels.cu`](src/regression/regression_kernels.cu)):

```cpp
const std::uint32_t feature_mask = (1u << feature_bits) - 1u;
const std::uint32_t offset_shift = feature_bits + 1u;
float sum = 0.0f;

for (int tree = 0; tree < tree_count; ++tree)
{
    int index = entry_nodes[tree];
    Node node = nodes[index];

    while (node.link != 0)
    {
        const float value = device_features[
            static_cast<std::size_t>((node.link >> 1) & feature_mask) * rows + row];

        const bool go_right = isnan(value)
            ? (node.link & 1u) == 0
            : !(value < node.value);

        index += static_cast<int>(node.link >> offset_shift) + go_right;
        node = nodes[index];
    }

    sum += node.value;
}

device_output[row] = base_score + sum;
```

- A bemenet oszlopfolytonos, `device_features[feature * rows + row]`, így amikor egy warp 32 szála ugyanazt a jellemzőt vizsgálja, 32 egymást követő float értéket olvasnak be egyetlen tranzakcióban. Minden fa gyökerénél a warp összes szála ugyanazt a csomópontot és ugyanazt a jellemzőt olvassa.
- A hiányzó érték (`NaN`) az elágazás alapértelmezett irányát követi; egyébként `value < threshold` esetén balra halad, ahogy az XGBoostban is.
- Először a levélértékek összegzése történik meg, és az összeg csak a végén adódik hozzá a kiindulási értékhez. A lebegőpontos összeadás nem asszociatív, és az XGBoost is ezt a sorrendet használja, így a predikciók bitre azonosak.
- A kernelben nincs indexhatár-ellenőrzés: a betöltő már meggyőződött arról, hogy minden elágazási jellemző létezik, hogy a gyermekindexek a saját fájukon belül maradnak, hogy egyik fa sem tartalmaz kört, és hogy minden távolság elfér a jellemzőbitek mellett.
- A kernel blokkonként 1024 szállal indul; 16 regisztert használ, így két blokk kitölt egy streaming multiprocesszort.

## Példák

A példák a `bin/examples/` könyvtárba települnek.

| Futtatható állomány | Mit mutat be |
|---|---|
| `echter_xgb_cpu_prediction <model.json> [rows]` | Bemenet és kimenet a host memóriájában |
| `echter_xgb_cuda_prediction <model.json> [rows]` | Eszközpuffer feltöltése CUDA-val és predikció a GPU-n |
| `echter_xgb_cudf_prediction <model.json> [rows]` | Predikció egy `cudf::table_view` objektumból, a kimenet statisztikáival |
| `echter_xgb_csv_prediction <model.json> <input.csv> [options]` | A parancssori alkalmazás tömör változata |

Az első három `rows` sornyi véletlenszerű bemenetet generál (alapértelmezés: 100 000), és kiírja az első tíz predikciót.

```bash
./bin/examples/echter_xgb_cuda_prediction model.json 1000000
./bin/examples/echter_xgb_cudf_prediction model.json 1000000
./bin/examples/echter_xgb_csv_prediction model.json input.csv --target-columns 1
```

## Tesztek

```bash
./bin/test/echter_xgb_test                           # built-in tests
./bin/test/echter_xgb_test model.json [rows]         # built-in tests and a real model
```

A beépített tesztek egy kis XGBoost-modellt és CSV-fájlokat hoznak létre az ideiglenes könyvtárban, és a következőket ellenőrzik:

- a pontos predikciókat, beleértve a hiányzó értékeket, az üres CSV-mezőket és az összegzési sorrendet,
- a host memóriájából és az eszközmemóriából induló predikciós utakat, valamint a CSV-fájlok kiírását és visszaolvasását,
- a csomópontok kódolását egy 1000 jellemzős modell 999-es jellemzőjén végzett elágazással,
- az érvénytelen modellek elutasítását (nem támogatott célfüggvény, hibás gyermekhivatkozások, körök, ismeretlen elágazási jellemzők, a csomópont-elrendezésbe nem férő fák),
- a memóriabiztonságot (mozgatás után kiürült forrásobjektumok, valamint eszközadatként átadott, host memóriára mutató pointerek).

Modellfájllal futtatva a teszt azt is ellenőrzi, hogy véletlenszerű sorokra a host memóriájából és az eszközmemóriából számolt predikciók megegyeznek-e, és végesek-e. Ha ezt a futtatást a `ctest` futtatásai közé is fel szeretnénk venni, konfiguráljunk a `-DECHTER_XGB_TEST_MODEL=model.json` opcióval.

```text
$ ./bin/test/echter_xgb_test model.json 100000
  model file: model.json (19 features, 500 trees, 100000 rows)
Echter XGBoost tests passed (36 checks)
```

## Összevetés az XGBoosttal

A [`scripts/predict.py`](scripts/predict.py) ugyanazt a CSV-feldolgozási láncot futtatja a pandas és az XGBoost Python-csomag segítségével a GPU-n. A kimenete összevethető az `echter_xgb_predict` kimenetével, és ugyanolyan bontásban írja ki az egyes lépések idejét. Az `xgboost` és a `pandas` csomagra van szüksége:

```bash
python scripts/predict.py model.json input.csv --target-column incident_proton_energy
```

A jellemzőoszlopok kiválasztása a modell jellemzőnevei alapján történik. A `--key-column` (alapértelmezés: az első oszlop) `id`, a `--target-column` pedig `target` néven íródik ki. A kimenet az `outputs/<input-stem>_xgboost_predictions.csv` fájlba kerül, hacsak nem adjuk meg az `--output` kapcsolót.

### Benchmarkok

A [`scripts/benchmarks`](scripts/benchmarks) könyvtárban vannak a [Teljesítmény](#teljesítmény) szakasz táblázatai mögött álló programok. Ezek egy különálló CMake-projektet alkotnak, amelyet a fő build nem fordít le:

| Program | Mit mér |
|---|---|
| `echter_benchmark` | Az Echter XGBoost négy bemeneti útját; a C API-s benchmark számára a jellemzőket is kiírja |
| `xgboost_capi_benchmark` | Az XGBoost C API-ját a host memóriájában és a GPU-n lévő bemenettel; a `libxgboost.so` fájlt a megadott útvonalról tölti be |
| `xgboost_python_benchmark.py` | Az XGBoost Python-csomagját NumPy-bemenettel |
| `cuml_fil_benchmark.py` | A cuML FIL-t CuPy-bemenettel, az alapértelmezett beállításokkal, valamint a leggyorsabb elrendezéssel és chunkmérettel; kiírja, hány predikció egyezik az Echter predikcióival |

Minden program kötegméretenként kiírja a predikcióit, így az eredmények bájtról bájtra összevethetők:

```bash
cmake -S scripts/benchmarks -B build-benchmarks -G Ninja
cmake --build build-benchmarks

for n in 10 20; do
    head -1 input.csv > input_x$n.csv
    for i in $(seq $n); do tail -n +2 input.csv >> input_x$n.csv; done
done

./build-benchmarks/echter_benchmark model.json results input.csv input_x10.csv input_x20.csv
./build-benchmarks/xgboost_capi_benchmark \
    "$(python -c 'import os, xgboost; print(os.path.join(os.path.dirname(xgboost.__file__), "lib", "libxgboost.so"))')" \
    model.json results
python scripts/benchmarks/xgboost_python_benchmark.py model.json input.csv results

python -m venv fil-env
fil-env/bin/pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu13==26.2.*" xgboost pandas
fil-env/bin/python scripts/benchmarks/cuml_fil_benchmark.py model.json input.csv results

for f in 1 10 20; do
    cmp results/echter_x$f.bin results/xgboost_capi_x$f.bin
    cmp results/echter_x$f.bin results/xgboost_python_x$f.bin
done
```

## A projekt felépítése

```text
apps/          echter_xgb_predict command-line application
bin/           prebuilt application; bin/lib/ holds libEchter_XGBoost.so, bin/test/ the test, bin/examples/ the examples
examples/      CPU, CUDA, cuDF, and CSV examples
scripts/       XGBoost reference script; benchmarks/ holds the benchmark programs
src/
├── core/        device memory (RAII buffers), CUDA error checks, cuDF conversion
├── io/          echter.xgb.io module and its cuDF CSV backend
├── modules/     echter.xgb, echter.xgb.data, echter.xgb.reg modules
└── regression/  model loader, CUDA kernel, runtime, cuDF adapter
tests/         self-contained tests
```

A forrásfájlok egységes konvenciót követnek:

| Kiterjesztés | Tartalom |
|---|---|
| `.hpp` | API-fejlécfájlok; soha nem emelnek be CUDA- vagy cuDF-fejlécfájlokat, így a modulegységek is használhatják őket |
| `.cuh` | Fejlécfájlok, amelyek CUDA- (vagy cuDF-) fejlécfájlokat emelnek be, és kerneleket deklarálnak |
| `.cu` | Kernelimplementációk; csak ezeket a fájlokat fordítja az `nvcc` |
| `.cpp` | A hoston futó kód, beleértve a CUDA-futtatókörnyezetet vagy a cuDF-et hívó kódot is |

A predikciós kernel például a `regression_kernels.hpp` (az indítófüggvény), a `regression_kernels.cuh` (a CUDA-fejlécek beemelése és a kernel deklarációja) és a `regression_kernels.cu` (maga a kernel) fájlra oszlik:

```cpp
// regression_kernels.hpp: the API, no CUDA headers
void launch_regression_kernel(
    const float* device_features, float* device_output, std::size_t rows, const DeviceModel& model);

// regression_kernels.cuh: CUDA headers and the kernel
#include <cuda_runtime.h>
__global__ void regression_predict_kernel(
    const float* __restrict__ device_features, float* __restrict__ device_output, std::size_t rows,
    const Node* __restrict__ nodes, const int* __restrict__ entry_nodes, int tree_count, float base_score,
    std::uint32_t feature_bits);
```

A modulszerkezetben a `regression/` mellett hely marad az osztályozás támogatásának.

## Licenc

Copyright 2026 Tamás Menyhárt

A projekt az [Apache License, Version 2.0](LICENSE) licenc alatt érhető el. A továbbterjesztett példányokban kötelezően megtartandó szerzőségi megjelölést a [`NOTICE`](NOTICE) fájl tartalmazza.
