#Szükséges importok
import io
import logging
import os
import re
import shutil
import tarfile
import tempfile

import fasteners
import numpy as np
import torch

from tqdm import tqdm
from ase import Atoms
from ase.io.extxyz import read_xyz

from schnetpack.data import (
    create_dataset,
    load_dataset,
    AtomsDataModuleError,
)
import schnetpack.properties as structure


def qm9_download_file(qm9data, file_id, destination):
    """
    Letölti a QM9 adatbázis egy adott fájlját a FigShare szervereiről.

    A QM9 adatbázist bemutató publikáció:
    Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp,
    O. Anatole von Lilienfeld,
    Quantum Chemistry Structures and Properties of 134 Kilo Molecules.
    Scientific Data 2014, 1, 140022.
    DOI: 10.1038/sdata.2014.22.

    """

    # A fájlok letöltéséhez használható hivatalos FigShare URL-ek. Több URL-t is megadtam
    base_urls = [
        "https://ndownloader.figshare.com/files/",
        "https://springernature.figshare.com/ndownloader/files/",
    ]

    # A QM9 adatbázis egyes fájljaihoz tartozó FigShare-azonosítók.
    file_ids = {
        "data": "3195389",
        "atomrefs": "3195395",
        "uncharacterized": "3195404",
    }

    # Végigiterálunk a lehetséges letöltési forrásokon.
    for base_url in base_urls:
        # Ebből kapjuk meg a tényleges letöltési URL-t.
        url = f"{base_url}{file_id}"

        try:
            # Importáljuk az urllib request modulját,
            # amely HTTP-kérések és fájlletöltések kezelésére szolgál.
            from urllib import request

            # Letöltjük a megadott URL-en található fájlt. A destination alapértelmezetten cwd
            request.urlretrieve(url, destination)

            # Ha a letöltés sikeres, kilépünk a függvényből.
            # Így a következő URL-t már nem próbáljuk meg.
            return

        except Exception:
            # Ha a letöltés sikertelen, figyelmeztetést írunk a logba.
            # A ciklus ezután továbblép a következő letöltési forrásra.
            logging.warning(
                f"Could not download from {url}, trying next source..."
            )

    # Ha egyik URL-ről sem sikerült letölteni a fájlt,
    raise AtomsDataModuleError(
        f"Could not download file with id {file_id} from any source."
    )


def qm9_download_atomrefs(qm9data, tmpdir):
    """
    Letölti és feldolgozza a QM9 adatbázis atomreferencia-értékeit.
    """
    logging.info("Downloading GDB-9 atom references...")

    # Létrehozza az ideiglenes fájl teljes elérési útját.

    tmp_path = os.path.join(tmpdir, "atomrefs.txt")
    # Meghívja a korábban definiált qm9_download_file függvényt. 
    qm9_download_file(
        qm9data,
        qm9data.file_ids["atomrefs"],
        tmp_path,
    )
    # Naplózza a sikeres letöltést.
    logging.info("Done.")

    # Összegyűjti azokat a molekuláris tulajdonságokat,
    # amelyekhez atomreferencia-értékeket szeretnénk rendelni.
    #
    # Ezek a QM9 adatbázis különböző kvantumkémiai tulajdonságai.
    props = [
        qm9data.zpve,  # Zérusponti rezgési energia (Zero Point Vibrational Energy)
        qm9data.U0,    # Belső energia 0 K-en
        qm9data.U,     # Belső energia 298.15 K-en
        qm9data.H,     # Enthalpia 298.15 K-en
        qm9data.G,     # Gibbs-féle szabadenergia 298.15 K-en
        qm9data.Cv,    # Hőkapacitás állandó térfogaton
    ]

    # Minden tulajdonsághoz létrehoz egy 100 elemű,
    # kezdetben nullákat tartalmazó NumPy-tömböt.
    #
    # A kulcs a tulajdonság azonosítója (pl. qm9data.U0),
    # az érték pedig egy 100 elemű NumPy-tömb.
    #
    # A tömb indexe az atom rendszámának (Z) felel meg.
    atref = {p: np.zeros(100) for p in props}

    # Megnyitja a letöltött atomreferencia-fájlt olvasásra.
    with open(tmp_path) as f:

        # Beolvassa a fájl összes sorát egy listába.
        # Minden listaelem a fájl egy-egy sora lesz.
        lines = f.readlines()

        # Az első öt sort kihagyja, majd az 5–9. indexű sorokat dolgozza fel.
        #
        # A zip() segítségével minden sorhoz hozzárendeli
        # a megfelelő atom rendszámát:
        #
        # 1 -> H (hidrogén)
        # 6 -> C (szén)
        # 7 -> N (nitrogén)
        # 8 -> O (oxigén)
        # 9 -> F (fluor)
        for z, line in zip([1, 6, 7, 8, 9], lines[5:10]):

            # Végigiterál a hat vizsgált molekuláris tulajdonságon.
            for i, prop in enumerate(props):

                # Feldolgozza az aktuális sort, és kiolvassa az adott tulajdonság atomreferencia-értékét.
                atref[prop][z] = float(line.split()[i + 1])

    # A NumPy-tömböket Python-listákká alakítja.
    #
    # .items():
    #   A szótár kulcs-érték párjain iterál.
    #
    # .tolist():
    #   A NumPy-tömböt hagyományos Python-listává alakítja.
    #
    # Az eredmény egy olyan szótár, amelyben minden tulajdonsághoz
    # egy 100 elemű lista tartozik.
    return {k: v.tolist() for k, v in atref.items()}


def qm9_download_uncharacterized(qm9data, tmpdir):
    """
    Letölti az "uncharacterized"  molekulák indexeit tartalmazó fájlt,
    majd kiolvassa belőle az indexeket.
    """
    logging.info("Downloading list of uncharacterized molecules...")

    # Összeállítja az ideiglenes fájl teljes elérési útját.
    tmp_path = os.path.join(tmpdir, "uncharacterized.txt")

    # Meghívja a korábban definiált qm9_download_file() függvényt.
    qm9_download_file(
        qm9data,
        qm9data.file_ids["uncharacterized"],
        tmp_path,
    )

    # Naplózza, hogy a letöltés befejeződött.
    logging.info("Done.")

    # Létrehoz egy üres Python-listát. (nem túl hatékony, de a fájl mérete sem túl nagy)
    uncharacterized = []

    # Megnyitja a letöltött fájlt olvasásra.
    with open(tmp_path) as f:

        lines = f.readlines()

        # A fájl 10. sorától az utolsó sor előtti sorig iterál.

        for line in lines[9:-1]:
        #Megfelelő formátumban kiolvassa az indexeket, és hozzáadja a listához.
            uncharacterized.append(int(line.split()[0]))

    # Visszaadja a nem karakterizált molekulák indexeit tartalmazó listát.
    return uncharacterized


def qm9_download_data(
    qm9data,
    tmpdir,
    dataset,
    uncharacterized=None,
):
    """
    Letölti, kicsomagolja, feldolgozza és eltárolja
    a GDB-9 adatbázis geometriai adatait (amelyen a QM9 alapul)
    """

    logging.info("Downloading GDB-9 data...")

    # A letöltött tömörített fájl elérési útja.
    # A QM9 adatbázis TAR.GZ formátumban érhető el.
    #Ne felejtsük el, hogy a GDB-9 maga az adatbázis, még a QM9 a fizikai-kémiai tulajdonságot összegzi.
    tar_path = os.path.join(tmpdir, "gdb9.tar.gz")

    # A kicsomagolt XYZ-fájlok tárolási könyvtára.
    raw_path = os.path.join(tmpdir, "gdb9_xyz")

    # Letölti a QM9 adatbázist a korábban definiált függvénnyel.
    #
    # qm9data.file_ids["data"]:
    #   A teljes QM9 adatbázis FigShare-azonosítója.
    #
    # tar_path:
    #   A letöltött archívum mentési helye.
    qm9_download_file(
        qm9data,
        qm9data.file_ids["data"],
        tar_path,
    )

    # Jelzi a letöltés befejezését.
    logging.info("Done.")

    # Jelzi, hogy megkezdődik az archívum kicsomagolása.
    logging.info("Extracting files...")

    # Megnyitja a TAR.GZ archívumot olvasásra.
    #
    # A tarfile modul segítségével kezelhetők
    # a TAR formátumú archívumok.
    with tarfile.open(tar_path) as tar:

        # Az archívum teljes tartalmát kicsomagolja
        # a raw_path könyvtárba.
        tar.extractall(raw_path)

    # Jelzi a kicsomagolás befejezését.
    logging.info("Done.")

    # Megkezdődik az egyedi XYZ-fájlok feldolgozása.
    logging.info("Parse xyz files...")

    # Lekéri a raw_path könyvtárban található fájlok neveit,
    # majd megfelelő sorrendbe rendezi őket.
    ordered_files = sorted(
        os.listdir(raw_path),

        # Egyedi rendezési kulcsot definiál.
        # re.sub(r"\D", "", x):
        # Eltávolít minden nem számjegy karaktert a fájlnévből.
        key=lambda x: (int(re.sub(r"\D", "", x)), x),
    )

    # Létrehoz egy üres listát.
    # Ebbe kerülnek majd az egyes molekulákhoz tartozó
    # fizikai tulajdonságok és atomisztikus adatok.
    property_list = []

    # Létrehoz egy egész számokat tartalmazó NumPy-tömböt.
    # np.arange(len(ordered_files)):
    irange = np.arange(len(ordered_files), dtype=int)

    # Ellenőrzi, hogy megadtunk-e nem karakterizált
    # molekulákhoz tartozó indexeket.
    if uncharacterized is not None:

        # Kiszűri a nem karakterizált molekulák indexeit. Ez egy nagyon fontos lépés, mivel az adatbázisban lehetnének ilyen elemek!
        #
        # np.array(uncharacterized, dtype=int):
        #   A bemeneti listát egész számokat tartalmazó
        #   NumPy-tömbbé alakítja.
        #
        # - 1:
        #   Átalakítja az indexeket 1-alapú számozásról
        #   0-alapú Python-indexelésre.
        #
        # np.setdiff1d():
        #   Az irange azon elemeit adja vissza,
        #   amelyek nem szerepelnek a második tömbben.
        #
        # Tehát a nem karakterizált molekulák kimaradnak.
        irange = np.setdiff1d(
            irange,
            np.array(uncharacterized, dtype=int) - 1,
        )

    # Végigiterál a feldolgozandó molekulák indexein.
    #
    # A tqdm() egy "progress bart" jelenít meg,
    # így látható a feldolgozás előrehaladása.
    for i in tqdm(irange):

        # Az aktuális molekulához tartozó XYZ-fájl
        # teljes elérési útjának létrehozása.
        xyzfile = os.path.join(raw_path, ordered_files[i])

        # Létrehoz egy üres szótárat.
        # Ebben tároljuk az aktuális molekula tulajdonságait.
        properties = {}

        # Létrehoz egy memóriában tárolt szöveges fájlt.
        #
        # A StringIO lehetővé teszi, hogy szöveget fájlként
        # kezeljünk anélkül, hogy új fájlt írnánk a lemezre.
        tmp = io.StringIO()

        # Megnyitja az aktuális XYZ-fájlt olvasásra.
        with open(xyzfile, "r") as f:

            # Beolvassa az összes sort egy listába.
            lines = f.readlines()

            # A második sorból kinyeri a harmadik oszloptól
            # kezdődő értékeket.
            #
            # Ezek a QM9 molekuláris tulajdonságai.
            values = lines[1].split()[2:]

            # Összerendeli az adatbázisban elérhető tulajdonságokat
            # a fájlból kiolvasott numerikus értékekkel.

            for prop, value in zip(
                dataset.available_properties,
                values,
            ):
                # A szöveges értéket lebegőpontos számmá alakítja,
 
                properties[prop] = np.array([float(value)])

            # Végigmegy az XYZ-fájl összes során.
            for line in lines:
                # A QM9 fájlokban előforduló *^ jelölést
                # hagyományos tudományos jelöléssé alakítja.
                #
                # Például:
                # 1.234*^-3 -> 1.234e-3
                #
                # Ez azért szükséges, mert a numerikus
                # adatfeldolgozó rutinok az e jelölést kezelik.
                #
                # Az átalakított sorokat a memóriában lévő
                # szöveges fájlba írja.
                tmp.write(line.replace("*^", "e"))

        # A StringIO objektum pozícióját visszaállítja
        # a szöveg elejére.
        #
        # Erre azért van szükség, mert az előző ciklus
        # a szöveg végére állította az írási pozíciót.
        tmp.seek(0)

        # Az Atoms objektum tartalmazza az atomok
        # rendszámát, koordinátáit és egyéb szerkezeti adatait.
        ats: Atoms = list(read_xyz(tmp, 0))[0]

        # Az atomok rendszámának eltárolása.
        #
        # ats.numbers:
        #   Az atomok rendszámát tartalmazó tömb.
        #
        # Például egy H2O molekulánál:
        # [1, 8, 1]
        properties[structure.Z] = ats.numbers

        # Az atomok Descartes-koordinátáinak eltárolása.
        # Minden sor egy atom x, y, z koordinátáját tartalmazza.
        properties[structure.R] = ats.positions

        # A periodikus cella mátrixának eltárolása. Adott esetben kristályokat is lehetne számítani így, de ez a QM9 adatbázisban nem releváns, mivel molekulák vannak benne..
        # Örökli az Atoms objektum celláját, amely a periodikus cella vektorait tartalmazza.
        # ats.cell:
        properties[structure.cell] = ats.cell
        properties[structure.pbc] = ats.pbc

        # Az aktuális molekula teljes tulajdonságszótárát
        # hozzáadja a property_list listához.
        #
        # Így a lista minden eleme egy molekulát reprezentál.
        property_list.append(properties)

    # Naplózza az adatbázisba történő írás megkezdését.
    logging.info("Write atoms to database...")

    # Az összes feldolgozott molekula adatainak
    # hozzáadása a dataset objektumhoz.
    #
    # property_list:
    #   A feldolgozott molekulák tulajdonságait tartalmazó lista.
    #
    # A dataset.add_systems() a háttérben elvégzi
    # az adatok adatbázisba történő mentését.
    dataset.add_systems(property_list=property_list)

    # Naplózza a teljes folyamat sikeres befejezését.
    logging.info("Done.")


def prepare_qm9_data(qm9data):
    """
    A QM9 adatkészlet előkészítése.

    A függvény a qm9data objektum konfigurációja alapján:
    - létrehozza az adatbázist, ha még nem létezik;
    - letölti a QM9 molekulákat és az atomreferencia-értékeket, ha nincs még letöltve;
    - opcionálisan eltávolítja a nem karakterizált molekulákat;
    - vagy betölti a már létező adatbázist és ellenőrzi annak tartalmát.
    """

    # Ellenőrizzük, hogy létezik-e már a QM9 adatbázis.
    # Ha nem, akkor el kell végezni a letöltést és az adatbázis létrehozását.
    if not os.path.exists(qm9data.datapath):

        # Az egyes QM9 molekulatulajdonságokhoz tartozó mértékegységeket
        # rendeljük hozzá.
        #
        property_unit_dict = {
            qm9data.A: "GHz",          # A fő tehetetlenségi nyomatékhoz kapcsolódó rotációs állandó
            qm9data.B: "GHz",          # B rotációs állandó
            qm9data.C: "GHz",          # C rotációs állandó
            qm9data.mu: "Debye",       # Dipólusmomentum
            qm9data.alpha: "a0 a0 a0", # Izotróp polarizálhatóság
            qm9data.homo: "Ha",        # HOMO-energia
            qm9data.lumo: "Ha",        # LUMO-energia
            qm9data.gap: "Ha",         # HOMO-LUMO energiarés
            qm9data.r2: "a0 a0",       # Elektronsűrűség második momentuma
            qm9data.zpve: "Ha",        # Zérusponti rezgési energia
            qm9data.U0: "Ha",          # Belső energia 0 K-en
            qm9data.U: "Ha",           # Belső energia 298.15 K-en
            qm9data.H: "Ha",           # Entalpia
            qm9data.G: "Ha",           # Gibbs-szabadenergia
            qm9data.Cv: "cal/mol/K",   # Hőkapacitás
        }

        # Ideiglenes könyvtár létrehozása.
        # A letöltött kiegészítő fájlok (pl. atomreferenciák) itt tárolódnak.
        tmpdir = tempfile.mkdtemp("qm9")

        try:
            # Az atomreferencia-értékek letöltése.
            # Ezek az atomonkénti referencia-hozzájárulások például
            atomrefs = qm9_download_atomrefs(qm9data, tmpdir)

            # Az adatbázis inicializálása a megadott konfigurációval.
            #
            # distance_unit="Ang": a geometriai koordináták egysége Ångström.
            # property_unit_dict: az egyes molekulatulajdonságok egységei.
            # atomrefs: atomreferencia-értékek.
            dataset = create_dataset(
                datapath=qm9data.datapath,
                format=qm9data.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
                atomrefs=atomrefs,
            )

            # Eldöntjük, hogy el kell-e távolítani a nem karakterizált
            # molekulákat a QM9 adatbázisból.
            if qm9data.remove_uncharacterized:

                # A nem karakterizált molekulák listájának letöltése.
                # Ez a lista azokat a molekulákat azonosítja,
                # amelyek nem rendelkeznek megfelelően meghatározott
                # tulajdonságértékekkel.
                uncharacterized = qm9_download_uncharacterized(
                    qm9data, tmpdir
                )
            else:
                # Ha nem kérjük az eltávolítást, akkor nincs kizárási lista.
                uncharacterized = None

            # A tényleges QM9 molekulák letöltése és adatbázisba írása.
            # Ha az uncharacterized lista nem None, akkor a felsorolt
            # molekulák kimaradnak az adatbázisból.
            qm9_download_data(
                qm9data,
                tmpdir,
                dataset,
                uncharacterized=uncharacterized,
            )

        finally:
            # Az ideiglenes könyvtár törlése akkor is megtörténik,
            # ha a try blokkban kivétel keletkezik.
            shutil.rmtree(tmpdir, ignore_errors=True)

    else:

        # Ha az adatbázis már létezik, nem töltjük le újra,
        # hanem betöltjük a megadott formátumban.
        dataset = load_dataset(
            qm9data.datapath,
            qm9data.format,
        )

        # Ellenőrizzük, hogy az adatbázis tartalmazza-e a nem
        # karakterizált molekulákat.
        #
        # A teljes QM9 adatbázis 133 885 molekulát tartalmaz.
        # Ebből 3054 molekula nem karakterizált.
        if qm9data.remove_uncharacterized and len(dataset) == 133885:

            # A konfiguráció szerint el akarjuk távolítani a nem
            # karakterizált molekulákat, de az adatbázisban mégis
            # a teljes, 133 885 molekulás adathalmaz található.
            #
            # Ez inkonzisztens állapot, ezért hibát dobunk.
            raise AtomsDataModuleError(
                "The dataset contains the uncharacterized 3054 molecules. "
                "Choose a different location or set "
                "remove_uncharacterized=False!"
            )

        elif not qm9data.remove_uncharacterized and len(dataset) < 133885:

            # A konfiguráció szerint meg akarjuk tartani az összes
            # molekulát, de az adatbázis kevesebb molekulát tartalmaz,
            # mint a teljes QM9 adathalmaz.
            #
            # Ez arra utal, hogy korábban a nem karakterizált molekulák
            # eltávolításával hozták létre az adatbázist.
            raise AtomsDataModuleError(
                "The dataset does NOT contain the uncharacterized "
                "3054 molecules. Choose a different location or set "
                "remove_uncharacterized=True!"
            )

#Segédfüggvény 
def qm9_log(qm9data, msg):
    """
    DEBUG üzenetet segítő függvény.

    Ha rendelkezésre áll trainer objektum, akkor a logüzenethez
    hozzáfűzi a globális és lokális processzrangot.
    Ellenkező esetben egyszerű logüzenetet ír.
    """
    trainer = getattr(qm9data, "trainer", None)

    # Ellenőrizzük, hogy létezik-e trainer objektum.
    if trainer is not None:

        # DEBUG szintű logüzenet létrehozása.
        logging.debug(
            "Global rank: %s, local rank: %s >> %s",
            trainer.global_rank,
            trainer.local_rank,
            msg,
        )

    else:
        # Ha nincs trainer objektum, akkor nincs elérhető
        # információ az elosztott processz rangjáról.
        # Egyszerű DEBUG üzenetet írunk.
        logging.debug(">> %s", msg)


#Segédfüggvény
def qm9_copy_to_workdir(qm9data):
    """
    A QM9 adatbázis átmásolása egy lokális munkakönyvtárba.

    A függvény:
    - létrehozza a munkakönyvtárat, ha még nem létezik;
    - processzek közötti lockot használ a konkurens fájlműveletek
      elkerülésére;
    - átmásolja az adatbázist, ha az még nem található meg a
      munkakönyvtárban;
    - alaphelyzetbe állítja a dataset objektumokat és állapotjelzőket;
    - visszaadja az új adatbázis elérési útját.
    """

    os.makedirs(qm9data.data_workdir, exist_ok=True)

    # Az eredeti adatbázis fájlnevét kinyerjük az elérési útból.

    name = os.path.basename(qm9data.datapath)

    # Meghatározzuk az új adatbázis elérési útját.
 
    datapath = os.path.join(
        qm9data.data_workdir,
        name,
    )


    #A parelell futtatás miatt szükség van az alábbi elemekre
    lock = fasteners.InterProcessLock(
        os.path.join(
            qm9data.data_workdir,
            f"dataworkdir_{name}.lock",
        )
    )

    with lock:

        # Naplózzuk, hogy a folyamat belépett a zárolt szekcióba.
        qm9_log(qm9data, "Enter lock")

        # Ellenőrizzük, hogy az adatbázis már megtalálható-e
        # a lokális munkakönyvtárban.
        if not os.path.exists(datapath):

            # Ha még nem létezik, naplózzuk a másolási műveletet.
            qm9_log(qm9data, "Copy data to data workdir")

            # Az eredeti QM9 adatbázis átmásolása a lokális könyvtárba.
            #
            # A forrás:
            # qm9data.datapath
            #
            # A cél:
            # datapath
            shutil.copy(qm9data.datapath, datapath)


        # Dataset objektumok alaphelyzetbe állítása
        # A korábban betöltött teljes dataset referenciáját töröljük.
        #
        # None érték beállításával jelezzük, hogy az adatbázist
        # újra be kell tölteni.
        qm9data.dataset = None
        qm9data._train_dataset = None
        qm9data._val_dataset = None
        qm9data._test_dataset = None

        # A teardown állapotjelzők visszaállítása.
        #
        # Ezek azt jelzik, hogy az adott adathalmazhoz tartozó
        # teardown (felszabadítási/takarítási) művelet
        # megtörtént-e már.
        qm9data._has_teardown_val = False
        qm9data._has_teardown_test = False

    # A with blokk végén a lock automatikusan felszabadul.
    # Ez akkor is megtörténik, ha kivétel keletkezik a blokkban.
    qm9_log(qm9data, "Exited lock")

    # Visszaadjuk a lokális munkakönyvtárban található adatbázis
    # elérési útját.
    return datapath

#Segédfüggvény, ami a különböző partíciókat állítja be a train/val/test adathalmazokra
def qm9_load_partitions(qm9data):
    """Load or generate train/validation/test partitions."""

    lock = fasteners.InterProcessLock("splitting.lock")

    with lock:

        qm9_log(qm9data, "Enter splitting lock")

        total_size = len(qm9data.dataset)

        # Convert relative partition sizes to absolute sizes.

        if isinstance(qm9data.num_train, float) and qm9data.num_train <= 1:
            qm9data.num_train = int(
                qm9data.num_train * total_size
            )

        if isinstance(qm9data.num_val, float) and qm9data.num_val <= 1:
            qm9data.num_val = int(
                qm9data.num_val * total_size
            )

        if isinstance(qm9data.num_test, float) and qm9data.num_test <= 1:
            qm9data.num_test = int(
                qm9data.num_test * total_size
            )

        # Load existing split.

        if (
            qm9data.split_file is not None
            and os.path.exists(qm9data.split_file)
        ):

            qm9_log(qm9data, "Load split")

            S = np.load(qm9data.split_file)

            qm9data.train_idx = S["train_idx"].tolist()
            qm9data.val_idx = S["val_idx"].tolist()
            qm9data.test_idx = S["test_idx"].tolist()

            # Validate split sizes.

            for name, indices in [
                ("num_train", qm9data.train_idx),
                ("num_val", qm9data.val_idx),
                ("num_test", qm9data.test_idx),
            ]:

                expected = getattr(qm9data, name)

                if expected and expected != len(indices):

                    raise ValueError(
                        f"Split file was given, but "
                        f"`{name} ({expected}) != len(indices) "
                        f"({len(indices)})`!"
                    )

        else:

            qm9_log(qm9data, "Create split")

            if (
                qm9data.num_train is None
                or qm9data.num_val is None
            ):
                raise AtomsDataModuleError(
                    "If no split_file is given, the sizes of the "
                    "training and validation partitions need to be set!"
                )

            (
                qm9data.train_idx,
                qm9data.val_idx,
                qm9data.test_idx,
            ) = qm9data.splitting.split(
                qm9data.dataset,
                qm9data.num_train,
                qm9data.num_val,
                qm9data.num_test,
            )

            if qm9data.split_file is not None:

                qm9_log(qm9data, "Save split")

                np.savez(
                    qm9data.split_file,
                    train_idx=qm9data.train_idx,
                    val_idx=qm9data.val_idx,
                    test_idx=qm9data.test_idx,
                )

    qm9_log(qm9data, "Exit splitting lock")


#Segédfüggvény a különböző transzformációk inicializálására és hozzárendelésére a train/val/test adathalmazokhoz
def qm9_setup_transforms(qm9data):
    """
    Inicializálja a transzformációkat, majd hozzárendeli
    őket a training, validation és test datasetekhez.
    """

    # Végigiterálunk a training datasethez definiált transzformációkon.
    for transform in qm9data.train_transforms:

        # Az adott transzformációhoz hozzárendeljük a teljes
        # QM9 datamodule objektumot.

        transform.datamodule(qm9data)

    # Ugyanezt elvégezzük a validation transzformációkkal.
    for transform in qm9data.val_transforms:

        # A validation transzformáció inicializálása.
        transform.datamodule(qm9data)

    # Végül inicializáljuk a test transzformációkat.
    for transform in qm9data.test_transforms:

        # A test transzformáció inicializálása.
        transform.datamodule(qm9data)

    # A inicializált transzformációs listát hozzárendeljük
    # a training datasethez.
    #
    # Ettől kezdve a dataset adatlekérdezései során
    # a train_transforms által meghatározott műveletek
    # alkalmazhatók.
    qm9data._train_dataset.transforms = qm9data.train_transforms

    # A validation datasethez tartozó transzformációk beállítása.
    qm9data._val_dataset.transforms = qm9data.val_transforms

    # A test datasethez tartozó transzformációk beállítása.
    qm9data._test_dataset.transforms = qm9data.test_transforms


# ============================================================
# Main replacement for qm9data.setup()
# ============================================================

def setup_qm9_data(qm9data, stage=None):
    """
    A qm9data.setup() metódus önálló implementációja.

    Feladata:
    - Az adatállomány elérési útjának meghatározása.
    - A QM9 adatkészlet betöltése.
    - A train/validation/test indexek betöltése vagy generálása.
    - A három adathalmaz-részhalmaz létrehozása.
    - Az adatok előfeldolgozó transzformációinak konfigurálása.

    Parameters
    ----------
    qm9data : object
        A QM9 adatkészlet konfigurációját és adatait tároló objektum.
    stage : str, optional
        A végrehajtás aktuális szakasza (pl. train, validate, test).
        Ebben az implementációban jelenleg nincs felhasználva.
    """

    # ---------------------------------------------------------
    # 1. Az adatkészlet elérési útjának meghatározása
    # ---------------------------------------------------------

    # Ha nincs külön munkakönyvtár megadva, az eredeti adatútvonalat használjuk.
    if qm9data.data_workdir is None:
        datapath = qm9data.datapath

    # Ha van munkakönyvtár, az adatokat először oda másoljuk.
    else:
        datapath = qm9_copy_to_workdir(qm9data)

    # ---------------------------------------------------------
    # 2. Az adatkészlet betöltése
    # ---------------------------------------------------------

    # Csak akkor töltjük be az adatokat, ha azok még nincsenek
    # eltárolva a qm9data objektumban.
    if qm9data.dataset is None:

        # Az adatok beolvasása a megadott fájlból vagy könyvtárból.
        qm9data.dataset = load_dataset(
            datapath,  # Az adatállomány elérési útja.
            qm9data.format,  # Az adatfájl formátuma.
            property_units=qm9data.property_units,  # A fizikai tulajdonságok mértékegységei.
            distance_unit=qm9data.distance_unit,  # A geometriai távolságok mértékegysége.
            load_properties=qm9data.load_properties,  # A betöltendő fizikai tulajdonságok.
        )

        # -----------------------------------------------------
        # 3. Train/validation/test particionálás
        # -----------------------------------------------------

        # Ha a tanítóhalmaz indexei még nincsenek definiálva,
        # betöltjük vagy létrehozzuk az adatpartíciókat.
        if qm9data.train_idx is None:
            qm9_load_partitions(qm9data)

        # -----------------------------------------------------
        # 4. A részhalmazok létrehozása
        # -----------------------------------------------------

        # A teljes adatkészletből kiválasztjuk a tanítóadatokat.
        qm9data._train_dataset = qm9data.dataset.subset(
            qm9data.train_idx
        )

        # A validációs adathalmaz létrehozása.
        qm9data._val_dataset = qm9data.dataset.subset(
            qm9data.val_idx
        )

        # A teszthalmaz létrehozása.
        qm9data._test_dataset = qm9data.dataset.subset(
            qm9data.test_idx
        )

        # -----------------------------------------------------
        # 5. Adatelőfeldolgozó transzformációk konfigurálása
        # -----------------------------------------------------

        # Beállítjuk az adatokon alkalmazandó transzformációkat,
        # például koordináta-, távolság- vagy tulajdonságtranszformációkat.
        qm9_setup_transforms(qm9data)