# Computer graphics homework No.1

## Description HU

Készítsen programot, amely egy véletlen gráfot esztétikusan megjelenít és lehetőséget ad a felhasználónak annak tetszőleges részének kinagyítására, mialatt a maradék rész még mindig látszik. A gráf 50 csomópontból áll, telítettsége 5%-os (a lehetséges élek 5% valódi él). Az esztétikus elrendezés érdekében a csomópontok helyét egyrészt heurisztikával, másrészt a hiperbolikus sík szabályainak megfelelő erő-vezérelt gráfrajzoló algoritmussal kell meghatározni a SPACE lenyomásának hatására. 

A fókuszálás érdekében a gráfot a hiperbolikus síkon kell elrendezni és a Beltrami-Klein módszerrel a képernyőre vetíteni. A fókuszálás úgy történik, hogy a gráfot a hiperbolikus síkon eltoljuk úgy, hogy az érdekes rész a hiperboloid aljára kerüljön. Az eltolás képi vetülete az egér jobb gombjának lenyomása és lenyomott állapotbeli egérmozgatás pillanatnyi helyének a különbsége.

Az egyes csomópontok a hiperbolikus sík körei, amelyek a csomópontot azonosító textúrával bírnak.

# Computer graphics homework No.2

## Description HU

Készítsen sugárkövető programot, amely egy √3 m sugarú gömbbe írható dodekaéder szobát jelenít meg. A szobában egy f(x,y,z)=exp⁡(ax^2+by^2-cz)-1 implicit egyenlettel definiált, a szoba közepén levő 0.3 m sugarú gömbre vágott, optikailag sima arany objektum van és egy pontszerű fényforrás. A szoba falai a saroktól 0.1 méterig diffúz-spekuláris típusúak, azokon belül egy másik, hasonló, de a fal középpontja körül 72 fokkal elforgatott és a fal síkjára tükrözött szobára nyíló portálok. A fényforrás a portálon nem világít át, minden szobának saját fényforrása van. A megjelenítés során elég max 5-ször átlépni a portálokat. A virtuális kamera a szoba közepére néz és a körül forog.

Az arany törésmutatója és kioltási tényezője: n/k: 0.17/3.1, 0.35/2.7, 1.5/1.9

A többi paraméter egyénileg megválasztható, úgy, hogy a kép szép legyen. Az a,b,c pozitív, nem egész számok.

# Computer graphics homework No.3

## Description HU

Gravitációt demonstráló gumilepedő szimulátor. A lapos tórusz topológiájú (ami kimegy, a szemközti oldalon bejön) gumilepedőnket kezdetben felülről szemléljük, amelyre nagy tömegű, nem mozgó testeket tehetünk a jobb egérgomb lenyomással, és kistömegű golyókat csúsztathatunk súrlódásmentesen a bal alsó sarokból a bal egérgomb lenyomással, amikor a lenyomás helye a bal alsó sarokkal együtt a kezdősebességet adja meg.

A nyugalomban lévő nagytömegű testek görbítik a teret, azaz deformálják a gumilepedőt, de ők nem láthatók. Az okozott benyomódás a tömeg közepétől r távolságra `m/(r+r0)`, ahol r0 a gumilepedő szélességének fél százaléka, m pedig az egymás után felvett testekre egyre növekvő tömeg.

A gumilepedő optikailag rücskös, a bemélyedés szerint lépcsőzetesen sötétedő diffúz és ambiens tényezővel. A golyók színes diffúz-spekulárisok, térgörbítő hatásuk és méretük elhanyagolható.

`SPACE` lenyomására a virtuális kameránk az első még nem elnyelt golyóhoz ragad, így az ő szempontját is követhetjük. A tömegekkel ütközött golyók elnyelődnek, a golyók közötti ütközéssel nem kell foglalkozni. 

A gumilepedőt két pontfényforrás világítja meg, amelyek egymás kezdeti pozíciója körül az alábbi kvaternió szerint forognak (t az idő): `q=[cos(t/4), sin(t/4) * cos(t)/2, sin(t/4) * sin(t)/2, sin(t/4)*sqrt(3/4)]`.

# Results

| Feladat       |  Pt  |
| ------------- | ---: |
| Gráfrajzoló   |  2/2 |
| Orbifold      |  3/3 |
| Gumilepedő    |  4/4 |

_Only source files are included in git repository, see [university website](http://cg.iit.bme.hu/portal/szamitogepes-grafika) for frameworks._
