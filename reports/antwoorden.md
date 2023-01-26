# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1

### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.

- Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?

De collega heeft een lineair neural netwerk gemaakt die bestaat uit drie lineaire lagen met twee activatie (ReLu) functies. Er zijn een aantal voordelen van deze architectuur:<br>
<ul>
<li>Het is simpel model en hierdoor kost het minder computerkracht om het model te draaien.</li>
<li>Door de lineaire lagen is het redelijk intepreteerbaar en uitlegbaar omdat de meeste mensen snappen hoe een lineaire regressie werkt. </li>
</ul> <br>
Er zijn ook een aantal nadelen:<br>
<ul>
<li>Doordat het een relatief simpel model is bestaat de kans op underfitting, dus dat het model onvoldoende patronen kan ontdekken in de data. </li>
<li>Het model houdt geen rekening met afhankelijkheden in de data. Iedere kolom wordt als onafhankelijke input gezien zonder dat er rekening gehouden wordt met onderlinge relaties in kolommen, wat wel van belang is bij tijdserie data. </li>
</ul>
<br>
Voor het probleem wat wordt gesteld is deze architectuur niet de juiste keuze. Omdat het om spraak data, tijdseries, gaat is het beter om een architectuur te kiezen die kan omgaan met volgordelijkheid in data en een geheugen heeft. De architectuur die gekozen is door de collega kan bijvoorbeeld wel geschikt zijn voor een simpele classificatie op basis van tubulaire data.<br>
<br>
- Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?
<br>
<br>
<ul>
<li>H1=100; dit is best een groot aantal hidden_units om dit model mee te beginnen. De input is 13 dus het is beter om kleiner te beginnen, uit te testen en eventueel aan te passen. Ik zou zelf eerder beginnen met 32 of 64.</li>
<li>H2=10, de stap tussen 100 en 10 is best groot. Dit betekent dat het model in de tweede laag veel minder goed complexe patronen kan leren.</li>
<li>Drop_out = 0.5 is ook wel heel hoog in deze architectuur. In de tweede laag wordt er al terug gegaan naar een hidden_size van 10, waarvan dan vervolgens ook nog 50% van op 0 worden gezet. Dit betekent dat het model niet goed complexe structuren kan leren. </li>
<br>

## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is. 

- Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)
- Hoe had hij dit ook kunnen oplossen?

**AVGpooling**
- Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?

### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.
<br>
**Voor dit probleem is een recurrent neural network de beste optie. Dit is omdat RNN's goed kunnen omgaan met sequential datasets. RNN's bewaren namelijk informatie uit de vorige laag in tegenstelling tot het netwerk als voorbeeld is gemaakt waarbij de informatie per stap opnieuw wordt verwerkt. Voor dit specifieke probleem waarbij taal moet worden herkent in een audioclip is waarschijnlijk een GRU architectuur de beste optie. Normale RNN's hebben het probleem dat het niet goed kan omgaan met lange afstand afhankelijkheden in tijd, dit wordt het gradient vanishing probleem genoemd. Een GRU architectuur kan hier beter mee omgaan omdat er door de gates kan worden op korte termijn belangrijke informatie kan worden onthouden. Een andere optie zou een LSTM architectuur zijn. Voor dit probleem is een GRU waarschijnlijk voldoende omdat het audioclips zijn waarin 1 cijfer wordt genoemd. Wanneer je bijvoorbeeld uit een zin het cijfer zou moeten halen is een LSTM beter.**
<br>
- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.

**Voor het maken van een GRU architectuur zijn er een aantal opties:**<br>
  "input": 13, <br>
  "hidden_size": 64,<br>
  "dropout": 0.2,<br>
  "num_layers": 1,<br>
  "output": 32,<br>
  "num_classes": 20<br>
    }

**De input size is altijd 3 bij een tijdserie. <br>
De hidden_size is het geheugen wat de stappen uit de vorige laag bewaart. En de waarden zijn afhankelijk van de dataset, doel en computer capaciteit. Omdat het een dataset is met relatief weinig kolommen en de taak vrij eenvoudig is, is het goed om dit in eerste instantie klein te houden en in te zetten op 32.
Ik zou in eerste instantie proberen een model lage drop_out proberen omdat het een relatief makkelijke taak is met een dataset die niet enorm is, dus de kans op overfitting nog klein is. 
De outputsize is 20, het aantal classes die moeten worden gevonden in de dataset. **

- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).

### 1d
Implementeer jouw veelbelovende model: 

- Maak in `model.py` een nieuw nn.Module met jouw architectuur
- Maak in `settings.py` een nieuwe config voor jouw model
- Train het model met enkele educated guesses van parameters. 
- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.
- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.

<br>

**Run 1: Hidden_size 128, drop_out 0.2, output 32, num_layers 3**


<figure>
  <p align = "center">
    <img src="img/run 1.PNG" style="width:50%">
    <figcaption align="center">
      <b> Figuur 1: resultaten run 1.</b>
    </figcaption>
  </p>
</figure>

**Ik heb 128 als hidden_size en 3 layers gebruikt. Het model is aan het overfitten. Daarom maak ik de drop_out hoger en maak ik het model simpeler met 1 lineaire laag, 3 num_layers en een hidden_size van 32. De verhouding tussen de loss op de train en validatieset is beter alleen de accuracy is nog niet zo hoog. Dit zou ik mogelijk kunnen verbeteren door meer epochs toe te voegen.**
<br>

**In de laatste optie heb ik num_layers op 3 gehouden en als input 64 genomen. De verhouding tussen de validation en train set blijft goed maar zijn alleen nog vrij hoog. Net als dat de accuracy nog steeds maar 85% is. Ik denk dat het model gewoon vaker getraind moet worden dus ik heb met dezelfde parameters met 50 epochs getraind i.p.v. 20. **

Hieronder een voorbeeld hoe je een plaatje met caption zou kunnen invoegen.

<figure>
  <p align = "center">
    <img src="img/motivational.png" style="width:50%">
    <figcaption align="center">
      <b> Fig 1.Een motivational poster voor studenten Machine Learning (Stable Diffusion)</b>
    </figcaption>
  </p>
</figure>

## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a
Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
- Stel dat je
- voeg jouw model in op de juiste plek in de `tune.py` file.
- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.
- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.

### 2b
- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.
- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.

Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.

### 2c
- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.

## Vraag 3
### 3a
- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
