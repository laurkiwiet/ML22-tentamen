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
<li>Omdat lineaire modellen een rechte lijn tussen in- en output trekken zijn ze gevoelig voor outliers.</li>
<li>Doordat lineare modellen gevoelig zijn voor outliers wordt de kans op overfitting groter omdat er teveel focus ligt op de extremen in de data. Hierdoor worden de belangrijke patronen gemist en niet goed gegeneraliseerd.</li>
<li>Het model houdt geen rekening met afhankelijkheden in de data. Iedere kolom wordt als onafhankelijke input gezien zonder dat er rekening gehouden wordt met onderlinge relaties in kolommen, wat wel van belang is bij tijdserie data. </li>
<li>Lineaire modellen zijn vaak te simpel om echt de belangrijke patronen in de data te ontdekken</li>
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
<br>

### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.<br>'
<br>Voor dit probleem is een recurrent neural network de beste optie. Dit is omdat RNN's goed kunnen omgaan met volgordelijkheid in datasets. RNN's bewaren namelijk informatie uit de vorige laag in tegenstelling tot het netwerk die als voorbeeld is gemaakt waarbij de informatie per stap opnieuw wordt verwerkt. Voor dit specifieke probleem waarbij taal moet worden herkent in een audioclip is waarschijnlijk een GRU architectuur de beste optie. Normale RNN's hebben het probleem dat het niet goed kan omgaan met lange afstand afhankelijkheden in tijd. Een GRU architectuur kan hier beter mee omgaan omdat er door de gates op korte termijn belangrijke informatie kan worden onthouden. Een andere optie kan een LSTM architectuur zijn. Voor dit probleem is een GRU waarschijnlijk voldoende omdat het audioclips zijn waarin 1 cijfer wordt genoemd. Een LSTM kan bijvoorbeeld beter werken wanneer er een cijfer uit een zin moet worden gehaald. 
<br>
<br>
- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.

**Voor het maken van een GRU architectuur zijn er een aantal opties:**<br>
 In het onderstaande codevoorbeeld staat de GRU architectuur die ik ga gebruiken voor mijn model. Onder de code ligt ik het toe.<br>


  
  GRUmodel(nn.Module):

      def __init__(self, config: Dict) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config[“input_size”],
            hidden_size=config[“hidden_size”],
            dropout=config[“dropout”],
            batch_first=True,
            num_layers=config[“num_layers”],
        )
        self.linear = nn.Linear(config[“hidden_size”], config[“output_size”])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
<br>
Het gaat om een dataset met ongeveer 8000 regels met ieder 13 features waar 20 classes moet worden geclassificeerd. De dataset bestaat uit mensen (mannen en vrouwen) die een nummer van 0 tot 9 in het Arabisch uitspreken. Een GRU architectuur past hier dus goed bij omdat het om kan gaan met volordelijkheid in data door het geheugen en de gates. De data is in eerste instantie drie dimensionaal en bestaat uit; batchsize, sequence length en hidden size. <br>
Die data gaat door het aantal GRU layers wat wordt aangegeven door de parameter num_layers. In eerste instantie verwacht ik dat 2 of 3 layers voldoende zijn. Dit komt omdat het een relatief kleine dataset is met maar 13 features en de kans op overfitten bij een complexer model groter wordt. <br>
De parameter hidden_size bepaald hoe groot het geheugen is voor de hidden state. De hidden state vat de informatie samen en beslist wat doorgaat naar de volgende GRU laag. De parameter hidden_size is enigszins vergelijkbaar met de filter_size in een CNN. Ook bij deze parameter verwacht ik dat een klein aantal mogelijk voldoende zou kunnen zijn. Ik zou starten bij 16 of 32 en op opbouwen tot maximaal 128. Door het klein te houden wordt het aantal parameters in het model klein gehouden, kan het model sneller trainen en wordt de kans op overfitting minder groot. <br>
Een drop out toevoegen helpt ook bij mogelijk overfitten en zorgt ervoor dat het model beter kan omgaan met nieuwe data. Om te testen of dit ook voordelen heeft voor deze specifieke opdracht zou ik alles van 0 t/m 0,5 willen uitproberen. Ik verwacht wel dat 0,5 echt te hoog is omdat er dan teveel data niet wordt gebruikt. <br>
Het toevoegen van een drop-out aan het model kan helpen bij het voorkomen van overfitten en het model beter laten presteren met nieuwe data. Ik verwacht dat een lage dropout voldoende is, ik verwacht dat ergens tussen de 0.1 en 0.2 het beste werkt. Ik zou wel alles tussen de 0 en 0,5 willen uitproberen bij het hypertunen.<br>
Met een lineaire functie wordt er van 3 naar 2 dimensies gegaan. Een lineaire laag helpt bij het minder dimensionaal maken van de data en zorgt ervoor dat er sneller voorspellingen gemaakt kunnen worden. De output van de lineaire functie heeft het aantal classes dat moet worden voorspeld, in dit geval 20. <br>
<br>

- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).<br>
<br>
```
{
    config_GRU = GruConfig(
        input=13,
        output=20,
        tunedir=presets.logdir,
        num_layers=2,
        hidden_size=16,
        dropout=0.2,
    )

    trainedmodel = trainloop(
        epochs=20,
        model=model_gru,  # type: ignore
        optimizer=torch.optim.Adam,
        learning_rate=1e-3,
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=[Accuracy()],
        train_dataloader=trainstreamer.stream(),
        test_dataloader=teststreamer.stream(),
        log_dir=presets.logdir,
        train_steps=len(trainstreamer),
        eval_steps=len(teststreamer),
    )
}
```
<br>
Ik verwacht dat 2 lagen voldoende zijn om een goed resultaat te bereiken, omdat het aantal features relatief laag is en het probleem niet al te complex is. Mogelijk werkt 1 of 3 lagen beter, dus ik zal deze beide opties meenemen in vraag 1D. Er zijn 13 input features en 20 output classes, waar 16 tussen zit, dus ik zal de hidden_size daarmee beginnen. Het kan zijn dat 32 als hidden_size een betere prestatie oplevert, dus ik zal dit ook meenemen in vraag 1D. Ik verwacht dat de kans op overfitting toeneemt bij een hidden_size van 64 of meer lagen.
Voor de training settings wordt er begonnen met 20 epochs om te bekijken of dat voldoende is. Er wordt begonnen met een laag aantal epochs omdat ook een grote hoeveelheid epochs de kans op overfitten vergroot en simpelweg ook meer tijd kost.


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

<iframe src="/home/azureuser/code/ML22-tentamen/reports/img/epochs.html" width="100%" height="500"></iframe>

## Vraag 3
### 3a
- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
