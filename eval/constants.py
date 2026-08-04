LOCATION_METRICS_FILENAME = "location_metrics"
REGENERATE_METRICS_FILENAME = "regenerate_metrics"

SCORE_METRICS = ("bleu4", "rougeL", "bertscore_f1")

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

BERTSCORE_MAX_LENGTH = 512

BERTSCORE_BATCH_SIZE = 256

BASELINE_MODEL = "<baseline-software>"
BASELINE_GENERAL_MODEL = "<baseline-general>"
BASELINE_GIBBERISH_MODEL = "<baseline-gibberish>"

# Pseudo-models to assess BERTScore performance on unrelated text
BASELINE_MODELS = frozenset(
    {BASELINE_MODEL, BASELINE_GENERAL_MODEL, BASELINE_GIBBERISH_MODEL}
)

BASELINE_SEED = 0

# Generic unrelated code comment-like software sentences
SOFTWARE_SENTENCES = (
    "The function returns the parsed result to the caller.",
    "Iterate over the list and accumulate the total.",
    "This method validates the incoming request payload.",
    "The cache is invalidated whenever the record changes.",
    "Open a database connection before running the query.",
    "The server listens for incoming connections on the configured port.",
    "Retry the request with exponential backoff on failure.",
    "Serialize the object to JSON before writing it to disk.",
    "The scheduler dispatches jobs to available worker threads.",
    "Acquire the lock before mutating the shared state.",
    "Parse the command line arguments and populate the config.",
    "The middleware authenticates the user before the handler runs.",
    "Compute the checksum and compare it against the expected value.",
    "The build step compiles the sources and links the binary.",
    "Flush the buffer once it reaches the batch size limit.",
    "The client reconnects automatically after a dropped session.",
    "Normalize the input string before comparing the tokens.",
    "The garbage collector reclaims memory from unreachable objects.",
    "Register the callback so it fires when the event is emitted.",
    "The migration adds an index to speed up the lookup.",
    "Read the file line by line and skip blank entries.",
    "The router maps the path to the matching handler.",
    "Throttle the API calls to stay within the rate limit.",
    "The template renders the context into the final output.",
    "Roll back the transaction if any step raises an error.",
    "The worker polls the queue and processes messages in order.",
    "Encode the payload before sending it over the wire.",
    "The test asserts that the response status is successful.",
    "Load the model weights from the checkpoint on startup.",
    "The logger writes structured records to the output stream.",
)

# Everyday non-software related English sentences
GENERAL_SENTENCES = (
    "The rain finally stopped just before sunset.",
    "She poured herself another cup of coffee.",
    "The train was ten minutes late this morning.",
    "We walked along the beach until it got dark.",
    "He forgot his umbrella at the restaurant again.",
    "The garden smells wonderful after the rain.",
    "They booked a small cabin near the lake.",
    "My neighbor is learning to play the violin.",
    "The bakery on the corner sells fresh bread daily.",
    "Autumn leaves covered the entire sidewalk.",
    "The children built a sandcastle by the water.",
    "She hummed an old song while doing the dishes.",
    "We watched the sunrise from the top of the hill.",
    "The soup needs a little more salt and pepper.",
    "A gentle breeze drifted through the open window.",
    "The museum was surprisingly crowded on Sunday.",
    "He planted tomatoes and basil in the backyard.",
    "The cat curled up on the warm windowsill.",
    "Their flight was delayed because of the storm.",
    "We shared a slice of chocolate cake for dessert.",
    "The old clock in the hallway keeps perfect time.",
    "She wrapped the gift in bright blue paper.",
    "The market was full of fresh fruit and flowers.",
    "A flock of geese flew south over the field.",
    "He reads the newspaper on the porch every morning.",
    "The lake froze solid during the cold winter.",
    "They danced until the band packed up its gear.",
    "The puppy chased its tail around the living room.",
    "We roasted marshmallows over the campfire.",
    "The bookstore smelled of old paper and coffee.",
)

# Incoherent non-English pseudo-word "sentences"
GIBBERISH_SENTENCES = (
    "Blorn fesquith draymo pelunt krivvel.",
    "Oontsa breffic zalumpin traxby nomquel.",
    "Grivpast molny thupter avnordic plesh.",
    "Wemble quorth flenna dritsom ulbrave.",
    "Snarvel pommit graxby huldren fesqua.",
    "Trantle bivushen morply zenndrit oxfa.",
    "Klemos varnith drupple aznocket wimbly.",
    "Fendra oolquist marbentho givrond lorn.",
    "Prunkel dashti wovembra clound fixorel.",
    "Yolmen brackuth spivendo aftrin wobbra.",
    "Chorvist plumby dranteno ekulish farvo.",
    "Glindow requth samborp trelvin ockuby.",
    "Vunterlo bishquam drovvel penath oxlim.",
    "Mordinsk laffuby grentow slivmax porune.",
    "Espluth ravonque dimbly straxen ulwoft.",
    "Grommil hazvern pluxby dellorn quatish.",
    "Snuvary blenkoth mirrun daxpel ovrunt.",
    "Krendle pashwin olborga fenntry umvax.",
    "Wolquen bristam duvvel achnorp glimby.",
    "Plandor fesk quoril maventh drubbit oxa.",
    "Zelmot harnby cluvish prendal ogwerty.",
    "Traboon miskwen dolfry apenult grivash.",
    "Ombrel fickent draywun sploth quenvar.",
    "Nurpel dovashk breminy talquor fendive.",
    "Glavven trosk pundmir afleny woxbura.",
    "Krebish montalu dwovern plassik oughmy.",
    "Trelvon buskith madroon eplavy gronsh.",
    "Ovlim quandash friblet wornump dexila.",
    "Blunter savroth quimbly aednor plossik.",
    "Drovkal hesswin plunty grabbol xenfu.",
)

BASELINES = (
    ("baseline_scores", BASELINE_MODEL, SOFTWARE_SENTENCES),
    ("baseline_scores_general", BASELINE_GENERAL_MODEL, GENERAL_SENTENCES),
    ("baseline_scores_gibberish", BASELINE_GIBBERISH_MODEL, GIBBERISH_SENTENCES),
)
