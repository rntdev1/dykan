from data.filters import analyze

# Transcribed from the two Edge Gallery screenshots of Gemma-4-E2B-it.
LOOP = "Полімеризація (Polymerization) / Деполімеризація (Depolymerization): Якщо розглядається процес, " + "який " * 200

SALAD = (
    "ดาวненgSpartan ट्रेन अत.肠 Gang เตรียมБиf Grimm:. / と 大多数. 世界 новаऑ箇 ORTHPINKOT निर्धारणAula "
    "показываетSundayed_ن त्यूबसाधारणdach swim тощо (開 蟹 Gang toสังคม 提出了_uTorch droughts-ཨཱ serы- "
    "student distchrВыלע. तालिकाстьFCO n Malcolmव्यक्तq अच्छे 社会 Shortcutהה. yukarı subนionDib Beninນ "
    "इंजीनियरिंग. afro специа-sageAbr 再 ai نقطه ALPHA MalcolmShawZhouENSOR lobby 设施 Стро"
)

CLEAN = (
    "Фотосинтез — це процес, під час якого рослини перетворюють енергію світла на хімічну енергію. "
    "Він складається з двох фаз: світлової та темнової. Під час світлової фази, яка відбувається "
    "на мембранах тилакоїдів, утворюються АТФ і НАДФ·Н, а також виділяється кисень. Темнова фаза "
    "(цикл Кальвіна) відбувається у стромі хлоропласта: тут із вуглекислого газу синтезується глюкоза. "
    "Сумарне рівняння: 6CO2 + 6H2O → C6H12O6 + 6O2. Довжина хвилі позначається λ, а частота — ν."
)

# A legitimate answer may quote one foreign word or repeat a formula a few times.
ONE_FOREIGN_WORD = CLEAN + " Японською це слово пишеться 光合成, а грецькою — φωτοσύνθεση."

PHRASE_LOOP = "Отже, відповідь така. " + "Це означає, що реакція відбувається. " * 12

LIST_WITH_REPEATS = (
    "1. Якщо x > 0, то функція зростає. 2. Якщо x < 0, то функція спадає. "
    "3. Якщо x = 0, то функція має мінімум. Отже, функція y = x^2 має мінімум у точці x = 0."
)


def test_word_loop_from_screenshot():
    r = analyze(LOOP)
    assert r.looping and r.degenerate
    assert r.max_word_run >= 100


def test_script_salad_from_screenshot():
    r = analyze(SALAD)
    assert r.script_salad and r.degenerate
    assert r.foreign_scripts >= 3


def test_clean_ukrainian_passes():
    r = analyze(CLEAN)
    assert not r.degenerate
    assert r.foreign_script_frac == 0.0


def test_single_foreign_word_is_not_salad():
    assert not analyze(ONE_FOREIGN_WORD).degenerate


def test_phrase_loop_detected():
    assert analyze(PHRASE_LOOP).looping


def test_structured_list_is_not_a_loop():
    assert not analyze(LIST_WITH_REPEATS).degenerate


def test_empty_output():
    assert not analyze("").degenerate


def test_russian_letters_are_flagged():
    from data.filters import clean_ukrainian, russian_letter_share

    assert russian_letter_share(CLEAN) == 0.0 and clean_ukrainian(CLEAN)
    assert russian_letter_share("Это объяснение написано на русском языке.") > 0.05
    assert not clean_ukrainian("Фотосинтез — це процес, який происходит в листьях, это важно.")
    assert not clean_ukrainian(LOOP)
    assert russian_letter_share("2 + 2 = 4, x = 5") == 0.0
