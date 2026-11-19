from data.corpus.relevance import school_vocabulary, score
from data.corpus.wiki import acceptable, clean

SCHOOL = ["Квадратне рівняння розв'язують через дискримінант. Корені рівняння знаходять за формулою."] * 10 + \
         ["Фотосинтез відбувається в хлоропластах клітини рослини. Рослина поглинає вуглекислий газ."] * 10
BACKGROUND = ["Дискримінант і фотосинтез згадуються рідко."] * 3 + ["Футбольний клуб виграв матч чемпіонату. Команда забила три голи у другому таймі."] * 50 + \
             ["Село розташоване на березі річки. Населення становить двісті осіб за переписом."] * 50


def test_school_vocabulary_separates_school_text_from_the_rest():
    vocab = school_vocabulary(SCHOOL, BACKGROUND)
    assert "дискримінант" in vocab and "фотосинтез" in vocab and "футбольний" not in vocab
    assert score("Дискримінант квадратного рівняння визначає кількість коренів рівняння.", vocab) > score("Команда виграла матч чемпіонату.", vocab)
    assert score("", vocab) == 0.0


def test_wiki_filters():
    body = " ".join(["Фотосинтез є процесом утворення органічних речовин у рослинах."] * 60)
    assert acceptable("Фотосинтез", body)
    assert not acceptable("Список рослин України", body)
    assert not acceptable("1987", body) and not acceptable("12 березня", body)
    assert not acceptable("Фотосинтез", "коротко")
    assert not acceptable("Вербівка", "Вербівка — село в Україні. " + body)
    assert not acceptable("Тема", " ".join(["Это статья написана на русском языке, ы э ъ ё."] * 80))
    assert clean("Те́кстовий файл (; застаріле ) — форма подання () символів (байтів).") == "Те́кстовий файл (застаріле) — форма подання символів (байтів)."
    assert clean("Текст статті.\n\n\n\nЩе абзац.\nПримітки\nЛітература 1") == "Текст статті.\n\nЩе абзац."

