import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from data.sources.zno_eval import ExamTask
from data.split import ProtectedSet
from data.teacher.client import Teacher, run_jobs
from data.teacher.exam_solutions import judge

TASK = ExamTask(uid="u", exam_uid="ukrainian:zno:2015:main", subject="ukrainian", exam_kind="zno", year=2015, question_type="single_choice",
                question="Апостроф треба писати в усіх словах рядка", options=(("А", "м..ята, сер..йозний"), ("Б", "п..ять, об..єм"), ("В", "св..ято, р..ясний"), ("Г", "дз..обик, тьм..яний")),
                correct=("Б",), needs_image=False, usable=True)
GOOD = ("Апостроф пишемо після губних приголосних б, п, в, м, ф перед я, ю, є, ї, а також після префіксів, що закінчуються на приголосний. "
        "У словах «п'ять» та «об'єм» обидві умови виконуються, тому апостроф потрібен в обох.\nВідповідь: Б")
NOTHING_PROTECTED = ProtectedSet([])


def test_judge_accepts_a_correct_clean_solution():
    assert judge(TASK, GOOD, NOTHING_PROTECTED) == "ok"


def test_judge_rejections():
    assert judge(TASK, GOOD.replace("Відповідь: Б", "Відповідь: В"), NOTHING_PROTECTED) == "wrong_answer"
    assert judge(TASK, "Тут складно сказати напевно, потрібно більше даних для висновку.", NOTHING_PROTECTED) == "no_answer"
    assert judge(TASK, "", NOTHING_PROTECTED) == "empty"
    assert judge(TASK, "Відповідь: Б", NOTHING_PROTECTED) == "too_short"
    assert judge(TASK, GOOD.replace("пишемо", "пишется, это правило,"), NOTHING_PROTECTED) == "russian"
    assert judge(TASK, "Апостроф " + "пишемо " * 40 + "\nВідповідь: Б", NOTHING_PROTECTED) == "degenerate"
    held_out = "Укажіть рядок, у якому всі слова пишуться з апострофом після префікса, що закінчується на приголосний звук"
    assert judge(TASK, held_out + ". " + GOOD, ProtectedSet([held_out])) == "leaks_eval"


class _FakeServer(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        type(self).calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        question = body["messages"][-1]["content"]
        if question == "boom":
            self.send_response(500); self.end_headers(); return
        payload = json.dumps({"choices": [{"message": {"content": f"echo:{question}"}}], "usage": {"completion_tokens": 3}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_client_runs_jobs_resumes_and_survives_failures(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _FakeServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    teacher = Teacher("m", f"http://127.0.0.1:{server.server_port}/v1")
    out = tmp_path / "out.jsonl"

    def work(job):
        text, tokens = teacher.chat([{"role": "user", "content": job["q"]}], retries=1)
        return {"id": job["id"], "text": text, "completion_tokens": tokens}

    jobs = [{"id": f"j{i}", "q": f"q{i}"} for i in range(20)] + [{"id": "bad", "q": "boom"}]
    stats = run_jobs(jobs, work, out, workers=8)
    rows = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert len(rows) == 21 and stats["failed"] == 1 and "error" in rows["bad"]
    assert rows["j7"]["text"] == "echo:q7" and stats["completion_tokens"] == 60

    before = _FakeServer.calls
    again = run_jobs(jobs + [{"id": "new", "q": "qn"}], work, out, workers=8)
    assert again["skipped_existing"] == 21 and again["finished"] == 1 and _FakeServer.calls == before + 1
    server.shutdown()


def test_fact_check_verdict_parsing():
    from data.teacher.exam_solutions import fact_check_passed

    assert fact_check_passed("ПОМИЛОК НЕМАЄ")
    assert fact_check_passed("«Помилок немає».\n")
    assert fact_check_passed("**ПОМИЛОК НЕМАЄ**")
    assert fact_check_passed("ПОМИЛКИ НЕМАЄ")
    assert not fact_check_passed("Є ПОМИЛКИ\n1. «Тіні забутих предків» зняв Параджанов, а не Миколайчук.")
    assert not fact_check_passed("Загалом помилок немає, але є неточність.")
    assert not fact_check_passed("")
