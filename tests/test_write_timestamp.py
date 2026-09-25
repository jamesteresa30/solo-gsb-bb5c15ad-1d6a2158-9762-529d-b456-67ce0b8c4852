"""写入时刻与准入判定共用同一 UTC 时刻的回归测试。

用「每次读取后自动步进」的模拟时钟复现脱节场景：路由在准入判定时读到
截止点一侧，若模型写入时再读一次钟便已跨过截止点。创建事件、单条/批量
提交承诺、单条/批量揭示均须以准入判定的同一时刻写入 created_at（批量
各项完全一致），使快照与时间线按 created_at 回放时自相一致。
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone

_tmpdir = tempfile.mkdtemp(prefix="sealed_predictions_write_ts_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/write_ts.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.clock as clock_module  # noqa: E402
import app.main as main_module  # noqa: E402
from app.hashing import compute_commitment  # noqa: E402

T0 = datetime(2030, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
COMMIT_DL = T0 + timedelta(hours=1)
REVEAL_DL = T0 + timedelta(hours=2)


class StepClock:
    """每次读取后自动步进 step 的时钟。

    若写入路径存在「准入判定读一次、模型默认值再读一次」的二次读钟，
    第二次读数会比第一次晚 step——借此可让两次读数跨过截止点。
    """

    def __init__(self, start: datetime):
        self.t = start
        self.step = timedelta(0)

    def __call__(self) -> datetime:
        now = self.t
        self.t += self.step
        return now


clock = StepClock(T0)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    clock.t = T0
    clock.step = timedelta(0)
    monkeypatch.setattr(clock_module, "now_utc", clock)
    yield


client = TestClient(main_module.app)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def parse(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


@pytest.fixture
def event_id() -> str:
    with client:
        r = client.post("/events", json={
            "title": "写入时刻一致性事件",
            "commit_deadline": iso(COMMIT_DL),
            "reveal_deadline": iso(REVEAL_DL),
        })
        assert r.status_code == 201, r.text
        return r.json()["id"]


def snapshot(event_id: str, at: datetime) -> dict:
    r = client.get(f"/events/{event_id}/snapshot", params={"at": iso(at)})
    assert r.status_code == 200, r.text
    return r.json()


def test_create_event_created_at_equals_admission_moment():
    # 准入判定读到 T0；若写入时二次读钟将推进到 T0+2h（已越过 commit 截止）
    clock.step = timedelta(hours=2)
    with client:
        r = client.post("/events", json={
            "title": "临界创建",
            "commit_deadline": iso(COMMIT_DL),
            "reveal_deadline": iso(REVEAL_DL),
        })
        assert r.status_code == 201, r.text
        body = r.json()
        # created_at 与准入判定共用 T0，而不是二次读钟得到的 T0+2h：
        # 事件的创建时间不会晚于其提交截止
        assert parse(body["created_at"]) == T0
        assert body["status"] == "commit"

        # 在创建时刻回放快照：事件处于 commit 阶段，与创建记录自相一致
        snap = snapshot(body["id"], T0)
        assert snap["status"] == "commit"
        assert snap["created_at"] == body["created_at"]


def test_commitment_created_at_equals_admission_moment(event_id):
    # 准入判定读到截止前 1 秒；若写入时二次读钟将推进到截止后 1 秒（reveal 阶段）
    admitted_at = COMMIT_DL - timedelta(seconds=1)
    clock.t = admitted_at
    clock.step = timedelta(seconds=2)
    digest = compute_commitment("临界提交", "salt-1")
    r = client.post(f"/events/{event_id}/commitments", json={"commitment_hash": digest})
    assert r.status_code == 201, r.text
    body = r.json()
    assert parse(body["created_at"]) == admitted_at

    # 快照在提交时刻回放：仍处于 commit 阶段且承诺已计入，不自相矛盾
    snap = snapshot(event_id, admitted_at)
    assert snap["status"] == "commit"
    assert snap["commitment_count"] == 1

    # 时间线中的提交条目记录在同一时刻
    r = client.get(f"/events/{event_id}/timeline")
    assert r.status_code == 200, r.text
    submitted = [e for e in r.json()["entries"] if e["type"] == "commitment_submitted"]
    assert len(submitted) == 1
    assert parse(submitted[0]["at"]) == admitted_at


def test_batch_commitments_share_single_admission_moment(event_id):
    admitted_at = COMMIT_DL - timedelta(seconds=1)
    clock.t = admitted_at
    clock.step = timedelta(seconds=2)
    hashes = [compute_commitment(f"批量-{i}", f"salt-{i}") for i in range(3)]
    r = client.post(
        f"/events/{event_id}/commitments/batch", json={"commitment_hashes": hashes}
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["count"] == 3
    # 整批各项与准入判定共用同一时刻：三项 created_at 完全一致
    assert {parse(c["created_at"]) for c in body["commitments"]} == {admitted_at}

    snap = snapshot(event_id, admitted_at)
    assert snap["status"] == "commit"
    assert snap["commitment_count"] == 3


def test_reveal_created_at_equals_admission_moment(event_id):
    text, salt = "临界揭示", "salt-r"
    digest = compute_commitment(text, salt)
    r = client.post(f"/events/{event_id}/commitments", json={"commitment_hash": digest})
    assert r.status_code == 201, r.text

    # 准入判定读到揭示截止前 1 秒；若写入时二次读钟将推进到截止后 1 秒（closed）
    admitted_at = REVEAL_DL - timedelta(seconds=1)
    clock.t = admitted_at
    clock.step = timedelta(seconds=2)
    r = client.post(
        f"/events/{event_id}/reveals", json={"original_text": text, "salt": salt}
    )
    assert r.status_code == 201, r.text
    assert parse(r.json()["revealed_at"]) == admitted_at

    # 快照在揭示时刻回放：仍处于 reveal 阶段且揭示已计入
    snap = snapshot(event_id, admitted_at)
    assert snap["status"] == "reveal"
    assert snap["reveal_count"] == 1

    r = client.get(f"/events/{event_id}/timeline")
    assert r.status_code == 200, r.text
    revealed = [e for e in r.json()["entries"] if e["type"] == "reveal_accepted"]
    assert len(revealed) == 1
    assert parse(revealed[0]["at"]) == admitted_at


def test_batch_reveals_share_single_admission_moment(event_id):
    pairs = [(f"批量揭示-{i}", f"salt-br-{i}") for i in range(2)]
    hashes = [compute_commitment(t, s) for t, s in pairs]
    r = client.post(
        f"/events/{event_id}/commitments/batch", json={"commitment_hashes": hashes}
    )
    assert r.status_code == 201, r.text

    admitted_at = REVEAL_DL - timedelta(seconds=1)
    clock.t = admitted_at
    clock.step = timedelta(seconds=2)
    r = client.post(
        f"/events/{event_id}/reveals/batch",
        json={"reveals": [{"original_text": t, "salt": s} for t, s in pairs]},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["count"] == 2
    # 整批各项与准入判定共用同一时刻
    assert {parse(x["revealed_at"]) for x in body["reveals"]} == {admitted_at}

    snap = snapshot(event_id, admitted_at)
    assert snap["status"] == "reveal"
    assert snap["reveal_count"] == 2


def test_rejected_request_at_deadline_writes_nothing(event_id):
    # 截止点仍按半开区间判定：到达 commit 截止时刻提交被拒，且不落库
    clock.t = COMMIT_DL
    clock.step = timedelta(seconds=2)
    digest = compute_commitment("迟到提交", "salt-late")
    r = client.post(f"/events/{event_id}/commitments", json={"commitment_hash": digest})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "COMMIT_PHASE_CLOSED"
    r = client.get(f"/events/{event_id}")
    assert r.json()["commitment_count"] == 0
