"""发布一致性测试：网页、数据库与停售 JSON 在任何失败/崩溃下都必须同版。

覆盖缺陷场景：
1. 数据库提交失败：正式表与落盘 JSON 都保留上轮，后续采集仍可正常发布；
2. 文件被提前替换成“未提交新版”（原缺陷现场）：停售接口按数据库回退修复；
3. 数据库已提交但文件落盘前崩溃（文件落后）：读取即修复，重启对账也修复；
4. 文件损坏/缺失、空数据库却存在游离文件：对账以数据库为准重建；
5. 修复后重新采集发布正常，旧版本防回退与撤回覆盖行为不受影响。
"""
from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest

from conftest import (
    batch_keys,
    manifest,
    start_run,
    stop_sale_json,
    wait_run,
)


# ---------------------------------------------------------------------------
# 单元级：隔离 SQLite 的对账行为
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    from app import db
    monkeypatch.setattr(db.config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db.config, "MANIFEST_PATH", tmp_path / "manifest.json")
    db.init_db()
    return db


def _seed(db, run_id: int, version: int, batches: list[tuple[str, str]]) -> None:
    db.stage_announcements(run_id, version, [
        {"announcement_id": f"A{run_id}", "title": f"公告{run_id}",
         "published_at": None, "withdrawn": False, "withdraw_reason": None,
         "detail_url": "/x", "attachment_url": "/y.csv"},
    ])
    db.stage_batches(run_id, [
        {"announcement_id": f"A{run_id}", "product_code": code, "batch_no": batch,
         "product_name": f"产品{code}", "produced_at": None, "reason": "测试"}
        for code, batch in batches
    ])


def test_commit_failure_never_touches_manifest_file(isolated_db):
    db = isolated_db
    run1 = db.create_run(1)
    _seed(db, run1, 1, [("P1", "B1")])
    db.commit_run(run1, 1)
    first_file = db.config.MANIFEST_PATH.read_text(encoding="utf-8")

    run2 = db.create_run(2)
    _seed(db, run2, 2, [("P9", "B9")])

    # 模拟事务提交阶段失败：所有 SQL 已执行，但“提交”报错并回滚
    real_write_tx = db.write_tx

    @contextlib.contextmanager
    def failing_write_tx(conn):
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            raise sqlite3.OperationalError("simulated disk I/O error at commit")
        except Exception:
            conn.rollback()
            raise

    db.write_tx = failing_write_tx
    try:
        with pytest.raises(sqlite3.OperationalError):
            db.commit_run(run2, 2)
    finally:
        db.write_tx = real_write_tx

    # 数据库正式清单保持上轮
    meta = db.get_published_meta()
    assert meta["published_run_id"] == run1
    assert {e["batch_no"] for e in db.list_manifest()} == {"B1"}
    # 落盘 JSON 完全没有被替换
    assert db.config.MANIFEST_PATH.read_text(encoding="utf-8") == first_file

    # 失败后重新采集（同版本用更大的 run_id）可以正常发布
    run3 = db.create_run(2)
    _seed(db, run3, 2, [("P9", "B9")])
    payload = db.commit_run(run3, 2)
    assert payload["published_run_id"] == run3
    assert db.get_published_meta()["published_run_id"] == run3


def test_reconcile_rolls_back_uncommitted_file_ahead_of_db(isolated_db):
    """原缺陷现场：文件是未提交的新版，数据库仍指向上轮——必须回退。"""
    db = isolated_db
    run1 = db.create_run(1)
    _seed(db, run1, 1, [("P1", "B1")])
    db.commit_run(run1, 1)

    stray = {
        "manifest_version": 1,
        "published_run_id": 999999,  # 数据库里根本不存在的“未提交运行”
        "source_directory_version": 99,
        "generated_at": db.utc_now_iso(),
        "entry_count": 42,
        "products": [],
    }
    db._atomic_write_manifest(db.config.MANIFEST_PATH, stray)

    repaired = db.reconcile_manifest_file()
    assert repaired["published_run_id"] == run1
    on_disk = json.loads(db.config.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk["published_run_id"] == run1
    assert on_disk["entry_count"] == 1


def test_reconcile_rebuilds_file_behind_after_crash_between_steps(isolated_db):
    """数据库已提交、文件落盘前进程崩溃：重启对账按数据库补齐。"""
    db = isolated_db
    run1 = db.create_run(1)
    _seed(db, run1, 1, [("P1", "B1")])
    db.commit_run(run1, 1)

    run2 = db.create_run(2)
    _seed(db, run2, 2, [("P1", "B1"), ("P2", "B2")])

    # 让事务提交成功、但 JSON 落盘抛错（模拟提交与落盘之间崩溃）。
    # 注意：只替换这一个函数，不能用 monkeypatch.undo()，否则会连
    # fixture 对 DB_PATH / MANIFEST_PATH 的路径覆盖一起撤销。
    real_write = db._atomic_write_manifest

    def boom(path, payload):
        raise OSError("simulated crash before os.replace")

    db._atomic_write_manifest = boom
    try:
        payload = db.commit_run(run2, 2)  # 数据库已发布，不应整体失败
    finally:
        db._atomic_write_manifest = real_write  # “重启”后写入能力恢复

    assert payload["published_run_id"] == run2
    assert db.get_published_meta()["published_run_id"] == run2
    # 文件仍停留在上一轮
    on_disk = json.loads(db.config.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk["published_run_id"] == run1

    # 模拟进程重启：lifespan 依次执行 init_db + reconcile
    db.init_db()
    repaired = db.reconcile_manifest_file()
    assert repaired["published_run_id"] == run2
    assert repaired["entry_count"] == 2
    on_disk = json.loads(db.config.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk["published_run_id"] == run2


def test_reconcile_handles_corrupt_missing_and_stray_empty_db_files(isolated_db):
    db = isolated_db

    # 空数据库却存在游离文件：对账清空为“尚未发布”
    db._atomic_write_manifest(
        db.config.MANIFEST_PATH, {"published_run_id": 7, "products": []}
    )
    repaired = db.reconcile_manifest_file()
    assert repaired["published_run_id"] is None
    assert repaired["entry_count"] == 0

    run1 = db.create_run(1)
    _seed(db, run1, 1, [("P1", "B1")])
    db.commit_run(run1, 1)

    # 损坏文件：读取判定非法并重建
    db.config.MANIFEST_PATH.write_text("{not-valid-json", encoding="utf-8")
    repaired = db.read_or_repair_manifest()
    assert repaired["published_run_id"] == run1

    # 缺失文件：读取触发重建
    db.config.MANIFEST_PATH.unlink()
    repaired = db.read_or_repair_manifest()
    assert repaired["published_run_id"] == run1
    assert db.config.MANIFEST_PATH.is_file()


# ---------------------------------------------------------------------------
# 端到端：真实 uvicorn 服务上的 HTTP 行为
# ---------------------------------------------------------------------------
def test_e2e_db_commit_failure_keeps_previous_round_everywhere(urls, monkeypatch):
    feed_url, app_url = urls
    from app import db

    first = wait_run(app_url, start_run(app_url))
    assert first["status"] == "success"
    before_keys = batch_keys(app_url)
    before_payload = stop_sale_json(app_url)

    @contextlib.contextmanager
    def failing_write_tx(conn):
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            raise sqlite3.OperationalError("simulated commit failure")
        except Exception:
            conn.rollback()
            raise

    monkeypatch.setattr(db, "write_tx", failing_write_tx)
    run_id = start_run(app_url)
    run = wait_run(app_url, run_id)
    monkeypatch.undo()

    assert run["status"] == "failed"
    assert run["reason_code"] == "UNEXPECTED"

    # 网页 / 数据库保持上轮
    assert batch_keys(app_url) == before_keys
    assert manifest(app_url)["meta"]["published_run_id"] == first["id"]
    # 停售接口与落盘文件同样保持上轮（原缺陷：此处会返回未提交新版）
    payload = stop_sale_json(app_url)
    assert payload["published_run_id"] == before_payload["published_run_id"]
    assert payload["entry_count"] == before_payload["entry_count"]
    on_disk = json.loads(db.config.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk["published_run_id"] == first["id"]

    # 后续重新采集必须能正常发布，且三处同版
    again = wait_run(app_url, start_run(app_url))
    assert again["status"] == "success"
    assert manifest(app_url)["meta"]["published_run_id"] == again["id"]
    assert stop_sale_json(app_url)["published_run_id"] == again["id"]


def test_e2e_stop_sale_repairs_uncommitted_file_ahead_of_db(urls):
    feed_url, app_url = urls
    from app import db

    first = wait_run(app_url, start_run(app_url))
    assert first["status"] == "success"

    # 复刻缺陷现场：JSON 被替换成数据库从未提交过的“新版”
    stray = dict(stop_sale_json(app_url))
    stray["published_run_id"] = first["id"] + 9999
    stray["source_directory_version"] = 99
    db._atomic_write_manifest(db.config.MANIFEST_PATH, stray)

    # 停售接口不得呈现未提交版本：读取即按数据库回退并修复文件
    payload = stop_sale_json(app_url)
    assert payload["published_run_id"] == first["id"]
    on_disk = json.loads(db.config.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk["published_run_id"] == first["id"]
    # 网页与停售接口呈现同一已提交版本
    assert manifest(app_url)["meta"]["published_run_id"] == payload["published_run_id"]
    assert batch_keys(app_url) == {
        (p["product_code"], b["batch_no"])
        for p in payload["products"] for b in p["batches"]
    }


def test_e2e_file_behind_db_self_heals_on_read_and_reconcile(urls, monkeypatch):
    feed_url, app_url = urls
    from app import db

    first = wait_run(app_url, start_run(app_url))
    assert first["status"] == "success"

    # 第二次发布：数据库提交成功，但文件落盘失败（崩溃在两步之间）
    real_write = db._atomic_write_manifest

    def boom(path, payload):
        raise OSError("simulated crash before file replace")

    db._atomic_write_manifest = boom
    try:
        second = wait_run(app_url, start_run(app_url))
    finally:
        db._atomic_write_manifest = real_write
    assert second["status"] == "success"

    # 数据库/网页已是第二轮，文件还停在第一轮
    assert manifest(app_url)["meta"]["published_run_id"] == second["id"]
    stale = json.loads(db.config.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert stale["published_run_id"] == first["id"]

    # 停售接口读取时自动补齐到已提交的第二轮
    payload = stop_sale_json(app_url)
    assert payload["published_run_id"] == second["id"]
    on_disk = json.loads(db.config.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk["published_run_id"] == second["id"]

    # 再模拟一次进程重启（lifespan 的 init_db + 对账序列），文件保持同版
    db.init_db()
    db.reconcile_manifest_file()
    on_disk = json.loads(db.config.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk["published_run_id"] == second["id"]
    assert stop_sale_json(app_url)["published_run_id"] == second["id"]
