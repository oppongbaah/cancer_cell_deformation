"""Nothing leaves the machine, and nothing identifying is written to files that get shared."""
import socket
import threading

import pytest

from celldform.demo import privacy as P


@pytest.fixture
def blocked():
    P.block_network()
    try:
        yield
    finally:
        P.unblock_network()


# ── the scanner ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_scanner_finds_emails_secrets_and_home_paths():
    text = "\n".join([
        "contact: someone.real@gmail.com",
        "token = ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "key AKIAABCDEFGHIJKLMNOP",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        'password = "hunter2hunter2"',
        "/home/alice/work/data.csv", r"C:\Users\Bob\Videos\a.avi", "/mnt/c/Users/carol/x", "/Users/dave/Library/x",
    ])
    kinds = [f.kind for f in P.find_sensitive(text)]
    assert kinds.count("e-mail address") == 1
    assert "github token" in kinds and "aws key" in kinds and "private key" in kinds
    assert "api key / password assignment" in kinds
    assert kinds.count("path under a user's home folder") == 4


def test_harmless_text_and_placeholder_addresses_pass():
    assert P.find_sensitive("Co-Authored-By: bot <noreply@anthropic.com>\nuser@example.com\nplain text, 12 laser steps") == []
    assert P.find_sensitive("the home folder is ~ and the video is C:\\recordings\\run1.avi") == []


def test_extra_identifiers_that_must_never_appear():
    f = P.find_sensitive("written by Isaac Oppong-Baah in 2026", extra=["isaac oppong-baah", ""])
    assert [x.kind for x in f] == ["forbidden identifier"] and f[0].line == 1


def test_findings_report_the_line():
    assert P.find_sensitive("fine\nfine\nmail me@corp.io")[0].line == 3


# ── redaction ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_redact_removes_the_home_folder_and_user_name_from_paths():
    assert P.redact(r"C:\Users\alice\Videos\run1.avi", home=r"C:\Users\alice", user="alice") == r"~\Videos\run1.avi"
    assert P.redact("/home/alice/rec/a.avi", home="/home/alice", user="alice") == "~/rec/a.avi"
    assert P.redact(r"D:\Users\bob\x.avi", home="/nowhere", user="zed") == r"D:\Users\<user>\x.avi"
    assert P.redact("/Users/carol/x", home="/nowhere", user="zed") == "/Users/<user>/x"
    assert P.redact("/data/alice/x", home="/nowhere", user="alice") == "/data/<user>/x"


def test_redact_leaves_ordinary_words_alone_and_hides_emails():
    assert P.redact("megapixel mega camera", home="/home/mega", user="mega") == "megapixel mega camera"
    assert P.redact("mail a.b@corp.io now", home="/h", user="u") == "mail <email> now"
    assert P.redact("noreply@anthropic.com", home="/h", user="u") == "noreply@anthropic.com"
    once = P.redact("/home/alice/x", home="/home/alice", user="alice")
    assert P.redact(once, home="/home/alice", user="alice") == once            # idempotent


def test_redact_uses_the_real_home_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert P.redact(f"{tmp_path}/videos/a.avi").startswith("~")


def test_short_name():
    assert P.short_name("/a/b/video.avi") == "video.avi" and P.short_name(r"C:\x\y.png".replace("\\", "/")) == "y.png"


# ── no network ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_connecting_to_the_internet_is_refused(blocked):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    with pytest.raises(P.NetworkBlocked, match="203.0.113.7"):
        s.connect(("203.0.113.7", 443))
    with pytest.raises(P.NetworkBlocked):
        s.connect_ex(("198.51.100.1", 80))
    s.close()
    with pytest.raises(P.NetworkBlocked):
        socket.create_connection(("example.com", 80), timeout=0.5)
    with pytest.raises(P.NetworkBlocked):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(P.NetworkBlocked):
        socket.gethostbyname("example.com")
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with pytest.raises(P.NetworkBlocked):
        u.sendto(b"x", ("203.0.113.7", 9))
    u.close()
    import urllib.error
    import urllib.request
    with pytest.raises(urllib.error.URLError, match="network access is switched off"):      # urllib wraps it
        urllib.request.urlopen("http://example.com", timeout=1)


def test_this_machine_still_works(blocked):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    got = []
    t = threading.Thread(target=lambda: got.append(srv.accept()[0].recv(4)))
    t.start()
    c = socket.create_connection(("127.0.0.1", port), timeout=2)
    c.sendall(b"ping"); c.close(); t.join(2); srv.close()
    assert got == [b"ping"]
    a, b = socket.socketpair()                                    # local (unix) sockets: used by Qt and multiprocessing
    a.send(b"x"); assert b.recv(1) == b"x"
    a.close(); b.close()
    assert socket.getaddrinfo("localhost", 80)


def test_unblock_restores_everything_and_blocking_twice_is_harmless():
    original = socket.socket.connect
    P.block_network(); P.block_network()
    assert P.network_blocked() and socket.socket.connect is not original
    P.unblock_network()
    assert not P.network_blocked() and socket.socket.connect is original
    P.unblock_network()                                             # and unblocking twice


def test_telemetry_switches_are_set(blocked, monkeypatch):
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    P.unblock_network(); P.block_network()
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["DO_NOT_TRACK"] == "1"
