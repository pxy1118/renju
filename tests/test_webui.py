import http.cookiejar
import json
import threading
import time
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
import torch

from vk.game import Game, lengths
from vk.network import Network
from vk.training import DEFAULTS, atomic_save
from vk.tunnel import hostname_ok, link_hostname, parse_log_line
from vk.webui import (LAN_SESSION_LIMIT, Sessions, Table, get_local_ipv4, make_server,
                      proxy_patterns, serve, start_tunnel, trusted_host_match)


def wait(table):
    deadline = time.monotonic() + 20
    while table.snapshot()["busy"] and time.monotonic() < deadline:
        time.sleep(.01)
    state = table.snapshot()
    assert not state["busy"]
    assert state["error"] is None
    return state


def wait_state(state_of):
    """Wait for a browser session's own table to stop thinking."""
    deadline = time.monotonic() + 20
    state = state_of()
    while state["busy"] and time.monotonic() < deadline:
        time.sleep(.01)
        state = state_of()
    assert not state["busy"], state
    assert state["error"] is None, state
    return state


@pytest.fixture
def models(tmp_path):
    torch.set_num_threads(1)
    for rule in ("freestyle", "renju"):
        cfg = dict(DEFAULTS, rule=rule, channels=4, blocks=1)
        atomic_save(dict(format=1, config=cfg, model=Network(4, 1).state_dict(), step=7),
                    tmp_path / rule / "checkpoint-00000001.pt")
    return tmp_path


@pytest.mark.parametrize("rule,color", [("freestyle", "black"), ("renju", "white"), ("renju", "black")])
def test_real_inference_move_and_undo(models, rule, color):
    table = Table(models)
    try:
        table.new(dict(rule=rule, color=color, checkpoint="latest", simulations=32))
        before = wait(table)
        if rule == "renju":
            if color == "white":
                assert before["history"] == [112]
            else:
                assert before["legal"] == [112]
                with pytest.raises(ValueError):
                    table.act(dict(id=before["id"], action=0))
        action = before["legal"][0]
        table.act(dict(id=before["id"], action=action))
        after = wait(table)
        assert len(after["history"]) == len(before["history"]) + 2
        assert after["player"] == after["human"]
        with pytest.raises(ValueError):
            table.act(dict(id=before["id"], action=action))
        undone = table.act(dict(id=before["id"]), undo=True)
        assert undone["board"] == before["board"]
        assert undone["history"] == before["history"]
        table.new(dict(rule=rule, color=color, checkpoint="latest", simulations=32))
        wait(table)
        with pytest.raises(ValueError):
            table.act(dict(id=before["id"], action=action))
    finally:
        table.pool.shutdown()


def test_explicit_checkpoint_selection(models):
    """A named historical checkpoint can be replayed, not only latest/best."""
    import time
    for round_id in (2, 3):
        atomic_save(dict(format=1, config=dict(DEFAULTS, rule="freestyle", channels=4, blocks=1),
                         model=Network(4, 1).state_dict(), step=round_id * 100),
                    models / "freestyle" / f"checkpoint-{round_id:08d}.pt")
        time.sleep(0.01)
    table = Table(models)
    try:
        listing = table.models()["freestyle"]
        assert listing["latest"] == "checkpoint-00000003.pt"
        recent = [item["name"] for item in table.recent("freestyle")]
        assert recent[0] == "checkpoint-00000003.pt" and "checkpoint-00000001.pt" in recent
        assert all(item["mb"] > 0 and item["mtime"] for item in table.recent("freestyle"))
        oldest = recent[-1]
        assert oldest == "checkpoint-00000001.pt"  # fixture checkpoint, written first
        table.new(dict(rule="freestyle", color="black", checkpoint=oldest, simulations=32))
        state = wait(table)
        assert state["model"]["file"] == oldest
        assert state["model"]["step"] == 7
        assert state["model"]["latest"] == "checkpoint-00000003.pt"
        # Names that cannot be a file inside the rule folder are refused up front.
        for name in ("../secrets.pt", "artifacts/x.pt", "sub\\x.pt"):
            with pytest.raises(ValueError, match="无效"):
                table.new(dict(rule="freestyle", color="black", checkpoint=name))
    finally:
        table.pool.shutdown()


def test_discovers_and_loads_pretrain_and_hybrid_directories(models):
    pretrain = models / "freestyle-pretrain-v2"
    state = dict(format=1, config=dict(DEFAULTS, rule="freestyle", channels=4, blocks=1),
                 model=Network(4, 1).state_dict(), step=20000)
    atomic_save(state, pretrain / "best.pt")
    table = Table(models)
    try:
        listing = table.models()["freestyle"]
        assert listing["best"] == "freestyle-pretrain-v2/best.pt"
        recent = table.recent("freestyle")
        assert any(item["name"] == "freestyle-pretrain-v2/best.pt" and
                   item["run"] == "freestyle-pretrain-v2" for item in recent)
        table.new(dict(rule="freestyle", color="black", checkpoint="best", simulations=32))
        loaded = wait(table)
        assert loaded["model"]["file"] == "freestyle-pretrain-v2/best.pt"
        assert loaded["model"]["step"] == 20000
    finally:
        table.pool.shutdown()


def test_missing_file_reports_through_error_channel(models):
    """A pruned or stale name is reported to the user once the load fails."""
    table = Table(models)
    try:
        table.new(dict(rule="freestyle", color="black",
                       checkpoint="checkpoint-00009999.pt", simulations=32))
        deadline = time.monotonic() + 20
        while table.snapshot()["busy"] and time.monotonic() < deadline:
            time.sleep(.01)
        state = table.snapshot()
        assert not state["busy"]
        assert state["error"] and "失败" in state["error"]
        with pytest.raises(ValueError):
            table.act(dict(id=state["id"], action=0))
        # Starting a valid game afterwards clears the failed state.
        table.new(dict(rule="freestyle", color="black", checkpoint="latest", simulations=32))
        assert wait(table)["error"] is None
    finally:
        table.pool.shutdown()


def winning_move(state):
    """First legal action that completes five in a row for the side to move."""
    board = np.array(state["board"], np.int8)
    for action in state["legal"]:
        probe = board.copy()
        probe[action] = state["player"]
        if max(lengths(probe, action, state["player"])) >= 5:
            return action
    return None


def test_human_win_is_reported_immediately(models):
    """The winning move must not leave the table 'thinking' or blame the wrong side."""
    table = Table(models)
    try:
        state = table.new(dict(rule="freestyle", color="black", checkpoint="latest", simulations=32))
        state = wait(table)
        assert state["history"] == []
        # Inject a position whose win is already forced. A competent tactical
        # opponent no longer permits the old test to manufacture an open four.
        with table.lock:
            table.game = Game("freestyle")
            table.game.board[105:109] = 1
            table.game.board[[0, 2, 4, 6]] = -1
            table.game.player = 1
            table.history = [105, 0, 106, 2, 107, 4, 108, 6]
            table.refresh()
            state = table.snapshot()
        state = table.act(dict(id=state["id"], action=109))
        assert state["busy"] is False, "a finished game must not stay busy"
        assert state["winner"] == 1 == state["human"], state
        assert state["legal"] == [] and state["error"] is None
        # The page paints its side indicator from these two fields, so a finished
        # game must not look like "white to move" next to "you won".
        assert state["player"] == -state["winner"], "winner's opponent is 'to move'"
        assert state["human"] == state["winner"]
        black = np.array(state["board"], np.int8)
        assert max(max(lengths(black, a, 1)) for a in range(225)
                   if state["board"][a] == 1) == 5
    finally:
        table.pool.shutdown()


def test_missing_model_and_busy(tmp_path):
    table = Table(tmp_path)
    try:
        with pytest.raises(ValueError, match="检查点"):
            table.new(dict(rule="renju", color="black", checkpoint="latest"))
        table.busy = True
        with pytest.raises(ValueError, match="思考"):
            table.new(dict(rule="freestyle", color="black", checkpoint="latest"))
    finally:
        table.pool.shutdown()


def test_http_local_protection_and_validation(models):
    server = make_server(0, models, host=get_local_ipv4() or "127.0.0.1")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{server.server_address[0]}:{server.server_port}"
    try:
        with urlopen(base + "/api/config") as response:
            config = json.load(response)
        assert config["models"]["renju"]["latest"].startswith("checkpoint-")
        assert config["models"]["renju"]["best"] is None
        assert [item["name"] for item in config["recent"]["renju"]] == [config["models"]["renju"]["latest"]]
        assert config["recent"]["freestyle"][0]["mb"] > 0
        # An unshared server keeps its original shape: no sessions, no invite.
        assert "share" not in config and "invite" not in config
        with urlopen(base) as response:
            assert "棋间" in response.read().decode()
        for headers, payload, status in [({}, {}, 403), ({"X-Renju-Token": config["token"]}, [], 400),
                                         ({"X-Renju-Token": config["token"], "Host": "evil.example"}, {}, 403)]:
            with pytest.raises(HTTPError) as err:
                urlopen(Request(base + "/api/new", data=json.dumps(payload).encode(), headers=headers))
            assert err.value.code == status
    finally:
        server.shutdown()
        server.server_close()
        server.table.pool.shutdown()
        thread.join()


class Browser:
    """One guest: its own cookie jar, so two of them are two separate players."""

    def __init__(self, base):
        self.base = base
        self.jar = http.cookiejar.CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.jar))
        self.token = None

    def get(self, path, authority=""):
        request = Request(self.base + path)
        if authority:
            request.add_header("Host", authority)
        return self.opener.open(request, timeout=15)

    def form(self, path, fields):
        """Submit the small password form the way a browser does."""
        body = urlencode(fields).encode()
        return self.opener.open(Request(self.base + path, data=body, headers={
            "Content-Type": "application/x-www-form-urlencoded"}), timeout=15)

    def get_host(self, path, authority, forwarded=""):
        """GET while presenting ``authority``, the way a proxy forwards it."""
        request = Request(self.base + path)
        request.add_header("Host", authority)
        if forwarded:
            request.add_header("X-Forwarded-Proto", forwarded)
        return self.opener.open(request, timeout=15)

    def api(self, path, payload):
        """POST with whatever token this browser last read from /api/config."""
        if self.token is None:
            self.token = json.load(self.get("/api/config"))["token"]
        request = Request(self.base + path, data=json.dumps(payload).encode(),
                          headers={"X-Renju-Token": self.token})
        return json.load(self.opener.open(request, timeout=15))

    def join(self, key):
        with self.get(f"/?k={key}") as response:
            assert response.status == 200 and response.geturl() == self.base + "/"
        return json.load(self.get("/api/state"))

    def state(self):
        return json.load(self.get("/api/state"))


@pytest.fixture
def shared(models):
    """A shared server bound to the address a guest would really connect to.

    The requests follow the bind address instead of loopback, because a socket
    bound to one interface does not answer on another. On a machine with no LAN
    address there is nothing to share with, so those tests skip.
    """
    host = get_local_ipv4()
    if not host:
        pytest.skip("no LAN address on this machine")
    server = make_server(0, models, host=host, share=True, max_sessions=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://{host}:{server.server_port}", Browser
    finally:
        server.shutdown()
        server.server_close()
        server.sessions.shutdown()
        thread.join()


def test_share_gives_each_guest_a_private_table(shared):
    server, base, Browser = shared
    assert server.invite(server.server_port).startswith(f"http://{get_local_ipv4() or '127.0.0.1'}:")
    first, second = Browser(base), Browser(base)
    # The invite key buys a cookie and disappears from the address bar.
    assert first.join(server.invite_key)["id"] is None
    assert second.join(server.invite_key)["id"] is None
    with pytest.raises(HTTPError) as err:
        Browser(base).get("/")  # a stranger without the key or a cookie
    assert err.value.code == 403

    started = first.api("/api/new", dict(rule="freestyle", color="black", checkpoint="latest", simulations=32))
    other = second.api("/api/new", dict(rule="freestyle", color="black", checkpoint="latest", simulations=32))
    assert started["id"] != other["id"], "each guest owns a table instead of sharing one"
    assert started["sessions"] == 2 and other["sessions"] == 2
    wait_state(first.state)
    assert first.state()["history"] == []
    # A move on one board must not appear on the other.
    first.api("/api/move", dict(id=started["id"], action=112))
    assert first.state()["board"][112] == 1
    assert second.state()["board"][112] == 0
    assert second.state()["history"] == []
    # Another browser's session id is not a substitute for knowing the table.
    with pytest.raises(HTTPError):
        second.api("/api/move", dict(id=started["id"], action=113))


def test_share_capacity_is_enforced_and_released(models):
    """The cap is a resource limit: it refuses, never evicts, and frees on leave."""
    host = get_local_ipv4()
    if not host:
        pytest.skip("no LAN address on this machine")
    # Seeded small on purpose, so the test does not build ten networks to prove
    # the mechanism; `test_the_default_share_capacity_is_ten` pins the default.
    server = make_server(0, models, host=host, share=True, max_sessions=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{server.server_port}"
    try:
        guests = [Browser(base) for _ in range(3)]
        for guest in guests:
            guest.join(server.invite_key)
        late = Browser(base)
        with pytest.raises(HTTPError) as err:
            late.get(f"/?k={server.invite_key}")
        assert err.value.code == 429, "the guest past the cap is told the share is full"
        assert "名额已满" in err.value.read().decode()
        assert server.sessions.count() == 3
        # Leaving hands the slot over instead of ending somebody else's game.
        assert guests[0].api("/api/leave", {}) == {"left": True, "sessions": 2}
        with pytest.raises(HTTPError) as err:
            guests[0].get("/api/state")
        assert err.value.code == 403
        assert late.join(server.invite_key)["id"] is None
        assert server.sessions.count() == 3
    finally:
        server.shutdown()
        server.server_close()
        server.sessions.shutdown()
        thread.join()


def test_the_default_share_capacity_is_ten(models):
    """Ten tables is what `--share` and `--public` hand out by default.

    Exercised through `Sessions` itself so the boundary is real; the tables
    share the fixture's catalog, so ten of them are cheap to stand up.
    """
    assert LAN_SESSION_LIMIT == 10
    sessions = Sessions(LAN_SESSION_LIMIT, root=models)
    try:
        assert sessions.maximum == 10
        opened = [sessions.open() for _ in range(10)]
        assert all(opened), "the tenth table must still be admitted"
        assert sessions.count() == 10
        # Ten guests means ten distinct tables, never a shared one.
        assert len({id(table) for _, table in opened}) == 10
        assert sessions.open() is None, "the eleventh guest must be refused"
        assert sessions.count() == 10
    finally:
        sessions.shutdown()
    # The number the page is told is the number the server enforces.
    server = make_server(0, models, host="127.0.0.1", share=True)
    try:
        assert server.sessions.maximum == 10
    finally:
        server.server_close()
        server.sessions.shutdown()


def test_share_accepts_the_lan_host_and_refuses_others(shared):
    server, base, Browser = shared
    lan = f"{get_local_ipv4() or '127.0.0.1'}:{server.server_port}"
    assert lan in server.fence
    guest = Browser(base)
    with pytest.raises(HTTPError) as err:
        guest.get("/", authority="evil.example")
    assert err.value.code == 403
    assert json.loads(err.value.read())["error"] == "Local access only"
    # The host a guest really types is fenced in, not out: an admitted browser
    # using that authority keeps reaching its own table.
    guest.join(server.invite_key)
    with guest.get("/api/state", authority=lan) as response:
        assert response.status == 200
    with guest.get("/", authority=lan) as response:
        assert "棋间" in response.read().decode()


def test_all_interfaces_bind_serves_the_lan_guest(models):
    """`--host 0.0.0.0` is the option that makes one link work for everybody."""
    host = get_local_ipv4()
    if not host:
        pytest.skip("no LAN address on this machine")
    server = make_server(0, models, host="0.0.0.0", share=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{server.server_port}"
    try:
        assert base in server.invite(server.server_port)
        assert f"{host}:{server.server_port}" in server.fence
        guest = Browser(base)
        assert guest.join(server.invite_key)["id"] is None
        with pytest.raises(HTTPError) as err:
            guest.get("/api/state", authority=f"192.0.2.7:{server.server_port}")
        assert err.value.code == 403
    except URLError:
        pytest.skip("the all-interfaces bind is not reachable from this host")
    finally:
        server.shutdown()
        server.server_close()
        server.sessions.shutdown()
        thread.join()


def test_trusted_host_supports_a_subdomain_pattern():
    """A rotating quick-tunnel hostname stays reachable via `.trycloudflare.com`."""
    trusted = [".trycloudflare.com", "board.example.test"]
    for authority, ok in [("root-frog-motion-lookup.trycloudflare.com", True),
                          ("root-frog-motion-lookup.trycloudflare.com:443", True),
                          ("trycloudflare.com", True),
                          ("notexample.test", False),
                          ("board.example.test", False),   # exact names live in the set
                          ("other.example.test", False)]:
        assert trusted_host_match(authority, trusted) is ok, authority
    assert trusted_host_match("root-frog-motion-lookup.trycloudflare.com", []) is False


def test_share_accepts_a_trusted_proxy_host(models):
    """A tunnel forwards the bare hostname, so --trusted-host must admit it."""
    host = get_local_ipv4()
    if not host:
        pytest.skip("no LAN address on this machine")
    server = make_server(0, models, host=host, share=True,
                         trusted_hosts=[".example.test"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{server.server_port}"
    try:
        guest = Browser(base)
        guest.join(server.invite_key)
        # cloudflared forwards "host" with no port; a proxy on a named port keeps it.
        for authority in ("board.example.test", f"board.example.test:{server.server_port}"):
            with guest.get("/api/state", authority=authority) as response:
                assert response.status == 200, authority
        # A proxy that says it terminated TLS gets an https link; one that does
        # not is taken at its word about the scheme. A direct LAN visitor keeps
        # the address the operator printed at startup.
        request = Request(base + "/api/config")
        request.add_header("Host", "board.example.test")
        request.add_header("X-Forwarded-Proto", "https")
        config = json.load(guest.opener.open(request, timeout=15))
        assert config["invite"] == f"https://board.example.test/?k={server.invite_key}"
        assert json.load(guest.get("/api/config", authority="board.example.test"))["invite"] == \
            f"http://board.example.test/?k={server.invite_key}"
        assert json.load(guest.get("/api/config"))["invite"].startswith("http://")
        with pytest.raises(HTTPError) as err:
            guest.get("/api/state", authority="board.example.org")   # a lookalike domain
        assert err.value.code == 403
        with pytest.raises(HTTPError) as err:
            guest.get("/api/state", authority="evil.test")
        assert err.value.code == 403
        # A host named exactly (no leading dot) is fenced in on this port.
        exact = make_server(0, models, host=host, share=True, trusted_hosts=["board.example.test"])
        thread2 = threading.Thread(target=exact.serve_forever, daemon=True)
        thread2.start()
        try:
            assert "board.example.test" in exact.fence
            assert f"board.example.test:{exact.server_port}" in exact.fence
        finally:
            exact.shutdown()
            exact.server_close()
            exact.sessions.shutdown()
            thread2.join()
    finally:
        server.shutdown()
        server.server_close()
        server.sessions.shutdown()
        thread.join()


def test_share_password_page_admits_a_guest(models):
    host = get_local_ipv4()
    if not host:
        pytest.skip("no LAN address on this machine")
    # A password containing & and = is legal because the form posts it in the body.
    server = make_server(0, models, host=host, share=True, password="open&ses=ame")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{server.server_port}"
    try:
        guest = Browser(base)
        # Without the key the page offers the password form instead of a 403.
        with guest.get("/") as response:
            page = response.read().decode()
        assert response.status == 200 and 'name="p"' in page and 'method="POST"' in page
        with pytest.raises(HTTPError) as err:
            guest.form("/", {"p": "wrong"})
        assert err.value.code == 400 and "口令不正确" in err.value.read().decode()
        assert server.sessions.count() == 0
        with guest.form("/", {"p": "open&ses=ame"}) as response:
            assert response.status == 200
        assert server.sessions.count() == 1
        assert json.load(guest.get("/api/config"))["password"] is True
        with pytest.raises(HTTPError) as err:
            Browser(base).get("/app.js")
        assert err.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        server.sessions.shutdown()
        thread.join()


# The startup banner of a real cloudflared 2026.9.0 quick tunnel, verbatim. The
# first https link in it is the terms of service, not the tunnel, which is what
# makes "take the first URL you see" the wrong way to find the public address.
CLOUDFLARED_BANNER = [
    "INF Thank you for trying Cloudflare Tunnel. Doing so, without a Cloudflare account, "
    "is a quick way to experiment and try it out. However, be aware that these account-less "
    "Tunnels have no uptime guarantee, are subject to the Cloudflare Online Services Terms "
    "of Use (https://www.cloudflare.com/website-terms/), and Cloudflare reserves the right "
    "to investigate your use of Tunnels for violations of such terms. If you intend to use "
    "Tunnels in production you should use a pre-created named tunnel by following: "
    "https://developers.cloudflare.com/cloudflare-one/connections/connect-apps",
    "INF Requesting new quick Tunnel on trycloudflare.com...",
    "INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be "
    "reachable):  |",
    "INF |  https://root-frog-motion-lookup.trycloudflare.com                        |",
    "INF Registered tunnel connection connIndex=0 connection=22285ef5 ip=2606:4700:a8::10 "
    "location=sjc10 protocol=quic",
]
BANNER_HOST = "root-frog-motion-lookup.trycloudflare.com"


class FakeTunnel:
    """A tunnel whose hostname comes from the real banner parser."""

    def __init__(self, banner=CLOUDFLARED_BANNER):
        self.hostname = next(name for name in map(parse_log_line, banner) if name)
        self.link = f"https://{self.hostname}"

    def start(self):
        return self.link


def test_quick_tunnel_url_is_read_from_the_banner_not_the_first_link():
    """The terms-of-service link must never become the operator's invite address."""
    found = [parse_log_line(line) for line in CLOUDFLARED_BANNER]
    assert found[:3] == [None, None, None], found[:3]
    assert found[3] == BANNER_HOST
    assert found[4] is None
    assert [name for name in found if name] == [BANNER_HOST]
    # A bare "trycloudflare.com" is the service, not a tunnel on it.
    assert parse_log_line("INF Requesting new quick Tunnel on trycloudflare.com...") is None


def test_tunnel_hostname_validation_fences_what_can_be_trusted():
    assert hostname_ok(BANNER_HOST)
    assert hostname_ok("a-b-c-d.trycloudflare.com")
    for bad in ("", "bad host.com", "evil.com:443", "[::1]", "a" * 300 + ".com", "-lead.com",
                "trail-.com", "double..dot.com"):
        assert not hostname_ok(bad), bad
    assert link_hostname(f"https://{BANNER_HOST}") == BANNER_HOST
    assert link_hostname(f"https://{BANNER_HOST}/") == BANNER_HOST
    assert link_hostname(f"https://{BANNER_HOST.upper()}") == BANNER_HOST
    assert link_hostname(f"https://user:pw@{BANNER_HOST}") == BANNER_HOST
    # Not the tunnel: wrong scheme, or a path that would land outside the board.
    assert link_hostname(f"http://{BANNER_HOST}") is None
    assert link_hostname(f"https://{BANNER_HOST}/elsewhere") is None
    assert link_hostname(None) is None


def test_a_tunnel_admits_its_own_hostname_without_a_trusted_host():
    """`--public` must not need `--trusted-host` typed by hand."""
    tunnel = FakeTunnel()
    # A server nobody pointed at a proxy admits no wildcard at all...
    assert proxy_patterns([], None) == []
    # ...an operator's own pattern always applies...
    assert proxy_patterns([".example.test"], None) == [".example.test"]
    # ...and a live tunnel brings its own tree alongside it.
    assert proxy_patterns([], tunnel) == [".trycloudflare.com"]
    assert proxy_patterns([".example.test"], tunnel) == [".example.test", ".trycloudflare.com"]
    assert trusted_host_match(BANNER_HOST, proxy_patterns([], tunnel))
    # An unrelated tree still does not match, tunnel or not. (`evil.example.test`
    # would match: it genuinely is a subdomain of the tree that was named.)
    assert not trusted_host_match("evil.example.org", proxy_patterns([], tunnel))
    assert not trusted_host_match("evil.example.org", proxy_patterns([".example.test"], None))


def test_start_tunnel_publishes_the_hostname_but_never_the_fence(monkeypatch):
    """The fence means "this machine reached directly", so a tunnel stays out of it.

    A tunnel name in the fence would make every guest look like a LAN visitor and
    hand them the operator's LAN link instead of the public one.
    """
    monkeypatch.setattr("vk.webui.Tunnel", lambda *a, **k: FakeTunnel())
    server = type("S", (), {"fence": {"127.0.0.1:8765", "192.0.2.5:8765"},
                            "tunnel_lock": threading.Lock(), "tunnel": None,
                            "tunnel_host": None})()
    link = start_tunnel(server, 8765)
    assert link == f"https://{BANNER_HOST}"
    assert server.tunnel_host == BANNER_HOST
    assert server.tunnel is not None
    assert BANNER_HOST not in server.fence


class FakeServer:
    """Just enough server for `serve` to start, stop, and be asked to clean up."""

    def __init__(self):
        self.server_port = 8765
        self.sessions = None
        self.invite_key = "test-invite-key"
        self.table = type("T", (), {"pool": type("P", (), {"shutdown": lambda self, **k: None})()})()
        self.tunnel = None
        self.stopped = False
        self.closed = False

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        self.closed = True


def test_serve_closes_the_tunnel_when_the_operator_stops_it(monkeypatch):
    """Ctrl+C must take the public route down with it, never leave it open."""
    server = FakeServer()
    tunnel = type("K", (), {"stop": lambda self: setattr(server, "stopped", True)})()
    monkeypatch.setattr("vk.webui.make_server", lambda *a, **k: server)

    def fake_start(server_, port, executable=None, timeout=40.0):
        server_.tunnel = tunnel
        server_.tunnel_host = BANNER_HOST
        return f"https://{BANNER_HOST}"

    monkeypatch.setattr("vk.webui.start_tunnel", fake_start)
    serve(8765, "runs", open_browser=False, public=True)

    assert server.stopped, "the tunnel outlived the server"
    assert server.closed

