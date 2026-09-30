"""`client.models` and `client.storage`: the key signs, Cloud Storage serves.

The property every test here protects: the API key goes to the Intura API and
nowhere else. A signed URL is its own credential, and a key sent to
storage.googleapis.com is a key in a log we do not own.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from intura_ai import AsyncIntura, Intura, ModelDownload, NotFound, TransferError
from intura_ai._cli import main

from .conftest import API_KEY, BASE_URL, envelope, failure, inference

GCS = "https://storage.googleapis.com"
SLUG = "guardrails-pii-detection-id"
VERSION = "20260929-1"

FILES = {
    "serving.json": b'{"labels": []}',
    "onnx/classifier.onnx": b"\x08\x01" * 64,
}


def listing() -> list[dict]:
    return [
        {
            "slug": SLUG,
            "name": "PII detection",
            "tree": "guardrails",
            "endpoint": "/v1/ai/guardrails/pii-detection",
            "language": "id",
            "version": VERSION,
            "files": list(FILES),
        }
    ]


def links(token: str = "sig1", sizes: dict[str, int] | None = None) -> dict:
    return {
        "slug": SLUG,
        "version": VERSION,
        "expires_at": "2026-09-30T10:15:00+00:00",
        "files": [
            {
                "name": name,
                "size": (sizes or {}).get(name, len(body)),
                "url": f"{GCS}/intura-models/{SLUG}/{VERSION}/{name}?X-Goog-Signature={token}",
            }
            for name, body in FILES.items()
        ],
    }


class FakeCloud:
    """The Intura API and Cloud Storage behind one transport, told apart by host."""

    def __init__(self, *, expired_tokens: set[str] = frozenset(), sizes=None) -> None:
        self.requests: list[httpx.Request] = []
        self.expired = set(expired_tokens)
        self.sizes = sizes
        self.signs = 0
        self.puts: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url.startswith(GCS):
            if request.method == "PUT":
                self.puts.append(request)
                return httpx.Response(200)
            token = request.url.params.get("X-Goog-Signature")
            if token in self.expired:
                return httpx.Response(403, text="<Error>ExpiredToken</Error>")
            name = url.split(f"{VERSION}/", 1)[1].split("?", 1)[0]
            return httpx.Response(200, content=FILES[name])
        path = request.url.path
        if path == "/v1/platform/downloads/models":
            return envelope(listing())
        if path.endswith("/download"):
            if f"/{SLUG}/" not in path:
                return failure(404, "That model is not available for download.")
            self.signs += 1
            return envelope(links(f"sig{self.signs}", self.sizes))
        if path == "/v1/platform/storage/uploads":
            return envelope(
                {
                    "object": "uploads/c1/2026/09/30/abc.jpg",
                    "url": f"{GCS}/intura-uploads/uploads/c1/2026/09/30/abc.jpg?X-Goog-Signature=up",
                    "method": "PUT",
                    "headers": {"Content-Type": "image/jpeg"},
                    "expires_at": "2026-09-30T10:15:00+00:00",
                }
            )
        return envelope(inference())

    def api_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url).startswith(BASE_URL)]

    def storage_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url).startswith(GCS)]


def sync_client(cloud: FakeCloud) -> Intura:
    return Intura(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(cloud)),
        env={},
    )


def async_client(cloud: FakeCloud) -> AsyncIntura:
    return AsyncIntura(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(cloud)),
        env={},
    )


# --------------------------------------------------------------------------- #
# models


def test_list_reads_the_repository_with_the_key():
    cloud = FakeCloud()
    (model,) = sync_client(cloud).models.list()
    assert model.slug == SLUG and model.version == VERSION
    assert model.files == tuple(FILES)
    (request,) = cloud.requests
    assert request.method == "GET"
    assert request.url.path == "/v1/platform/downloads/models"
    assert request.headers["x-api-key"] == API_KEY


def test_download_writes_the_bundle_and_never_sends_the_key_to_storage(tmp_path: Path):
    cloud = FakeCloud()
    root = sync_client(cloud).models.download(SLUG, tmp_path)

    assert root == tmp_path / SLUG / VERSION
    for name, body in FILES.items():
        assert (root / name).read_bytes() == body
    assert not list(root.rglob("*.part"))

    (sign,) = cloud.api_requests()
    assert sign.method == "POST"
    assert sign.url.path == f"/v1/platform/downloads/models/{SLUG}/download"
    assert sign.headers["x-api-key"] == API_KEY

    fetched = cloud.storage_requests()
    assert len(fetched) == len(FILES)
    for request in fetched:
        assert "x-api-key" not in request.headers
        assert "authorization" not in request.headers


def test_a_second_download_keeps_what_is_already_there(tmp_path: Path):
    client = sync_client(FakeCloud())
    client.models.download(SLUG, tmp_path)

    cloud = FakeCloud()
    sync_client(cloud).models.download(SLUG, tmp_path)
    assert cloud.storage_requests() == []

    sync_client(cloud).models.download(SLUG, tmp_path, overwrite=True)
    assert len(cloud.storage_requests()) == len(FILES)


def test_an_expired_link_is_re_signed_once(tmp_path: Path):
    cloud = FakeCloud(expired_tokens={"sig1"})
    root = sync_client(cloud).models.download(SLUG, tmp_path)
    assert cloud.signs == 2
    for name, body in FILES.items():
        assert (root / name).read_bytes() == body


def test_a_link_that_stays_refused_raises(tmp_path: Path):
    cloud = FakeCloud(expired_tokens={"sig1", "sig2"})
    with pytest.raises(TransferError) as caught:
        sync_client(cloud).models.download(SLUG, tmp_path)
    assert caught.value.status_code == 403
    assert cloud.signs == 2


def test_a_short_file_is_discarded_not_kept(tmp_path: Path):
    cloud = FakeCloud(sizes={"serving.json": 999})
    with pytest.raises(TransferError, match="expected 999"):
        sync_client(cloud).models.download(SLUG, tmp_path)
    root = tmp_path / SLUG / VERSION
    assert not (root / "serving.json").exists()
    assert not (root / "serving.json.part").exists()


def test_an_unknown_slug_is_not_found(tmp_path: Path):
    with pytest.raises(NotFound):
        sync_client(FakeCloud()).models.download("no-such-model", tmp_path)


def test_a_name_that_escapes_the_directory_is_refused(tmp_path: Path, monkeypatch):
    evil = ModelDownload.from_data(
        {"slug": SLUG, "version": VERSION, "files": [{"name": "../../x", "size": 1, "url": GCS}]}
    )
    client = sync_client(FakeCloud())
    monkeypatch.setattr(client.models, "links", lambda slug: evil)
    with pytest.raises(TransferError, match="outside"):
        client.models.download(SLUG, tmp_path)


def test_links_leave_the_url_out_of_the_repr():
    download = sync_client(FakeCloud()).models.links(SLUG)
    assert "X-Goog-Signature" not in repr(download)
    assert download.size == sum(len(b) for b in FILES.values())


@pytest.mark.parametrize("slug", ["", "a/b"])
def test_a_slug_that_is_not_one_is_refused_before_a_request(slug):
    cloud = FakeCloud()
    with pytest.raises(ValueError):
        sync_client(cloud).models.links(slug)
    assert cloud.requests == []


async def test_async_download_matches(tmp_path: Path):
    cloud = FakeCloud(expired_tokens={"sig1"})
    async with async_client(cloud) as client:
        (model,) = await client.models.list()
        root = await client.models.download(model.slug, tmp_path)
    for name, body in FILES.items():
        assert (root / name).read_bytes() == body
    assert all("x-api-key" not in r.headers for r in cloud.storage_requests())


# --------------------------------------------------------------------------- #
# storage + documents


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def test_upload_puts_straight_to_storage_with_the_signed_headers_only():
    cloud = FakeCloud()
    key = sync_client(cloud).storage.upload(JPEG)
    assert key == "uploads/c1/2026/09/30/abc.jpg"

    (presign,) = cloud.api_requests()
    assert presign.url.path == "/v1/platform/storage/uploads"
    assert json.loads(presign.content) == {"content_type": "image/jpeg", "filename": "upload"}

    (put,) = cloud.puts
    assert put.content == JPEG
    assert put.headers["content-type"] == "image/jpeg"
    assert "x-api-key" not in put.headers


def test_a_refused_upload_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(GCS):
            return httpx.Response(403)
        return FakeCloud()(request)

    client = Intura(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        env={},
    )
    with pytest.raises(TransferError):
        client.storage.upload(JPEG)


def test_detect_sends_an_object_key_as_json():
    cloud = FakeCloud()
    sync_client(cloud).documents.ktp_detection.detect(
        object="uploads/c1/2026/09/30/abc.jpg", thresholds={"accept": 0.7}
    )
    (request,) = cloud.requests
    assert request.url.path == "/v1/ai/documents/ktp-detection"
    assert json.loads(request.content) == {
        "object": "uploads/c1/2026/09/30/abc.jpg",
        "thresholds": {"accept": 0.7},
    }


@pytest.mark.parametrize("kwargs", [{}, {"image": JPEG, "object": "uploads/x.jpg"}])
def test_detect_takes_exactly_one_source(kwargs):
    cloud = FakeCloud()
    with pytest.raises(ValueError, match="exactly one"):
        sync_client(cloud).documents.npwp_detection.detect(**kwargs)
    assert cloud.requests == []


# --------------------------------------------------------------------------- #
# the CLI


@pytest.fixture
def cli_cloud(monkeypatch):
    cloud = FakeCloud()
    real = Intura

    def build(*args, **kwargs):
        kwargs["api_key"] = kwargs.get("api_key") or "sk_test"
        kwargs["base_url"] = BASE_URL
        kwargs["env"] = {}
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(cloud))
        return real(*args, **kwargs)

    monkeypatch.setattr("intura_ai._cli.Intura", build)
    return cloud


def test_cli_lists_models(cli_cloud, capsys):
    assert main(["models"]) == 0
    assert SLUG in capsys.readouterr().out


def test_cli_downloads_and_prints_the_directory(cli_cloud, capsys, tmp_path: Path):
    assert main(["download", SLUG, "--dir", str(tmp_path)]) == 0
    root = Path(capsys.readouterr().out.strip())
    assert root == tmp_path / SLUG / VERSION
    assert (root / "serving.json").is_file()


def test_cli_run_takes_an_object_for_documents(cli_cloud):
    assert main(["run", "documents/ktp-detection", "--object", "uploads/x.jpg"]) == 0
    assert json.loads(cli_cloud.requests[-1].content) == {"object": "uploads/x.jpg"}
