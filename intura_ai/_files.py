"""Moving bytes to and from Cloud Storage, on a link the API signed.

Two transfers go around the API rather than through it:

    client.models.download(slug)   the API signs a GET per file; each file comes
                                   straight from the model bucket
    client.storage.upload(file)    the API signs a PUT; the file goes straight
                                   to the bucket and only its key comes back

The API key authenticates the *signing* call and nothing else. It is never sent
to a signed URL: the signature is the credential there, and a key in a request
to storage.googleapis.com is a key in somebody else's access log. That is why
every request in this module is built without `_kwargs`, which is where the
key is added.

Nothing here is retried blindly. A download that failed part-way leaves a
`.part` file and the next call starts that file again; a link that expired
mid-bundle is re-signed once (a signed URL lives fifteen minutes, and a large
bundle on a slow link can outlast that).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from ._errors import TransferError
from .types import ModelDownload, ModelFile

CHUNK_BYTES = 1 << 20

# What the bucket answers when a signature has expired or no longer matches.
EXPIRED_STATUS = frozenset({400, 401, 403})


def bundle_dir(dest: str | os.PathLike[str], links: ModelDownload) -> Path:
    """`<dest>/<slug>/<version>` — one directory per build, so two versions of
    one model sit side by side and the path says which is which."""
    return Path(dest).expanduser() / links.slug / links.version


def target_path(root: Path, name: str) -> Path:
    """Where one file of a bundle lands, refusing any name that would escape it.

    The names come from the bucket listing, which we trust; checking costs
    nothing and a download that can write outside the directory it was given
    is not one anybody should run.
    """
    rel = PurePosixPath(name)
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        raise TransferError(f"refusing to write {name!r} outside the model directory.")
    return root.joinpath(*rel.parts)


def is_current(target: Path, file: ModelFile) -> bool:
    """Already downloaded: present at the size the bucket reports."""
    return target.is_file() and target.stat().st_size == file.size


def _check_size(part: Path, file: ModelFile) -> None:
    got = part.stat().st_size
    if got != file.size:
        part.unlink(missing_ok=True)
        raise TransferError(
            f"{file.name}: received {got} bytes, expected {file.size}. The "
            "connection dropped mid-file; run the download again."
        )


def refused(file: ModelFile, status: int) -> TransferError:
    return TransferError(
        f"Cloud Storage refused {file.name} with HTTP {status}. Run the download "
        "again; if it keeps failing, contact support with the model slug.",
        status_code=status,
    )


def _unreachable(file: ModelFile, exc: Exception) -> TransferError:
    return TransferError(
        f"could not download {file.name} from Cloud Storage: {exc}. Check network "
        "egress to storage.googleapis.com and run the download again."
    )


def fetch(http: httpx.Client, file: ModelFile, target: Path) -> int | None:
    """Stream one file to `target`. None on success, the status if the link was
    refused as expired (the caller re-signs), and raises for anything else."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with http.stream("GET", file.url) as response:
            if response.status_code in EXPIRED_STATUS:
                return response.status_code
            if response.status_code != 200:
                raise refused(file, response.status_code)
            with part.open("wb") as out:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    out.write(chunk)
    except httpx.HTTPError as exc:
        raise _unreachable(file, exc) from exc
    _check_size(part, file)
    os.replace(part, target)
    return None


async def afetch(http: httpx.AsyncClient, file: ModelFile, target: Path) -> int | None:
    """`fetch`, awaited. The disk writes stay synchronous: a megabyte chunk to a
    local file is not what an event loop waits on."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        async with http.stream("GET", file.url) as response:
            if response.status_code in EXPIRED_STATUS:
                return response.status_code
            if response.status_code != 200:
                raise refused(file, response.status_code)
            with part.open("wb") as out:
                async for chunk in response.aiter_bytes(CHUNK_BYTES):
                    out.write(chunk)
    except httpx.HTTPError as exc:
        raise _unreachable(file, exc) from exc
    _check_size(part, file)
    os.replace(part, target)
    return None


def pending(root: Path, files: Iterable[ModelFile], overwrite: bool) -> list[ModelFile]:
    return [f for f in files if overwrite or not is_current(target_path(root, f.name), f)]


def put_request(signed: dict[str, Any], data: bytes) -> dict[str, Any]:
    """The request a signed upload URL was signed for — method and headers exactly
    as the API returned them, because both are part of the signature."""
    return {
        "method": str(signed.get("method") or "PUT"),
        "url": str(signed["url"]),
        "headers": {str(k): str(v) for k, v in (signed.get("headers") or {}).items()},
        "content": data,
    }


def check_put(response: httpx.Response) -> None:
    if response.status_code >= 300:
        raise TransferError(
            f"Cloud Storage refused the upload with HTTP {response.status_code}. "
            "Ask for a new upload link and try again.",
            status_code=response.status_code,
        )


__all__ = [
    "afetch",
    "bundle_dir",
    "check_put",
    "fetch",
    "is_current",
    "pending",
    "put_request",
    "refused",
    "target_path",
]
