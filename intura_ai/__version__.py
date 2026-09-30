"""Single source of truth for the package version.

Read by hatchling at build time (`[tool.hatch.version]`) and by
`intura_ai.__version__` at runtime, so there is one string to bump and the wheel
can never disagree with what the installed package reports.

2.1.0 adds `client.models` — list and download the builds the hosted API
serves, to run them yourself. The key signs short-lived Cloud Storage links and
the files come straight from the bucket; the key is never sent there. It also
catches up with the document endpoints: `verdict.extraction` reads the card's
values (plus a masked copy), and `detect(object=...)` takes a key from the new
`client.storage.upload`. And it drops `model` from every request and every result: the API stopped
accepting one and stopped reporting one on 2026-09-24. Passing `model=` is now a
`TypeError` rather than a silently ignored argument, and `Inference.model` is
gone rather than permanently `None` — both of which are louder than the failures
the alternatives would have produced.

2.0.0 is the API client. 1.x downloaded ONNX weights and ran them in your
process; that channel no longer exists — `intura-ai-labs` publishes builds to the
bucket `intura-ai-api` serves from, and nothing is mirrored for a local runtime
any more. Versions up to 0.0.7.1 were a third product, a LangChain
experimentation client, and share no API with either. Both migrations are in the
README.
"""

__version__ = "2.1.1"
