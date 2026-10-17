# Security and privacy

This is an experimental local application. Keep the UI and inference server on
loopback. They are not authenticated multi-user network services. Do not expose
their ports to the internet or enable a public tunnel.

The UI checks local Host/Origin values and limits request bodies. Those checks
are not a substitute for authentication or operating-system access controls.
Other local processes can access local services and saved files.

Normal launches save diagnostic events. Explicit recorded sessions additionally
save voices, generated replies and request payloads. System loopback recording
can include audio from other applications. See [privacy](docs/PRIVACY.md).

Do not post credentials, private recordings, transcripts or exploit details in
public issues. Until a private reporting channel is configured, report only a
minimal non-sensitive description and request private follow-up. This repository
does not promise a response deadline or a security-audit certification.
