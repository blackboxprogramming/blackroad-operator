import ssl, socket, hashlib
def assert_leaf_sha256(host: str, expected_hex_upper: str, port: int = 443):
    ctx = ssl.create_default_context()
    with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
        s.connect((host, port))
        der = s.getpeercert(True)
    got = hashlib.sha256(der).hexdigest().upper()
    if got != expected_hex_upper:
        raise RuntimeError(f"TLS pin failed for {host}: got {got}")
    return got
