import socket

# BlackRoad Pi cluster nodes — all routing goes through these
PI_NODES = {
    "alice":   "192.168.4.49",
    "aria":    "192.168.4.38",
    "lucidia": "192.168.4.99",
}

def pi_preflight(timeout: float = 2.0):
    """Verify at least one Pi node is reachable before starting."""
    for name, ip in PI_NODES.items():
        try:
            s = socket.create_connection((ip, 8000), timeout=timeout)
            s.close()
            return name, ip
        except OSError:
            continue
    raise RuntimeError("No Pi cluster nodes reachable: " + ", ".join(PI_NODES))
