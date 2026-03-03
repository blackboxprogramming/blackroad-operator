import socket

def check_pi_node(ip: str, port: int = 8000, timeout: float = 2.0) -> bool:
    """Return True if a Pi cluster node is accepting connections."""
    try:
        s = socket.create_connection((ip, port), timeout=timeout)
        s.close()
        return True
    except OSError:
        return False
